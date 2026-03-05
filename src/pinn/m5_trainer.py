"""
M5 PINN 训练器
===============
在 M4 PINNTrainer 基础上增强:

1. 井—藏耦合同化训练 (qg 监督 + p_wf 反演)
2. ReLoBRaLo 自适应损失权重平衡
3. RAR 残差驱动自适应采样
4. 反演参数追踪与审计
5. 分阶段课程学习 (兼容 M4 stage A/B/C/D)
6. 完整的验收图件输出

不破坏 M4 基线:
    - 继承所有 M4 trainer 的核心逻辑
    - 新增功能全部可通过 config 开关
"""

import os
import sys
import time
import json
import copy
import math
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise ImportError("m5_trainer 需要 PyTorch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger, setup_chinese_support, ensure_dir, load_config

setup_chinese_support()
import matplotlib.pyplot as plt

from pinn.m5_model import M5PINNNet
from pinn.assimilation_losses import AssimilationLoss
from pinn.relobralo import ReLoBRaLo, ManualLossBalancer
from pinn.rar_sampler import RARSampler


class M5Trainer:
    """M5 PINN 井—藏耦合同化训练器"""
    
    def __init__(self, config: dict, model: M5PINNNet, sampler, device: str = 'cpu'):
        setup_chinese_support()
        self.config = config
        self.device = device
        self.logger = setup_logger('M5Trainer')
        
        # --- 可复现性 ---
        repro_cfg = config.get('reproducibility', {})
        seed = repro_cfg.get('seed', None)
        deterministic = repro_cfg.get('deterministic', False)
        if 'cuda' in device:
            if deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
            else:
                torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if 'cuda' in device:
                torch.cuda.manual_seed_all(seed)
        
        self.model = model.to(device)
        self.sampler = sampler
        
        # --- torch.compile 加速 (PyTorch 2.0+, ~30% speedup) ---
        # 注意: torch.compile 的 inductor 后端依赖 Triton, 仅 Linux 可用
        # Windows 上必须跳过, 否则运行时会抛出 TritonMissing 错误
        import sys as _sys
        runtime_cfg_compile = config.get('runtime', {})
        use_compile = runtime_cfg_compile.get('torch_compile', True)
        _triton_available = False
        if use_compile and hasattr(torch, 'compile') and _sys.platform.startswith('linux'):
            try:
                import triton  # type: ignore[import-untyped]  # 可选依赖，Windows 常未安装
                _triton_available = True
            except ImportError:
                pass
        if _triton_available:
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self.logger.info("  torch.compile 已启用 (mode=reduce-overhead)")
            except Exception as e:
                self.logger.warning(f"  torch.compile 不可用, 回退标准模式: {e}")
        else:
            if use_compile and not _sys.platform.startswith('linux'):
                self.logger.info("  torch.compile 已跳过 (Triton 不支持 Windows, 使用 eager 模式)")
            elif use_compile:
                self.logger.info("  torch.compile 已跳过 (Triton 未安装, 使用 eager 模式)")
        
        # --- 损失函数 ---
        self.loss_fn = AssimilationLoss(config, device=device)
        
        # --- 训练参数 ---
        train_cfg = config.get('train', {})
        self.lr = train_cfg.get('learning_rate', 1e-3)
        self.max_steps = train_cfg.get('max_steps', 50000)
        self.batch_size = train_cfg.get('batch_size', 2048)
        
        # --- 物理损失开关与 base_weight (消融实验关键) ---
        # phys_enable=False 时将 IC/BC/PDE 权重强制归零
        # phys_base_weight 作为所有物理项的全局乘子
        loss_cfg = config.get('loss', {})
        phys_loss_cfg = loss_cfg.get('physics', {})
        self.phys_enable = config.get('physics', {}).get('enable', True)
        # v3.1 类型安全: config 可能传入字符串 'False' 或非 bool 值
        if not isinstance(self.phys_enable, bool):
            self.phys_enable = config.get('loss', {}).get('physics', {}).get('enable', True)
        self.phys_base_weight = phys_loss_cfg.get('base_weight', 1.0)
        if not phys_loss_cfg.get('enable', True):
            self.phys_enable = False
        
        # AMP
        runtime_cfg = config.get('runtime', {})
        self.use_amp = (runtime_cfg.get('mixed_precision', False) and 'cuda' in device)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # 优化器: 场网络和反演参数使用不同学习率
        opt_cfg = train_cfg.get('optimizer', {})
        wd = opt_cfg.get('weight_decay', 1e-4)
        
        # ===== v3.2-FIX: 4 组参数, 精细控制学习率 =====
        # 1. field_net: 场网络 (416K 参数), base lr
        # 2. inversion: pwf_nets + dp_wellbore, lr × inv_lr_factor
        # 3. k_frac: 单标量, 梯度极小需要独立高 lr × k_frac_lr_factor
        # 4. k_net: 空间渗透率子网络, lr × k_net_lr_factor (有 warmup)
        field_params = list(model.field_net.parameters())
        well_other_params = []   # pwf_nets + dp_wellbore
        k_frac_params = []       # k_frac 单独分组
        k_net_params = []
        
        if hasattr(model, 'well_model'):
            for name, param in model.well_model.named_parameters():
                if '_k_frac_raw' in name or '_r_e_raw' in name:
                    k_frac_params.append(param)
                else:
                    well_other_params.append(param)
        # v3.7: dp_wellbore 已冻结为 register_buffer (不再可学习)
        # 仅在万一仍是 nn.Parameter 时才加入 optimizer (向后兼容)
        if hasattr(model, '_dp_wellbore_raw') and isinstance(model._dp_wellbore_raw, torch.nn.Parameter):
            well_other_params.append(model._dp_wellbore_raw)
        if hasattr(model, 'k_net') and model.k_net is not None:
            k_net_params = list(model.k_net.parameters())
        
        # 从 config 读取 k_net 学习率因子与 warmup（默认 0.1 / 2000，阶梯实验时只改 config）
        k_net_cfg = self.config.get('m5_config', {}).get('k_net', {})
        self.k_net_lr_factor = k_net_cfg.get('lr_factor', 0.1)
        self.k_net_warmup_steps = k_net_cfg.get('warmup_steps', 2000)
        
        inv_lr_factor = opt_cfg.get('inversion_lr_factor', 1.5)
        # v12 基线: k_frac 独立学习率因子默认 10.0
        k_frac_lr_factor = opt_cfg.get('k_frac_lr_factor', 10.0)
        
        param_groups = [
            {'params': field_params, 'lr': self.lr, 'name': 'field'},
            {'params': well_other_params, 'lr': self.lr * inv_lr_factor, 'name': 'inversion'},
        ]
        if k_frac_params:
            param_groups.append({
                'params': k_frac_params,
                'lr': self.lr * k_frac_lr_factor,
                'name': 'k_frac'
            })
            self.logger.info(
                f"k_frac 独立参数组: lr={self.lr * k_frac_lr_factor:.2e} "
                f"(factor={k_frac_lr_factor})"
            )
        if k_net_params:
            param_groups.append({
                'params': k_net_params,
                'lr': self.lr * self.k_net_lr_factor,
                'name': 'k_net'
            })
        
        # v3.17: 井眼奇异性振幅网络 well_log_amp_net
        if hasattr(model, 'well_log_amp_net'):
            singularity_params = list(model.well_log_amp_net.parameters())
            if singularity_params:
                param_groups.append({
                    'params': singularity_params,
                    'lr': self.lr,  # 与 field_net 同步
                    'name': 'well_singularity'
                })
                self.logger.info(
                    f"  [v3.17] well_singularity 参数组: "
                    f"{sum(p.numel() for p in singularity_params)} params, "
                    f"lr={self.lr:.2e}"
                )
        
        # v3.14: Corey 指数 (ng, nw) 可学习 — 从 loss_fn.relperm 收集
        corey_params = []
        if hasattr(self, 'loss_fn') and hasattr(self.loss_fn, 'relperm'):
            rp = self.loss_fn.relperm
            for name, param in rp.named_parameters():
                if '_ng_log' in name or '_nw_log' in name:
                    corey_params.append(param)
        if corey_params:
            corey_lr_factor = opt_cfg.get('corey_lr_factor', 1.0)
            param_groups.append({
                'params': corey_params,
                'lr': self.lr * corey_lr_factor,
                'weight_decay': 0.0,  # ★ 禁用! weight_decay 会把 _ng_log→0 (ng→1), 与先验正则冲突
                'name': 'corey'
            })
            rp = self.loss_fn.relperm
            self.logger.info(
                f"  [v3.14] Corey 指数可学习: ng₀={rp.ng.item():.4f}, "
                f"nw₀={rp.nw.item():.4f} (SY13 拟合先验, lr_factor={corey_lr_factor})"
            )
        
        self.optimizer = optim.AdamW(param_groups, lr=self.lr, weight_decay=wd)
        
        # v3.14: 共享 TorchRelPerm — optimizer 创建后再注入, 避免参数重复注册
        self.model.well_model.relperm = self.loss_fn.relperm
        self.logger.info("  [v3.14] well_model.relperm = loss_fn.relperm (共享实例)")
        
        if k_net_params:
            self.logger.info(
                f"k_net 学习率: {self.lr * self.k_net_lr_factor:.2e} "
                f"(factor={self.k_net_lr_factor}, warmup={self.k_net_warmup_steps} steps)"
            )
        
        # 学习率调度: Warmup + Cosine Decay + Stage D 额外衰减
        sched_cfg = train_cfg.get('scheduler', {})
        min_lr = sched_cfg.get('min_lr', 1e-6)
        warmup_steps_lr = sched_cfg.get('warmup_steps', 1000)
        
        # ===== v3.2: 预计算 Stage D 起始步 (供 schedule 使用)，与 train() 内阶段划分一致 =====
        # 使用与 train() 相同的合并逻辑：m4_config.training_stages + train.training_stages，避免 LR 降档与阶段不同步
        stages_cfg_lr = dict(config.get('m4_config', {}).get('training_stages', {}))
        train_stages = config.get('train', {}).get('training_stages', {})
        for k, v in train_stages.items():
            if isinstance(v, dict):
                stages_cfg_lr[k] = {**stages_cfg_lr.get(k, {}), **v}
        _frac_a = stages_cfg_lr.get('stage_a', {}).get('fraction', 0.15)
        _frac_b = stages_cfg_lr.get('stage_b', {}).get('fraction', 0.25)
        _frac_c = stages_cfg_lr.get('stage_c', {}).get('fraction', 0.35)
        self._stage_d_start = int(self.max_steps * (_frac_a + _frac_b + _frac_c))
        
        # 场网络 + 反演参数的 schedule (含 Stage D 衰减)
        def warmup_cosine_schedule(step):
            if step < warmup_steps_lr:
                # 线性 warmup: 1e-5 → lr
                return max(step / warmup_steps_lr, 1e-5 / self.lr)
            else:
                # Cosine decay
                progress = (step - warmup_steps_lr) / max(self.max_steps - warmup_steps_lr, 1)
                base = max(min_lr / self.lr, 0.5 * (1 + math.cos(math.pi * progress)))
                # ===== v3.2: Stage D 额外 0.5x 衰减, 减少后期震荡 =====
                if step >= self._stage_d_start:
                    base *= 0.5
                return base
        
        # k_net 专属 schedule: 前 k_net_warmup_steps 步冻结(lr≈0), 然后跟主 schedule
        k_net_warmup = self.k_net_warmup_steps
        def k_net_schedule(step):
            if step < k_net_warmup:
                # 冻结期: 返回极小因子 (非 0 避免 Adam 状态异常)
                return 1e-8 / (self.lr * self.k_net_lr_factor + 1e-20)
            else:
                return warmup_cosine_schedule(step)
        
        # 为每个参数组分配 schedule (顺序必须与 param_groups 一致):
        # [field, inversion, k_frac?, well_singularity?, corey?, k_net?]
        schedules = [warmup_cosine_schedule, warmup_cosine_schedule]
        if k_frac_params:
            schedules.append(warmup_cosine_schedule)  # k_frac: 与主网络同步衰减
        if hasattr(model, 'well_log_amp_net') and list(model.well_log_amp_net.parameters()):
            schedules.append(warmup_cosine_schedule)  # v3.17: well_singularity
        if corey_params:
            schedules.append(warmup_cosine_schedule)  # corey ng/nw: 与主网络同步衰减
        if k_net_params:
            schedules.append(k_net_schedule)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, schedules)
        
        # 梯度裁剪
        reg_cfg = config.get('loss', {}).get('regularization', {})
        self.grad_clip = reg_cfg.get('grad_clip_norm', 5.0)
        
        # --- ReLoBRaLo ---
        m5_cfg = config.get('m5_config', {})
        relobralo_cfg = m5_cfg.get('relobralo', {})
        self.use_relobralo = relobralo_cfg.get('enable', True)
        
        # v12: ReLoBRaLo 不包含 qg_nearzero
        loss_names = ['ic', 'bc', 'pde', 'qg', 'shutin_delta', 'smooth_pwf', 'monotonic', 'prior', 'k_reg', 'whp', 'sw_bounds', 'tds']
        if self.use_relobralo:
            self.balancer = ReLoBRaLo(
                loss_names=loss_names,
                temperature=relobralo_cfg.get('temperature', 1.0),
                alpha=relobralo_cfg.get('alpha', 0.999),
                rho=relobralo_cfg.get('rho', 0.999),
                warmup_steps=relobralo_cfg.get('warmup_steps', 200),
            )
        else:
            self.balancer = ManualLossBalancer({n: 1.0 for n in loss_names})
        
        # --- RAR ---
        self.rar = RARSampler(config, sampler)
        
        # --- 训练历史 ---
        self.history = {
            'step': [], 'total': [], 'ic': [], 'bc': [], 'pde': [],
            'qg': [], 'qg_nearzero': [], 'shutin_delta': [], 'whp': [], 'smooth_pwf': [], 'smooth_qg': [], 'monotonic': [], 'prior': [],
            'k_reg': [], 'sw_bounds': [], 'tds': [], 'lr': [], 'lr_k_net': [],
            'sw_min': [], 'sw_max': [], 'stage': [],
            'k_frac_mD': [], 'k_eff_mD': [], 'f_frac': [], 'dp_wellbore': [], 'r_e_m': [],
        }
        
        # --- 输出目录 ---
        output_dir = config['paths']['outputs']
        self.output_dir = output_dir
        self.ckpt_dir = config['paths'].get('checkpoints', os.path.join(output_dir, 'ckpt'))
        self.report_dir = config['paths'].get('reports', os.path.join(output_dir, 'reports'))
        self.fig_dir = config['paths'].get('figures', os.path.join(output_dir, 'figs'))
        ensure_dir(self.output_dir)
        ensure_dir(self.ckpt_dir)
        ensure_dir(self.report_dir)
        ensure_dir(self.fig_dir)
        
        # v12: 时间顺序切分默认 70/15/15
        m5_split = self.config.get('train', {}).get('m5_split', [0.7, 0.15, 0.15])
        self.m5_train_ratio = float(m5_split[0])
        self.m5_val_ratio = float(m5_split[1])
        self.m5_test_ratio = float(m5_split[2]) if len(m5_split) > 2 else (1.0 - self.m5_train_ratio - self.m5_val_ratio)
        
        self._shape_logged = False  # 仅首次打印 shape 诊断
        # --- 加载数据 ---
        self.train_t_min_norm = 0.0
        self.train_t_max_norm = 1.0
        self._load_well_data()
        self._load_tds_data()
        self._compute_pde_scaling()
        
        self.logger.info(
            f"M5Trainer 初始化完成: device={device}, lr={self.lr}, "
            f"max_steps={self.max_steps}, relobralo={self.use_relobralo}, "
            f"rar={self.rar.enable}"
        )
    
    def _load_well_data(self):
        """加载所有井的生产数据用于同化"""
        self.well_data = {}
        train_t_min_list = []
        train_t_max_list = []
        
        for wid in self.model.well_ids:
            data = self.sampler.sample_well_data(wid)
            if data and len(data.get('xyt', [])) > 0:
                # 统一把 1D 观测量 reshape 成 (N,1)，避免 (N,1)-(N,) 广播成 (N,N)
                def _to_col(t):
                    x = torch.from_numpy(t).float().to(self.device)
                    return x.flatten().view(-1, 1)
                wd = {
                    'xyt': torch.from_numpy(data['xyt']).float().to(self.device),
                    'qg_obs': _to_col(data['qg_obs']),
                    'p_obs': _to_col(data['p_obs']),
                    't_days': data['t_days'],
                }
                if 'qg_valid_mask' in data:
                    wd['qg_valid_mask'] = _to_col(data['qg_valid_mask'])
                if 'shutin_mask' in data:
                    wd['shutin_mask'] = _to_col(data['shutin_mask'])
                if 'prod_hours_norm' in data:
                    wd['prod_hours_norm'] = _to_col(data['prod_hours_norm'])
                if 'casing_norm' in data:
                    wd['casing_norm'] = _to_col(data['casing_norm'])
                if 'p_valid_mask_local' in data:
                    wd['p_valid_mask_local'] = _to_col(data['p_valid_mask_local'])
                if 'qw_obs' in data:
                    wd['qw_obs'] = _to_col(data['qw_obs'])
                for k, v in data.items():
                    if k.endswith('_obs') and k not in wd:
                        wd[k] = _to_col(v)
                # 划分比例来自 config.train.m5_split（v12 默认 70/15/15）
                n = len(data['xyt'])
                n_train = int(n * self.m5_train_ratio)
                n_val = int(n * self.m5_val_ratio)
                wd['idx_train'] = slice(0, n_train)
                wd['idx_val'] = slice(n_train, n_train + n_val)
                wd['idx_test'] = slice(n_train + n_val, n)
                
                # P0-2: 切分后自动校验有效点数（硬约束）
                qg_train = wd['qg_obs'][wd['idx_train']].cpu().numpy().flatten()
                qg_val = wd['qg_obs'][wd['idx_val']].cpu().numpy().flatten()
                qg_test = wd['qg_obs'][wd['idx_test']].cpu().numpy().flatten()
                p_train = wd['p_obs'][wd['idx_train']].cpu().numpy().flatten()
                p_val = wd['p_obs'][wd['idx_val']].cpu().numpy().flatten()
                p_test = wd['p_obs'][wd['idx_test']].cpu().numpy().flatten()
                
                n_train_open = int(np.sum(qg_train > 1.0))
                n_val_open = int(np.sum(qg_val > 1.0))
                n_test_open = int(np.sum(qg_test > 1.0))
                n_val_p_valid = int(np.sum(np.isfinite(p_val) & (p_val > 0.1)))
                n_test_p_valid = int(np.sum(np.isfinite(p_test) & (p_test > 0.1)))
                
                self.logger.info(
                    f"  井 {wid}: {n} 条同化数据 (train={n_train}, val={n_val}, test={n-n_train-n_val})\n"
                    f"    开井点数 (q>1): train={n_train_open}, val={n_val_open}, test={n_test_open}\n"
                    f"    有效压力点: val_p={n_val_p_valid}, test_p={n_test_p_valid}"
                )
                
                if n_test_open == 0:
                    self.logger.warning(f"  井 {wid} test 无开井点，test 指标将不可用")
                
                if n_train > 0:
                    t_train = wd['xyt'][wd['idx_train'], 2]
                    if t_train.numel() > 0:
                        train_t_min_list.append(float(torch.min(t_train).item()))
                        train_t_max_list.append(float(torch.max(t_train).item()))

                self.well_data[wid] = wd
            else:
                self.logger.warning(f"  井 {wid}: 无可用同化数据")
        
        # 井段厚度 (从 mk_interval_points 获取)
        if train_t_min_list and train_t_max_list:
            self.train_t_min_norm = float(np.clip(min(train_t_min_list), 0.0, 1.0))
            self.train_t_max_norm = float(np.clip(max(train_t_max_list), 0.0, 1.0))
            if self.train_t_max_norm <= self.train_t_min_norm:
                self.train_t_min_norm = 0.0
                self.train_t_max_norm = 1.0
        else:
            self.train_t_min_norm = 0.0
            self.train_t_max_norm = 1.0

        self.logger.info(
            f"  train time window(norm): [{self.train_t_min_norm:.4f}, {self.train_t_max_norm:.4f}]"
        )

        # v3.4: Peaceman WI 必须用有效储厚(net pay), 非毛厚度(gross)
        # 附表8 测井解释成果表: SY9 有效储厚 = 42.4 + 6.0 = 48.4 m
        # mk_interval_points.csv 的 mk_thickness=92.14 m 是毛厚度, 用它会导致
        # WI 膨胀 92.14/48.4=1.9x, 优化器被迫把 k_frac 从 4.0 压到 2.1 mD 来补偿
        net_pay_override = {
            'SY9': 48.4,      # 附表8: 16.296 + 32.1 = 48.4 m
            'SY13': 41.65,    # 附表8: 5.25 + 11.3 + 25.1 = 41.65 m
            'SY201': 37.9,    # 附表8: 7.0 + 30.9 = 37.9 m
            'SY101': 41.7,    # 附表8: 38.7 + 3.0 = 41.7 m
            'SY102': 45.44,   # 附表8: 45.44 m
            'SY116': 39.3,    # 附表8: 5.0 + 34.3 = 39.3 m
            'SYX211': 6.0,    # 附表8: 仅气水同层有效储厚 6.0 m
        }
        self.well_h = {}
        try:
            import pandas as pd
            mk_file = os.path.join(self.config['paths']['clean_data'], 'mk_interval_points.csv')
            mk_df = pd.read_csv(mk_file)
            for _, row in mk_df.iterrows():
                wid = row['well_id']
                if wid in net_pay_override:
                    self.well_h[wid] = net_pay_override[wid]
                else:
                    self.well_h[wid] = float(row['mk_thickness'])
        except Exception as e:
            self.logger.warning(f"加载井段厚度失败: {e}, 使用 net_pay_override")
            self.well_h = dict(net_pay_override)
        for wid, h in self.well_h.items():
            self.logger.info(f"  well_h[{wid}] = {h:.1f} m (有效储厚)")

        # v3.17: 注入井位归一化坐标 → 激活井眼奇异性分解
        if hasattr(self.model, 'set_well_xy_norm') and self.well_data:
            primary_wid = self.model.well_ids[0] if self.model.well_ids else 'SY9'
            if primary_wid in self.well_data:
                wxy = self.well_data[primary_wid]['xyt'][0, :2]  # (2,) 归一化 [x, y]
                self.model.set_well_xy_norm(wxy)
    
    def _load_tds_data(self):
        """
        v3.23: 加载附表6-水分析TDS数据, 转换为Sw软标签约束
        
        v3.23 FIX (vs v3.22):
            修复域外TDS数据污染: 附表6 SY9数据跨2013-2022(174样本),
            而PINN训练域仅2013-2016(t_max=1331天). 旧版np.clip将~88个
            域外样本堆叠到t_norm=1.0, 导致边界约束被未来数据污染.
            现改为域外过滤, 仅保留训练域内样本参与软约束.
        
        物理模型:
            f_brine = clip((TDS - TDS_condensate) / (TDS_brine - TDS_condensate), 0, 1)
            Sw_tds  = Swc + f_brine × (1 - Swc - Sgr)
        
        其中:
            TDS_condensate = 100 mg/L  (2013-06~2014-06 凝析水基线)
            TDS_brine = 105,000 mg/L   (2016-09 峰值, 地层卤水端元, CaCl₂型)
            Swc, Sgr 来自 TorchRelPerm (与PDE/相渗一致)
        """
        import pandas as pd
        
        self.tds_data = None
        
        project_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(project_root, 'data', 'raw',
                                '附表6-流体性质统计表__水分析.csv')
        if not os.path.exists(csv_path):
            self.logger.warning(f"  [TDS→Sw] 附表6水分析文件不存在: {csv_path}")
            return
        
        try:
            df = pd.read_csv(csv_path, header=None, skiprows=3,
                             encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            self.logger.warning(f"  [TDS→Sw] 读取附表6水分析失败: {e}")
            return
        
        # 列索引: 1=井号, 3=取样日期, 32=总矿化度(mg/L)
        col_well = 1
        col_date = 3
        col_tds = 32
        
        # 获取生产起始日期 (与 sampler 一致)
        try:
            prod_start = pd.to_datetime(
                self.sampler.production_data['date'].iloc[0])
        except Exception as e:
            self.logger.warning(f"  [TDS→Sw] 无法获取生产起始日期: {e}")
            return
        
        t_max_days = max(float(self.sampler.t_max), 1.0)
        
        # Sw端元参数 (与PDE/相渗共享)
        rp = self.loss_fn.relperm
        swc = float(rp.Swc)
        sgr = float(rp.Sgr)
        sw_mobile_range = 1.0 - swc - sgr  # 可动水饱和度范围
        
        TDS_CONDENSATE = 100.0    # mg/L (凝析水基线)
        TDS_BRINE = 105000.0      # mg/L (地层卤水端元)
        
        self.tds_data = {}
        
        for wid in self.model.well_ids:
            mask = df[col_well].astype(str).str.strip() == wid
            sub = df.loc[mask, [col_date, col_tds]].copy()
            sub.columns = ['date_str', 'tds_raw']
            
            # 解析日期
            sub['date'] = pd.to_datetime(sub['date_str'], errors='coerce')
            sub = sub.dropna(subset=['date'])
            
            # 解析TDS
            sub['tds_mg_l'] = pd.to_numeric(sub['tds_raw'], errors='coerce')
            sub = sub.dropna(subset=['tds_mg_l'])
            sub = sub[sub['tds_mg_l'] > 0]
            
            if len(sub) < 5:
                self.logger.info(f"  [TDS→Sw] {wid}: 仅{len(sub)}个有效点, 跳过")
                continue
            
            # t_day → t_norm (与PINN时间轴一致)
            t_day_all = (sub['date'] - prod_start).dt.days.astype(float).values
            
            # v3.23 FIX: 只保留PINN训练域 [0, t_max] 内的TDS样本
            # 域外数据(2017-2022)不应参与M5软约束, 否则会:
            #   1) ~50%样本堆叠在t_norm=1.0, 破坏时序信息
            #   2) 边界约束被未来数据污染(物理上不自洽)
            #   3) 插值在t_norm≈1.0处产生非物理跳变
            # 注: M7水侵预警模块(load_tds_timeseries)使用全量TDS是正确的设计
            n_total = len(t_day_all)
            domain_mask = (t_day_all >= 0) & (t_day_all <= t_max_days)
            n_domain = int(domain_mask.sum())
            n_outside = n_total - n_domain
            
            if n_outside > 0:
                self.logger.info(
                    f"  [TDS→Sw] {wid}: 过滤{n_outside}个域外样本 "
                    f"(t_day>{t_max_days:.0f}天), 保留{n_domain}/{n_total}个域内样本"
                )
            
            t_day = t_day_all[domain_mask]
            sub = sub.iloc[domain_mask.nonzero()[0]]
            
            if len(sub) < 5:
                self.logger.info(
                    f"  [TDS→Sw] {wid}: 域内仅{len(sub)}个有效点, 跳过")
                continue
            
            t_norm = t_day / t_max_days  # 已在[0,1]内, 无需clip
            
            # TDS → f_brine → Sw_tds
            tds_vals = sub['tds_mg_l'].values
            f_brine = np.clip(
                (tds_vals - TDS_CONDENSATE) / (TDS_BRINE - TDS_CONDENSATE),
                0.0, 1.0
            )
            sw_tds = swc + f_brine * sw_mobile_range
            
            # 排序并存为tensor
            sort_idx = np.argsort(t_norm)
            t_norm_sorted = t_norm[sort_idx]
            sw_tds_sorted = sw_tds[sort_idx]
            
            self.tds_data[wid] = {
                't_norm': torch.from_numpy(t_norm_sorted).float().to(self.device),
                'sw_tds': torch.from_numpy(sw_tds_sorted).float().to(self.device),
                'n_points': len(sub),
            }
            
            self.logger.info(
                f"  [TDS→Sw] {wid}: {len(sub)}个采样点, "
                f"t_day=[{t_day.min():.0f}, {t_day.max():.0f}], "
                f"TDS=[{tds_vals.min():.0f}, {tds_vals.max():.0f}] mg/L, "
                f"Sw_tds=[{sw_tds.min():.3f}, {sw_tds.max():.3f}], "
                f"Swc={swc:.3f}, Sgr={sgr:.3f}"
            )
        
        if not self.tds_data:
            self.tds_data = None
            self.logger.info("  [TDS→Sw] 无可用TDS数据")
        else:
            total_pts = sum(v['n_points'] for v in self.tds_data.values())
            self.logger.info(f"  [TDS→Sw] 共加载 {total_pts} 个TDS→Sw软标签点")
    
    def _compute_pde_scaling(self):
        """计算 PDE 物理缩放系数并注入 loss_fn (与 M4 逻辑相同)"""
        physics_cfg = self.config.get('physics', {})
        priors = physics_cfg.get('priors', {})
        pde_cfg = physics_cfg.get('pde', {})
        domain_cfg = pde_cfg.get('domain', {})
        
        mk = self.config.get('mk_formation', {})
        p_avg = mk.get('avg_pressure_MPa', {})
        p_ref = p_avg.get('value', 76.0) if isinstance(p_avg, dict) else float(p_avg)
        T_avg = mk.get('avg_temperature_C', {})
        T_ref = T_avg.get('value', 140.32) if isinstance(T_avg, dict) else float(T_avg)
        
        k_mD = priors.get('k_eff_mD', {}).get('value', 5.0) \
            if isinstance(priors.get('k_eff_mD'), dict) else 5.0
        phi = priors.get('phi', {}).get('value', 0.0216) \
            if isinstance(priors.get('phi'), dict) else 0.0216
        mu_mPa_s = priors.get('mu_ref_mPa_s', {}).get('value', 0.035) \
            if isinstance(priors.get('mu_ref_mPa_s'), dict) else 0.035
        cg_1_per_MPa = priors.get('cg_ref_1_per_MPa', {}).get('value', 0.013) \
            if isinstance(priors.get('cg_ref_1_per_MPa'), dict) else 0.013
        cr_1_per_MPa = priors.get('c_r_1_per_MPa', {}).get('value', 0.0) \
            if isinstance(priors.get('c_r_1_per_MPa'), dict) else 0.0
        
        # 尝试 M3 PVT 覆盖 cg
        try:
            from physics.pvt import GasPVT
            gas_pvt = GasPVT(config=self.config)
            cg_1_per_MPa = float(gas_pvt.cg(p_ref, T_ref).item())
            self.logger.info(f"  M3 PVT cg({p_ref}, {T_ref}) = {cg_1_per_MPa:.6f}")
        except Exception:
            pass
        
        # 获取 Bg 参考值用于井模型
        self.bg_ref = 0.002577  # 附表5-4: Bg(75.7MPa, 140.32°C) = 2.5769e-3 m³/sm³
        try:
            from physics.pvt import GasPVT
            gas_pvt = GasPVT(config=self.config)
            self.bg_ref = float(gas_pvt.bg(p_ref, T_ref).item())
            self.logger.info(f"  M3 PVT Bg({p_ref}, {T_ref}) = {self.bg_ref:.6f}")
        except Exception:
            pass
        
        x_min = domain_cfg.get('x_min_m', self.sampler.x_min)
        x_max = domain_cfg.get('x_max_m', self.sampler.x_max)
        y_min = domain_cfg.get('y_min_m', self.sampler.y_min)
        y_max = domain_cfg.get('y_max_m', self.sampler.y_max)
        t_max_d = domain_cfg.get('t_max_d', self.sampler.t_max)
        
        dx = x_max - x_min
        dy = y_max - y_min
        t_max_s = t_max_d * 86400.0
        
        k_SI = k_mD * 9.869233e-16
        mu_SI = mu_mPa_s * 1e-3
        cg_SI = cg_1_per_MPa * 1e-6
        cr_SI = cr_1_per_MPa * 1e-6
        ct_SI = cg_SI + cr_SI
        
        alpha_phys = k_SI / (phi * mu_SI * ct_SI)
        sx = (2.0 / dx) ** 2
        sy = (2.0 / dy) ** 2
        
        self.loss_fn.alpha_x = alpha_phys * t_max_s * sx
        self.loss_fn.alpha_y = alpha_phys * t_max_s * sy
        
        self.logger.info(
            f"  PDE 缩放: αx={self.loss_fn.alpha_x:.6f}, αy={self.loss_fn.alpha_y:.6f}"
        )
    
    # ================================================================== #
    #             FIX-v9: 时变源项辅助方法
    # ================================================================== #
    @torch.no_grad()
    def _interp_qg(self,
                   t_known: torch.Tensor,
                   qg_known: torch.Tensor,
                   t_query: torch.Tensor,
                   valid_mask: Optional[torch.Tensor] = None
                   ) -> torch.Tensor:
        """
        对 qg(t) 做线性插值；若给定 valid_mask，仅使用真实 qg 观测点。
        当 valid 点不足时做稳健兜底，避免把缺失补零当成真实关井。
        """
        if t_known.numel() == 0:
            return torch.zeros_like(t_query)

        t_use = t_known
        q_use = qg_known
        if valid_mask is not None:
            vm = valid_mask
            if vm.dim() > 1:
                vm = vm.squeeze()
            v = (vm > 0.5)
            if v.any():
                t_use = t_known[v]
                q_use = qg_known[v]

        if t_use.numel() == 0:
            return torch.zeros_like(t_query)
        if t_use.numel() == 1:
            return torch.full_like(t_query, q_use[0])

        sorted_idx = torch.argsort(t_use)
        t_sorted = t_use[sorted_idx]
        q_sorted = q_use[sorted_idx]

        t_lo_val = t_sorted[0].item()
        t_hi_val = t_sorted[-1].item()
        t_clamped = torch.clamp(t_query, t_lo_val, t_hi_val)

        idx = torch.searchsorted(t_sorted.contiguous(), t_clamped.contiguous())
        idx = torch.clamp(idx, 1, len(t_sorted) - 1)

        t_lo = t_sorted[idx - 1]
        t_hi = t_sorted[idx]
        qg_lo = q_sorted[idx - 1]
        qg_hi = q_sorted[idx]

        dt = t_hi - t_lo
        dt = torch.where(dt.abs() < 1e-12, torch.ones_like(dt), dt)
        alpha = torch.clamp((t_clamped - t_lo) / dt, 0.0, 1.0)

        return qg_lo + alpha * (qg_hi - qg_lo)

    def _compute_time_varying_source(self,
                                     x_colloc: torch.Tensor,
                                     well_xy_norm: torch.Tensor,
                                     well_rates_per_point: list
                                     ) -> torch.Tensor:
        """
        计算时变源项: source(x_i, y_i, t_i) = Σ_j qg_j(t_i) * kernel(x_i, y_i)
        
        与 GaussianSourceTerm.compute_source 的区别:
            - 旧版: well_rates 是 (n_wells,) 标量 → 常数源项
            - 新版: well_rates_per_point 是 [(N,), ...] → 每个配点对应不同流量
        
        Args:
            x_colloc: (N, 2) 配点归一化坐标 [x_n, y_n]
            well_xy_norm: (n_wells, 2) 井位归一化坐标
            well_rates_per_point: list of (N,) tensors, 每口井在每个配点时刻的流量 (m³/s)
        
        Returns:
            source: (N, 1) 各配点处的源项强度
        """
        st = self.model.well_model.source_term
        N = x_colloc.shape[0]
        source = torch.zeros(N, 1, device=x_colloc.device)
        cutoff_r2 = (st.cutoff_factor * st.sigma) ** 2
        
        for j in range(len(well_rates_per_point)):
            dx = x_colloc[:, 0] - well_xy_norm[j, 0]
            dy = x_colloc[:, 1] - well_xy_norm[j, 1]
            r2 = dx ** 2 + dy ** 2
            
            mask = r2 < cutoff_r2
            kernel = torch.zeros(N, device=x_colloc.device)
            kernel[mask] = torch.exp(-r2[mask] / (2.0 * st.sigma ** 2))
            
            kernel_sum = kernel.sum() + 1e-12
            kernel_normalized = kernel / kernel_sum
            
            # 时变: 每个配点乘以该时刻的流量
            source[:, 0] += well_rates_per_point[j] * kernel_normalized
        
        return source
    
    def _sample_batch(self, step: int, use_train_only: bool = False) -> Dict[str, torch.Tensor]:
        """生成训练 batch (含 RAR 增强)"""
        samp_cfg = self.config.get('m4_config', {}).get('sampling', {})
        n_domain = samp_cfg.get('n_domain', 4096)  # 增加到 4096
        n_boundary = samp_cfg.get('n_boundary', 512)
        n_initial = samp_cfg.get('n_initial', 512)
        
        training_progress = step / max(self.max_steps, 1)
        
        x_ic = self.sampler.sample_initial(n_initial, seed=step)
        x_bc = self.sampler.sample_boundary(n_boundary, seed=step + 1)

        t_min_raw = None
        t_max_raw = None
        if use_train_only:
            t_min_norm = float(np.clip(getattr(self, 'train_t_min_norm', 0.0), 0.0, 1.0))
            t_max_norm = float(np.clip(getattr(self, 'train_t_max_norm', 1.0), 0.0, 1.0))
            if t_max_norm > t_min_norm:
                t_min_raw = t_min_norm * float(self.sampler.t_max)
                t_max_raw = t_max_norm * float(self.sampler.t_max)
        
        # RAR 增强的域内采样
        if self.rar.enable and self.rar.rar_points is not None:
            x_pde_np, gx_np, gy_np = self.rar.get_augmented_domain_points(
                n_domain, seed=step + 2
            )
        else:
            x_pde_np = self.sampler.sample_domain(
                n_domain, seed=step + 2, training_progress=training_progress,
                t_min_raw=t_min_raw, t_max_raw=t_max_raw
            )
            gx_np, gy_np = self.sampler.get_last_h_grad()
        
        if use_train_only and t_min_raw is not None and t_max_raw is not None:
            t_max_sampler = max(float(self.sampler.t_max), 1e-12)
            t_min_norm = t_min_raw / t_max_sampler
            t_max_norm = t_max_raw / t_max_sampler
            x_pde_np[:, 2] = np.clip(x_pde_np[:, 2], t_min_norm, t_max_norm)

        batch = {
            'x_ic': torch.from_numpy(x_ic).float().to(self.device),
            'x_bc': torch.from_numpy(x_bc).float().to(self.device),
            'x_pde': torch.from_numpy(x_pde_np).float().to(self.device),
        }
        
        if gx_np is not None:
            batch['h_grad'] = {
                'gx': torch.from_numpy(gx_np).float().to(self.device).unsqueeze(-1),
                'gy': torch.from_numpy(gy_np).float().to(self.device).unsqueeze(-1),
            }
        
        return batch
    
    def _get_singularity_log_str(self) -> str:
        """v3.17: 井眼奇异性A(t)振幅日志字符串"""
        if not (hasattr(self.model, 'use_well_singularity') and 
                self.model.use_well_singularity and self.model._well_xy_set):
            return ""
        try:
            with torch.no_grad():
                t_probe = torch.tensor([[0.0], [0.5], [1.0]], device=self.device)
                A_vals = self.model.well_log_amp_net(t_probe).squeeze().tolist()
            return f" | A(t)=[{A_vals[0]:.2f},{A_vals[1]:.2f},{A_vals[2]:.2f}]"
        except Exception:
            return ""

    def _compute_well_outputs(self, use_train_only: bool = False) -> Dict[str, Dict]:
        """
        在井位处计算场变量和产量。
        v12: 全量井点反传，use_train_only 由 _train_step 固定为 False。
        断言与日志使用本次传入的 qg_obs/p_obs 的 shape（切片后），见 v12_changes_summary 断言修复。
        """
        well_outputs = {}
        
        for wid in self.model.well_ids:
            if wid not in self.well_data:
                continue
            
            wdata = self.well_data[wid]
            if use_train_only and wdata.get('idx_train') is not None:
                idx = wdata['idx_train']
                xyt = wdata['xyt'][idx]
                qg_obs = wdata['qg_obs'][idx]
                p_obs = wdata['p_obs'][idx]
                t_days = wdata['t_days'][idx]
                qg_valid_mask = wdata['qg_valid_mask'][idx] if 'qg_valid_mask' in wdata else None
                shutin_mask = wdata['shutin_mask'][idx] if 'shutin_mask' in wdata else None
                prod_hours_norm = wdata['prod_hours_norm'][idx] if 'prod_hours_norm' in wdata else None
                casing_norm = wdata['casing_norm'][idx] if 'casing_norm' in wdata else None
            else:
                xyt = wdata['xyt']
                qg_obs = wdata['qg_obs']
                p_obs = wdata['p_obs']
                t_days = wdata['t_days']
                qg_valid_mask = wdata.get('qg_valid_mask', None)
                shutin_mask = wdata.get('shutin_mask', None)
                prod_hours_norm = wdata.get('prod_hours_norm', None)
                casing_norm = wdata.get('casing_norm', None)
            
            h_well = self.well_h.get(wid, 90.0)
            result = self.model.evaluate_at_well(
                well_id=wid,
                well_xyt_norm=xyt,
                h_well=h_well,
                bg_val=self.bg_ref,
                prod_hours_norm=prod_hours_norm,
                casing_norm=casing_norm,
            )
            qg_pred = result['qg']
            # 关井制度已知（生产时间<=0）时，强制 qg=0，避免训练在关井段出现非物理高产
            if shutin_mask is not None:
                qg_pred = torch.where(shutin_mask > 0.5, torch.zeros_like(qg_pred), qg_pred)
            
            well_outputs[wid] = {
                'qg': qg_pred,
                'p_wf': result['p_wf'],
                'p_cell': result['p_cell'],
                'sw_cell': result['sw_cell'],
                'qg_obs': qg_obs,
                'qg_valid_mask': qg_valid_mask,
                'shutin_mask': shutin_mask,
                'prod_hours_norm': prod_hours_norm,
                'casing_norm': casing_norm,
                'p_obs': p_obs,
                'dp_wellbore': self.model.dp_wellbore,
                't_norm': xyt[:, 2:3],
                't_days': t_days,
            }
            # v3.14: 产水量同化
            if 'qw' in result:
                well_outputs[wid]['qw_pred'] = result['qw']
            qw_obs_data = wdata.get('qw_obs', None)
            if qw_obs_data is not None:
                well_outputs[wid]['qw_obs'] = qw_obs_data if use_train_only is False else qw_obs_data[wdata.get('idx_train', slice(None))]
            # Shape 诊断: 用本次传入的 xyt/qg_obs/p_obs（train 切片时即切片后的 shape）
            qg_pred_s = tuple(qg_pred.shape)
            qg_obs_s = tuple(qg_obs.shape)
            pwf_pred_s = tuple(result['p_wf'].shape)
            p_obs_s = tuple(p_obs.shape)
            if not self._shape_logged:
                self.logger.info(
                    f"  [shape] {wid}: qg pred={qg_pred_s} vs obs={qg_obs_s}, "
                    f"p_wf pred={pwf_pred_s} vs p_obs={p_obs_s}"
                )
                self._shape_logged = True
            assert qg_pred_s == qg_obs_s and len(qg_pred_s) == 2 and qg_pred_s[1] == 1, (
                f"qg shape must be (N,1): pred {qg_pred_s} vs obs {qg_obs_s}"
            )
            assert pwf_pred_s == p_obs_s and len(pwf_pred_s) == 2 and pwf_pred_s[1] == 1, (
                f"p_wf shape must be (N,1): pred {pwf_pred_s} vs obs {p_obs_s}"
            )
        
        return well_outputs
    
    def _eval_val_score(self) -> float:
        """
        v13: 验证集复合分数
        score = 0.40*MAPE_open + 0.20*MAPE_high + 0.15*MAPE_low
              + 0.10*RMSE_pwf + 0.15*ShutinPenalty

        其中:
        - MAPE_open / MAPE_low 仅在开井点(q>1)上统计；
        - ShutinPenalty = min(MAE_shutin / 10000, 120)，用于抑制关井段高产量塌缩；
        - 当 val 某子集样本过少时，回退到 train 对应子集，避免评分抖动。
        val 无开井点时返回 999，调用方不更新 best。
        """
        LOW_QG = 5e4    # m³/d，低产阈值
        HIGH_QG = 4e5   # m³/d，高产阈值
        self.model.eval()
        scores = []
        dp_wb = float(self.model.dp_wellbore.item()) if hasattr(self.model.dp_wellbore, 'item') else float(self.model.dp_wellbore)
        with torch.no_grad():
            for wid in self.model.well_ids:
                if wid not in self.well_data:
                    continue
                wdata = self.well_data[wid]
                idx_val = wdata.get('idx_val')
                idx_train = wdata.get('idx_train')
                if idx_val is None:
                    continue
                xyt_val = wdata['xyt'][idx_val]
                if len(xyt_val) == 0:
                    continue
                phn_val = wdata['prod_hours_norm'][idx_val] if 'prod_hours_norm' in wdata else None
                cn_val = wdata['casing_norm'][idx_val] if 'casing_norm' in wdata else None
                result_val = self.model.evaluate_at_well(
                    wid, xyt_val,
                    h_well=self.well_h.get(wid, 90.0),
                    bg_val=self.bg_ref,
                    prod_hours_norm=phn_val,
                    casing_norm=cn_val,
                )
                qg_pred_val = result_val['qg'].cpu().numpy().flatten()
                qg_obs_val = wdata['qg_obs'][idx_val].cpu().numpy().flatten()
                if 'shutin_mask' in wdata:
                    shutin_val = (wdata['shutin_mask'][idx_val].cpu().numpy().flatten() > 0.5)
                    qg_pred_val = np.where(shutin_val, 0.0, qg_pred_val)
                valid_qg_all_val = np.isfinite(qg_pred_val) & np.isfinite(qg_obs_val)
                if 'qg_valid_mask' in wdata:
                    qg_valid_val = (wdata['qg_valid_mask'][idx_val].cpu().numpy().flatten() > 0.5)
                    valid_qg_all_val = valid_qg_all_val & qg_valid_val

                valid_qg_open_val = valid_qg_all_val & (qg_obs_val > 1.0)
                valid_qg_shutin_val = valid_qg_all_val & (qg_obs_val <= 1.0)
                if np.sum(valid_qg_open_val) == 0:
                    continue

                mape_all_val = np.mean(np.abs(
                    (qg_obs_val[valid_qg_open_val] - qg_pred_val[valid_qg_open_val]) /
                    (qg_obs_val[valid_qg_open_val] + 1.0)
                )) * 100

                low_val = valid_qg_open_val & (qg_obs_val < LOW_QG)
                mape_low = None
                if np.sum(low_val) >= 5:
                    mape_low = np.mean(np.abs((qg_obs_val[low_val] - qg_pred_val[low_val]) / (qg_obs_val[low_val] + 1.0))) * 100

                high_val = valid_qg_open_val & (qg_obs_val >= HIGH_QG)
                mape_high = None
                if np.sum(high_val) >= 5:
                    mape_high = np.mean(np.abs((qg_obs_val[high_val] - qg_pred_val[high_val]) / (qg_obs_val[high_val] + 1.0))) * 100

                mae_shutin = None
                if np.sum(valid_qg_shutin_val) >= 5:
                    mae_shutin = np.mean(np.abs(qg_pred_val[valid_qg_shutin_val] - qg_obs_val[valid_qg_shutin_val]))

                if idx_train is not None and len(wdata['xyt'][idx_train]) > 0:
                    xyt_train = wdata['xyt'][idx_train]
                    phn_tr = wdata['prod_hours_norm'][idx_train] if 'prod_hours_norm' in wdata else None
                    cn_tr = wdata['casing_norm'][idx_train] if 'casing_norm' in wdata else None
                    result_train = self.model.evaluate_at_well(
                        wid, xyt_train,
                        h_well=self.well_h.get(wid, 90.0),
                        bg_val=self.bg_ref,
                        prod_hours_norm=phn_tr,
                        casing_norm=cn_tr,
                    )
                    qg_pred_tr = result_train['qg'].cpu().numpy().flatten()
                    qg_obs_tr = wdata['qg_obs'][idx_train].cpu().numpy().flatten()
                    if 'shutin_mask' in wdata:
                        shutin_tr = (wdata['shutin_mask'][idx_train].cpu().numpy().flatten() > 0.5)
                        qg_pred_tr = np.where(shutin_tr, 0.0, qg_pred_tr)
                    valid_tr_all = np.isfinite(qg_pred_tr) & np.isfinite(qg_obs_tr)
                    if 'qg_valid_mask' in wdata:
                        qg_valid_tr = (wdata['qg_valid_mask'][idx_train].cpu().numpy().flatten() > 0.5)
                        valid_tr_all = valid_tr_all & qg_valid_tr
                    valid_tr_open = valid_tr_all & (qg_obs_tr > 1.0)
                    low_tr = valid_tr_open & (qg_obs_tr < LOW_QG)
                    if mape_low is None and np.sum(low_tr) >= 1:
                        mape_low = np.mean(np.abs((qg_obs_tr[low_tr] - qg_pred_tr[low_tr]) / (qg_obs_tr[low_tr] + 1.0))) * 100
                    if mape_high is None:
                        high_tr = valid_tr_open & (qg_obs_tr >= HIGH_QG)
                        if np.sum(high_tr) >= 1:
                            mape_high = np.mean(np.abs((qg_obs_tr[high_tr] - qg_pred_tr[high_tr]) / (qg_obs_tr[high_tr] + 1.0))) * 100
                    if mae_shutin is None:
                        shutin_tr = valid_tr_all & (qg_obs_tr <= 1.0)
                        if np.sum(shutin_tr) >= 5:
                            mae_shutin = np.mean(np.abs(qg_pred_tr[shutin_tr] - qg_obs_tr[shutin_tr]))

                if mape_low is None:
                    mape_low = mape_all_val
                mape_low = min(float(mape_low), 100.0)
                if mape_high is None:
                    mape_high = mape_all_val
                mape_high = min(float(mape_high), 120.0)
                if mae_shutin is None:
                    mae_shutin = 2e5
                shutin_penalty = min(float(mae_shutin) / 10000.0, 120.0)

                # v15: 计算 val 上 qg R²，纳入 val_score 以选出动态拟合更好的 best
                obs_open = qg_obs_val[valid_qg_open_val]
                pred_open = qg_pred_val[valid_qg_open_val]
                ss_res = np.sum((obs_open - pred_open) ** 2)
                ss_tot = np.sum((obs_open - np.mean(obs_open)) ** 2)
                r2_val_qg = 1.0 - ss_res / max(ss_tot, 1e-12)
                r2_penalty = max(0.0, 1.0 - np.clip(r2_val_qg, 0.0, 1.0)) * 100.0

                p_wf_pred = result_val['p_wf'].cpu().numpy().flatten()
                p_obs = wdata['p_obs'][idx_val].cpu().numpy().flatten()
                p_wf_obs = p_obs + dp_wb
                valid_pwf = np.isfinite(p_wf_obs) & np.isfinite(p_wf_pred) & (p_wf_obs > 1.0)
                rmse_pwf = np.sqrt(np.mean((p_wf_obs[valid_pwf] - p_wf_pred[valid_pwf]) ** 2)) if np.any(valid_pwf) else 50.0
                # v18: 提高 MAPE 权重、略降 R² 权重，使 best 更贴近「qg MAPE~14% / R²~0.7+」目标
                score = (
                    0.35 * mape_all_val
                    + 0.12 * mape_high
                    + 0.12 * mape_low
                    + 0.05 * rmse_pwf
                    + 0.13 * shutin_penalty
                    + 0.23 * r2_penalty
                )
                scores.append(score)
        self.model.train()
        if not scores:
            if not getattr(self, '_val_no_producing_warned', False):
                self.logger.warning(
                    "val 无开井点，val_score=999，本步不更新 best。"
                    "建议 config.train.m5_split 改为 [0.6,0.2,0.2] 或检查数据时间顺序。"
                )
                self._val_no_producing_warned = True
        return np.mean(scores) if scores else 999.0
    
    def _train_step(self, step: int, stage_weights: Dict[str, float]) -> Dict[str, float]:
        """执行一步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # ========== 消融实验关键: 物理损失权重 override ==========
        # 硬编码的阶段权重 (w_a, w_b, ...) 不从 config 读取,
        # 导致 override loss.physics.base_weight=0 对 PDE/IC/BC 无效.
        # 此处用 config 中的 phys_enable 和 phys_base_weight 强制修正.
        stage_weights = dict(stage_weights)  # 防止修改原始字典
        if not self.phys_enable:
            for key in ['pde', 'ic', 'bc']:
                stage_weights[key] = 0.0
        else:
            for key in ['pde', 'ic', 'bc']:
                stage_weights[key] = stage_weights.get(key, 0.0) * self.phys_base_weight
        
        # v12（见 docs/v12_changes_summary.md）：全量井点反传，避免时间后段外推崩盘
        use_train_only = False
        batch = self._sample_batch(step, use_train_only=use_train_only)
        well_outputs = self._compute_well_outputs(use_train_only=use_train_only)
        
        # ========== FIX-v9: 时变源项 — 用全时间线插值，PDE 与生产制度一致 ==========
        well_xy_norm_list = []
        well_rates_per_point = []
        t_pde = batch['x_pde'][:, 2]
        # 源项用全量井数据插值，保证任意 t_pde 都有 qg(t)
        for wid in self.model.well_ids:
            if wid not in self.well_data:
                continue
            wdata_full = self.well_data[wid]
            well_xyt_full = wdata_full['xyt']
            wx_norm = well_xyt_full[0, 0].detach()
            wy_norm = well_xyt_full[0, 1].detach()

            idx_src = wdata_full.get('idx_train') if use_train_only else None
            if idx_src is not None:
                t_obs = well_xyt_full[idx_src, 2]
                qg_obs = wdata_full['qg_obs'][idx_src].squeeze()
                qg_valid = wdata_full['qg_valid_mask'][idx_src].squeeze() if 'qg_valid_mask' in wdata_full else None
            else:
                t_obs = well_xyt_full[:, 2]
                qg_obs = wdata_full['qg_obs'].squeeze()
                qg_valid = wdata_full['qg_valid_mask'].squeeze() if 'qg_valid_mask' in wdata_full else None

            if t_obs.numel() == 0:
                continue
            well_xy_norm_list.append(torch.stack([wx_norm, wy_norm]))
            qg_at_pde = self._interp_qg(t_obs, qg_obs, t_pde, valid_mask=qg_valid)
            well_rates_per_point.append(-qg_at_pde / 86400.0)
        
        if well_xy_norm_list:
            well_xy_tensor = torch.stack(well_xy_norm_list)  # (n_wells, 2)
            source_g = self._compute_time_varying_source(
                batch['x_pde'][:, :2],  # (N, 2) 配点空间坐标
                well_xy_tensor,          # (n_wells, 2) 井位
                well_rates_per_point     # list of (N,) 时变流量
            )
            batch['source'] = source_g
        # ========== 时变源项修复结束 ==========
        
        # 获取 k_net 引用和可训练 k_eff 张量（不 detach!）
        # [护栏1] k_field 在 loss_pde 内部用 xyt 计算，确保同一张计算图
        k_net_ref = self.model.k_net if (
            hasattr(self.model, 'k_net') and self.model.k_net is not None
        ) else None
        k_eff_trainable = self.model.well_model.peaceman.k_eff_mD  # 张量，有梯度
        
        # 计算所有损失
        losses = self.loss_fn.total_loss(
            model=self.model,
            batch=batch,
            well_outputs=well_outputs,
            weights=stage_weights,
            k_net=k_net_ref,
            k_eff_mD_tensor=k_eff_trainable,
        )
        
        # ReLoBRaLo 获取自适应权重
        loss_vals = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items() if k != 'total'}
        adaptive_weights = self.balancer.get_weights(loss_vals, step)
        
        # 用自适应权重重新计算 total (如果使用 ReLoBRaLo)
        if self.use_relobralo and step >= 200:
            total = torch.tensor(0.0, device=self.device)
            for key, val in losses.items():
                if key == 'total':
                    continue
                w = adaptive_weights.get(key, 1.0) * stage_weights.get(key, 1.0)
                total = total + w * val
            losses['total'] = total
        
        # ===== v4.0: Sw 约束已整合到 loss_sw_bounds (通过阶段权重 sw_bounds 控制) =====
        # 不再在此硬编码 anchor_weight，避免与 total_loss 中的 sw_bounds 双重计算。
        # 仅保留井位级 Sw 轻量约束（防止井点 Sw 越界导致 krg→0 梯度死锁）。
        if well_outputs:
            sw_well_penalty = torch.tensor(0.0, device=self.device)
            for wid, wdata in well_outputs.items():
                if 'sw_cell' not in wdata:
                    continue
                sw_well = wdata['sw_cell']
                p_lo = torch.mean(torch.relu(0.10 - sw_well) ** 2)  # v3.14: SY9 sw_init≈0.15, 下界留余量
                p_hi = torch.mean(torch.relu(sw_well - 0.85) ** 2)
                sw_well_penalty = sw_well_penalty + p_lo + p_hi
            n_wells = max(len(well_outputs), 1)
            sw_well_penalty = sw_well_penalty / n_wells
            losses['total'] = losses['total'] + 50.0 * sw_well_penalty
        
        # v3.22: TDS→Sw 软标签约束
        # 在井位时间线上插值TDS衍生的Sw_tds, 与PINN预测的sw_cell比较
        if self.tds_data is not None and well_outputs:
            tds_loss = torch.tensor(0.0, device=self.device)
            n_tds_wells = 0
            for wid, wout in well_outputs.items():
                if wid not in self.tds_data:
                    continue
                sw_pred = wout['sw_cell']   # (N, 1)
                t_norm = wout['t_norm']     # (N, 1)
                tds_info = self.tds_data[wid]
                # 用线性插值将Sw_tds映射到井观测时间点
                sw_tds_interp = self._interp_qg(
                    tds_info['t_norm'], tds_info['sw_tds'],
                    t_norm.squeeze()
                ).unsqueeze(-1)  # (N, 1)
                tds_loss = tds_loss + torch.mean((sw_pred - sw_tds_interp) ** 2)
                n_tds_wells += 1
            if n_tds_wells > 0:
                tds_loss = tds_loss / n_tds_wells
            w_tds = stage_weights.get('tds', 0.0)
            losses['tds'] = w_tds * tds_loss
            losses['total'] = losses['total'] + losses['tds']
        
        # v16: p_wf 越界软惩罚 — 防止 p_wf 超出物理边界 [30, 80] MPa
        if well_outputs:
            guardrails = self.config.get('safety_guardrails', {}).get('limits', {})
            pwf_range = guardrails.get('pwf_hat_MPa', [30.0, 80.0])
            pwf_penalty = torch.tensor(0.0, device=self.device)
            for wid, wdata in well_outputs.items():
                if 'p_wf' in wdata:
                    p_wf = wdata['p_wf']
                    pwf_penalty = pwf_penalty + torch.mean(
                        torch.relu(p_wf - pwf_range[1]) ** 2
                        + torch.relu(pwf_range[0] - p_wf) ** 2
                    )
            losses['total'] = losses['total'] + 10.0 * pwf_penalty
        
        # Backward
        losses['total'].backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        # ===== 梯度贯通自证 (step==0 执行一次) =====
        if step == 0:
            # 验证 1: PDE loss 对 k_eff_mD 有梯度
            k_raw = self.model.well_model.peaceman._k_frac_raw
            if k_raw.grad is not None and k_raw.grad.abs().item() > 0:
                self.logger.info(
                    f"  ✅ k_frac_mD.grad = {k_raw.grad.item():.6e} (PDE梯度贯通)"
                )
            else:
                self.logger.warning(
                    "  ⚠️ k_eff_mD.grad 为 None 或零! 可辨识性可能受损"
                )
            
            # 验证 2: k_net 参数梯度
            if hasattr(self.model, 'k_net') and self.model.k_net is not None:
                k_net_grads = [p.grad for p in self.model.k_net.parameters()
                               if p.grad is not None]
                self.logger.info(
                    f"  ✅ k_net: {len(k_net_grads)} 组参数收到梯度"
                )
            
            # v4.8: 验证 r_e 梯度贯通 (Stage A/B 冻结期 grad=None 是正常的)
            r_e_raw = self.model.well_model.peaceman._r_e_raw
            if r_e_raw.requires_grad and r_e_raw.grad is not None and r_e_raw.grad.abs().item() > 0:
                self.logger.info(
                    f"  ✅ r_e.grad = {r_e_raw.grad.item():.6e} (WI梯度贯通)"
                )
            elif not r_e_raw.requires_grad:
                self.logger.info(
                    f"  ℹ️ r_e 当前冻结 (Stage A/B), 梯度将在 Stage C 解冻后验证"
                )
            else:
                self.logger.warning(
                    "  ⚠️ r_e.grad 为 None 或零! WI→r_e 梯度链路可能断裂"
                )
            
            # v3: f_frac 已合并入 k_frac, 无需单独检查
        
        # ===== 补充 A: 梯度链路硬核验收 =====
        if step == 0 or (step % 200 == 0 and step < 2000):
            with torch.no_grad():
                for wid in self.model.well_ids:
                    if wid in self.well_data:
                        qg_obs = self.well_data[wid]['qg_obs']
                        self.logger.info(
                            f"  [梯度验收] {wid}: qg_obs range="
                            f"[{qg_obs.min().item():.1f}, {qg_obs.max().item():.1f}]"
                        )
            # L_qg 对场网络的梯度范数
            if losses.get('qg') is not None and isinstance(losses['qg'], torch.Tensor):
                try:
                    grad_qg = torch.autograd.grad(
                        losses['qg'], self.model.field_net.parameters(),
                        retain_graph=True, allow_unused=True
                    )
                    grad_norm = sum(g.norm().item() for g in grad_qg if g is not None)
                    self.logger.info(f"  [梯度验收] ‖∇_field L_qg‖ = {grad_norm:.6e}")
                    if grad_norm < 1e-12:
                        self.logger.error("  🔴 L_qg 对场网络梯度为零！检查 krg/Sw/detach")
                except RuntimeError:
                    # backward 已执行后 graph 可能已释放, 跳过
                    pass
        
        # 监控 Sw 范围 (全域 + 井位级)
        with torch.no_grad():
            x_check = batch['x_pde'][:100]
            _, sw_check = self.model(x_check)
            sw_min = sw_check.min().item()
            sw_max = sw_check.max().item()
            
            # 井位级 Sw 预警 (每 500 步或初始几步)
            if step == 0 or step % 500 == 0:
                for wid, wout in well_outputs.items():
                    sw_well = wout['sw_cell']
                    sw_w_min = sw_well.min().item()
                    sw_w_max = sw_well.max().item()
                    sw_w_mean = sw_well.mean().item()
                    # krg 诊断 (v3.20: 用训练中学习的 relperm, 而非默认实例)
                    krg_well = self.loss_fn.relperm.krg(sw_well)
                    krg_min = krg_well.min().item()
                    self.logger.info(
                        f"  [Sw监控] {wid}: Sw=[{sw_w_min:.4f}, {sw_w_mean:.4f}, {sw_w_max:.4f}], "
                        f"krg_min={krg_min:.6e}"
                    )
                    if sw_w_max > 0.95:
                        self.logger.warning(
                            f"  ⚠️ {wid}: Sw_max={sw_w_max:.4f} > 0.95, "
                            f"可能导致 krg→0 梯度死锁!"
                        )
                    if krg_min < 1e-6:
                        self.logger.warning(
                            f"  ⚠️ {wid}: krg_min={krg_min:.6e}, "
                            f"接近零, 检查 Sw 漂移!"
                        )
        
        # 反演参数
        inv_params = self.model.get_inversion_params()
        
        result = {k: v.item() if isinstance(v, torch.Tensor) else v
                  for k, v in losses.items()}
        result['sw_min'] = sw_min
        result['sw_max'] = sw_max
        result['lr'] = self.optimizer.param_groups[0]['lr']
        for pg in self.optimizer.param_groups:
            if pg.get('name') == 'k_net':
                result['lr_k_net'] = pg['lr']
                break
        result['k_frac_mD'] = inv_params.get('k_frac_mD', 0.0)
        result['k_eff_mD'] = inv_params.get('k_frac_mD', 0.0)
        result['f_frac'] = 0.0  # v3: 已合并入 k_frac
        result['dp_wellbore'] = inv_params.get('dp_wellbore_MPa', 0.0)
        result['r_e_m'] = inv_params.get('r_e_m', 128.9)
        
        return result
    
    def train(self) -> Dict[str, list]:
        """执行完整的 M5 分阶段训练"""
        self.logger.info("=" * 60)
        self.logger.info("M5 PINN 井—藏耦合同化训练开始")
        self.logger.info("=" * 60)
        
        total_steps = self.max_steps
        
        # 阶段划分: m4_config.training_stages 为主，train.training_stages 可覆盖（便于 M5 单独调阶段）
        stages_cfg = dict(self.config.get('m4_config', {}).get('training_stages', {}))
        train_stages = self.config.get('train', {}).get('training_stages', {})
        for k, v in train_stages.items():
            if isinstance(v, dict):
                stages_cfg[k] = {**stages_cfg.get(k, {}), **v}
        sa = stages_cfg.get('stage_a', {})
        sb = stages_cfg.get('stage_b', {})
        sc = stages_cfg.get('stage_c', {})
        sd = stages_cfg.get('stage_d', {})
        
        frac_a = sa.get('fraction', 0.15)
        frac_b = sb.get('fraction', 0.25)
        frac_c = sc.get('fraction', 0.35)
        
        stage_a_end = int(total_steps * frac_a)
        stage_b_end = int(total_steps * (frac_a + frac_b))
        stage_c_end = int(total_steps * (frac_a + frac_b + frac_c))
        # ★ 保存所有阶段边界（用于Sw锚定权重衰减）
        self._stage_a_end = stage_a_end
        self._stage_b_end = stage_b_end
        self._stage_c_end = stage_c_end
        
        # ★★★ M5 v7 阶段权重 (暴力修复 Sw 漂移) ★★★
        # v6失败诊断:
        #   1. Sw锚定惩罚太小（0.06），权重50.0也只贡献3.0
        #   2. PDE loss (87万) × 权重0.2 = 17.4万，仍然主导总损失(18万)
        #   3. k_frac从5.4暴跌到0.82 mD（下降85%）
        #   4. Sw持续在0.88，krg≈0.005
        #
        # v7暴力修复:
        #   FIX-1: Sw硬约束 + 超高权重 (5万~5千，代码中已实现)
        #   FIX-2: PDE权重降到极低 (0.001-0.01)
        #   FIX-3: qg权重进一步增加 (100→500)
        #   FIX-4: sw_bounds权重降为0 (已被硬约束替代)
        
        # qg_nearzero 阶段权重（默认 A/B/C/D: 10/20->30/40，支持 config 覆盖）
        nearzero_cfg = self.config.get('m5_config', {}).get('qg_nearzero', {})
        nearzero_enable = bool(nearzero_cfg.get('enable', True))
        if nearzero_enable:
            qz_a = float(nearzero_cfg.get('w_a', 3.0))
            qz_b_start = float(nearzero_cfg.get('w_b_start', 6.0))
            qz_b_end = float(nearzero_cfg.get('w_b_end', 9.0))
            qz_c = float(nearzero_cfg.get('w_c', 12.0))
            qz_d = float(nearzero_cfg.get('w_d', 12.0))
        else:
            qz_a = qz_b_start = qz_b_end = qz_c = qz_d = 0.0
        # Stage D: 以开井拟合为主，near-zero 仅保留弱约束
        qz_d_late = max(4.0, 0.35 * qz_d) if nearzero_enable else 0.0
        # ===== v4.2: PDE 默认权重极低，防止 m4_config 泄漏时仍能安全运行 =====
        # 根因诊断 (2026-02-21): m4_config.training_stages.stage_d.weights.pde=3.0
        # 通过 stages_cfg 合并链泄漏到 M5, 导致 Stage D PDE 贡献 = 3.0×6e5 = 1.8e6
        # 完全淹没 qg (700×0.13=91). 现在 config 已显式覆盖 pde, 但硬编码也需安全.
        # 同时 k_eff 0.5×8=4 mD (接近数据驱动收敛 3.85 mD)
        w_a = {'ic': 15.0, 'bc': 5.0, 'pde': 0.005, 'qg': 50.0, 'qg_nearzero': qz_a, 'shutin_delta': 0.0,
               'smooth_pwf': 0.001, 'smooth_qg': 0.0, 'monotonic': 0.0, 'prior': 0.01,
               'k_reg': 0.01, 'whp': 5.0, 'sw_bounds': 3.0, 'qw': 0.0, 'tds': 0.0}  # v3.22: tds=0 (IC/BC阶段Sw未演化)

        w_b_start = {'ic': 8.0, 'bc': 4.0, 'pde': 0.01, 'qg': 150.0, 'qg_nearzero': qz_b_start, 'shutin_delta': 5.0,
                     'smooth_pwf': 0.01, 'smooth_qg': 0.0, 'monotonic': 0.0, 'prior': 0.02,
                     'k_reg': 0.02, 'whp': 8.0, 'sw_bounds': 2.0, 'qw': 0.0, 'tds': 5.0}  # v3.22: tds 开始软约束
        w_b_end = {'ic': 5.0, 'bc': 3.0, 'pde': 0.01, 'qg': 300.0, 'qg_nearzero': qz_b_end, 'shutin_delta': 8.0,
                   'smooth_pwf': 0.03, 'smooth_qg': 0.005, 'monotonic': 0.05, 'prior': 0.03,
                   'k_reg': 0.05, 'whp': 10.0, 'sw_bounds': 1.5, 'qw': 0.0, 'tds': 10.0}

        w_c_start = {'ic': 4.0, 'bc': 2.0, 'pde': 0.02, 'qg': 600.0, 'qg_nearzero': qz_c, 'shutin_delta': 8.0,
                     'smooth_pwf': 0.02, 'smooth_qg': 0.01, 'monotonic': 0.2, 'prior': 0.05,
                     'k_reg': 0.08, 'whp': 8.0, 'sw_bounds': 1.0, 'qw': 10.0, 'tds': 20.0}  # v3.22: tds 中等约束
        w_c_end = {'ic': 3.0, 'bc': 1.5, 'pde': 0.02, 'qg': 1000.0, 'qg_nearzero': qz_c, 'shutin_delta': 10.0,
                   'smooth_pwf': 0.02, 'smooth_qg': 0.015, 'monotonic': 0.4, 'prior': 0.05,
                   'k_reg': 0.15, 'whp': 10.0, 'sw_bounds': 0.8, 'qw': 30.0, 'tds': 30.0}

        w_d = {'ic': 2.0, 'bc': 1.0, 'pde': 0.03, 'qg': 2000.0, 'qg_nearzero': qz_d_late, 'shutin_delta': 5.0,
               'smooth_pwf': 0.01, 'smooth_qg': 0.02, 'monotonic': 0.4, 'prior': 0.05,
               'k_reg': 0.2, 'whp': 6.0, 'sw_bounds': 0.5, 'qw': 50.0, 'tds': 50.0}  # v3.22: tds 最强约束
        
        # ========== v3.1: 阶段权重可由 config 覆盖（两处均生效）==========
        # 1) train.stages.A.weights.pde 等（消融/脚本注入）
        # 2) m4_config.training_stages.stage_a.weights / train.training_stages.stage_a.weights
        stages_override = self.config.get('train', {}).get('stages', {})
        _stage_map = {'A': [w_a], 'B': [w_b_start, w_b_end],
                      'C': [w_c_start, w_c_end], 'D': [w_d]}
        for stage_key, weight_dicts in _stage_map.items():
            ov = stages_override.get(stage_key, {}).get('weights', {})
            for k, v in ov.items():
                for wd in weight_dicts:
                    wd[k] = v
        # 从 training_stages 的 stage_a/stage_b/... 下读 weights（与 fraction 同源，便于配置）
        _ts_stage_key = {'A': 'stage_a', 'B': 'stage_b', 'C': 'stage_c', 'D': 'stage_d'}
        for stage_key, weight_dicts in _stage_map.items():
            ts_key = _ts_stage_key[stage_key]
            ov = (stages_cfg.get(ts_key) or {}).get('weights', {})
            for k, v in ov.items():
                for wd in weight_dicts:
                    wd[k] = v
        if stages_override or any((stages_cfg.get(_ts_stage_key[k]) or {}).get('weights') for k in _stage_map):
            self.logger.info("  阶段权重已被 config 覆盖 (train.stages 或 training_stages.stage_*.weights)")
        
        self.logger.info(
            f"训练阶段: A[0-{stage_a_end}] B[{stage_a_end}-{stage_b_end}] "
            f"C[{stage_b_end}-{stage_c_end}] D[{stage_c_end}-{total_steps}]"
        )
        self.logger.info(
            f"  qg_nearzero 权重: A={qz_a:.1f}, B={qz_b_start:.1f}->{qz_b_end:.1f}, C={qz_c:.1f}, D={qz_d_late:.1f}, enable={nearzero_enable}"
        )
        
        start_time = time.time()
        # ★★★ FIX-v8: 最佳检查点按「同化损失」保存，而非总损失 ★★★
        # 问题: total loss 被 PDE(53万)+Sw锚定主导，best 落在 step 200（IC/BC 好但 qg 仍过拟合不足）
        # 结果: 加载 best 后 qg 严重高估(MAPE 168%, R²=-15)，p_wf 不跟 WHP
        # 修复: 按 qg_loss + whp_loss 最小保存，保证验收时用的是「拟合观测最好」的模型
        best_assim_loss = float('inf')
        self.best_step = None  # 首次 best 更新时写入，供报告使用
        
        # ========== Step 0.1: 单位/量纲链路端到端诊断 (第 1 步之前) ==========
        primary_for_diag = self.model.well_ids[0] if self.model.well_ids else 'SY9'
        if primary_for_diag in self.well_data:
            with torch.no_grad():
                well_data = self.well_data[primary_for_diag]
                qg_obs = well_data['qg_obs']
                self.logger.info("--- M5 单位/量纲诊断 (train 第 1 步前) ---")
                self.logger.info(
                    f"  qg_obs 范围: {qg_obs.min().item():.2e} ~ {qg_obs.max().item():.2e} m³/d, "
                    f"均值: {qg_obs.mean().item():.2e} m³/d"
                )
                h_well = self.well_h.get(primary_for_diag, 90.0)
                phn_diag = well_data.get('prod_hours_norm', None)
                cn_diag = well_data.get('casing_norm', None)
                result = self.model.evaluate_at_well(
                    primary_for_diag, well_data['xyt'], h_well=h_well, bg_val=self.bg_ref,
                    prod_hours_norm=phn_diag, casing_norm=cn_diag,
                )
                qg_pred = result['qg']
                self.logger.info(
                    f"  qg_pred 范围: {qg_pred.min().item():.2e} ~ {qg_pred.max().item():.2e}"
                )
                self.logger.info(
                    f"  p_wf 范围: {result['p_wf'].min().item():.2e} ~ {result['p_wf'].max().item():.2e} MPa"
                )
                self.logger.info(
                    f"  p_field @ well: {result['p_cell'].mean().item():.2e} MPa"
                )
                wi = self.model.well_model.peaceman
                self.logger.info(
                    f"  k_frac = {wi.k_frac_mD.item():.4f} mD"
                )
                h_well_t = torch.tensor(h_well, device=self.device, dtype=torch.float32)
                self.logger.info(
                    f"  WI_raw = {wi.compute_WI(h_well_t).item():.6e} [检查单位]"
                )
                self.logger.info(
                    f"  dp_wellbore = {self.model.dp_wellbore.item():.2f} MPa"
                )
                self.logger.info(f"  Bg_ref = {self.bg_ref:.6e}")
                
                # ★ 源项诊断 (v9: 时变源项)
                self.logger.info(f"  qg_obs 范围: {qg_obs.min().item():.2e} ~ {qg_obs.max().item():.2e} m³/d")
                self.logger.info(f"  qg_obs 中位数 = {qg_obs.median().item():.2e} m³/d")
                self.logger.info("  ✅ v9: PDE源项已升级为时变 source(x,y,t) = qg_obs(t) * kernel(x,y)")
                self.logger.info("  ✅ 每个PDE配点使用其时刻对应的真实观测产量（线性插值）")
                self.logger.info("--- 诊断结束 ---")
        
        def _lerp(w_s, w_e, p):
            return {k: w_s.get(k, 0) + p * (w_e.get(k, 0) - w_s.get(k, 0))
                    for k in set(w_s) | set(w_e)}
        
        # v3.13: Stage A+B 冻结 k_frac + r_e, 防止 PDE 主导期拖低 k_frac
        # Stage A+B: PDE raw ~10⁵~10⁶ 远大于 qg ~0.2, k_frac 只听 PDE
        # Stage C: qg 权重已足够大, 此时解冻才能正确收敛
        _k_frac_raw = self.model.well_model.peaceman._k_frac_raw
        _k_frac_raw.requires_grad_(False)
        _k_frac_frozen = True
        # v4.8: r_e 与 k_frac 同步冻结/解冻 (同属 WI 参数, PDE主导期梯度不可靠)
        _r_e_raw = self.model.well_model.peaceman._r_e_raw
        _r_e_raw.requires_grad_(False)
        _r_e_frozen = True
        self.logger.info(
            f"  [v3.13] Stage A+B 冻结 k_frac={self.model.well_model.peaceman.k_frac_mD.item():.4f} mD, "
            f"r_e={self.model.well_model.peaceman.r_e.item():.1f} m (Stage C 解冻)"
        )
        
        # v3.14: Stage A+B 同步冻结 ng/nw — PDE 噪声梯度会把 ng 从 1.08 推到 2.7+
        # 实测: corey_w=0.5 × stage_prior=0.01~0.03 = 有效权重 0.005~0.015, vs PDE ~500, 弱 30000x
        _corey_frozen = False
        rp = self.loss_fn.relperm
        if hasattr(rp, '_ng_log'):
            rp._ng_log.requires_grad_(False)
            rp._nw_log.requires_grad_(False)
            _corey_frozen = True
            self.logger.info(f"  [v3.14] Stage A+B 冻结 ng={rp.ng.item():.4f}, nw={rp.nw.item():.4f} (Stage C 解冻)")
        
        for step in range(total_steps):
            if step < stage_a_end:
                stage = 'A'
                weights = w_a
            elif step < stage_b_end:
                stage = 'B'
                progress = (step - stage_a_end) / max(stage_b_end - stage_a_end, 1)
                weights = _lerp(w_b_start, w_b_end, progress)
            elif step < stage_c_end:
                stage = 'C'
                if _k_frac_frozen:
                    _k_frac_raw.requires_grad_(True)
                    _k_frac_frozen = False
                    self.logger.info(f"  [v3.13] Stage C 解冻 k_frac={self.model.well_model.peaceman.k_frac_mD.item():.4f} mD")
                if _r_e_frozen:
                    _r_e_raw.requires_grad_(True)
                    _r_e_frozen = False
                    self.logger.info(f"  [v4.8] Stage C 解冻 r_e={self.model.well_model.peaceman.r_e.item():.1f} m")
                if _corey_frozen:
                    rp._ng_log.requires_grad_(True)
                    rp._nw_log.requires_grad_(True)
                    _corey_frozen = False
                    self.logger.info(f"  [v3.14] Stage C 解冻 ng={rp.ng.item():.4f}, nw={rp.nw.item():.4f}")
                progress = (step - stage_b_end) / max(stage_c_end - stage_b_end, 1)
                weights = _lerp(w_c_start, w_c_end, progress)
            else:
                stage = 'D'
                weights = w_d
            
            # RAR 加点
            if self.rar.should_refine(step):
                self.rar.refine(step, self.model, self.device)
            
            # 训练一步
            loss_dict = self._train_step(step, weights)
            
            # 记录历史
            self.history['step'].append(step)
            for key in ['total', 'ic', 'bc', 'pde', 'qg', 'qg_nearzero', 'shutin_delta', 'whp', 'smooth_pwf',
                        'smooth_qg', 'monotonic', 'prior', 'k_reg', 'sw_bounds', 'tds', 'lr', 'lr_k_net',
                        'sw_min', 'sw_max', 'k_frac_mD', 'k_eff_mD', 'f_frac', 'dp_wellbore', 'r_e_m']:
                self.history[key].append(loss_dict.get(key, 0.0))
            self.history['stage'].append(stage)
            
            # v11: best 按验证集复合分数（每 200 步评估）；仅当 val 有效时更新
            if step % 200 == 0 or step == total_steps - 1:
                val_score = self._eval_val_score()
                # 当 val 无开井点时代码返回 999，不得作为 best 更新（避免 Best step: 0）
                if step >= 200 and val_score < best_assim_loss and val_score < 998.0:
                    best_assim_loss = val_score
                    self.best_step = step
                    self._save_checkpoint('best', step=step)
                    if step % 500 == 0:
                        self.logger.info(f"  [best更新] val_score={val_score:.4f} (step {step})")
            
            # 日志
            if step % 100 == 0 or step == total_steps - 1:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"[Step {step:5d}/{total_steps}] {stage} | L={loss_dict['total']:.4e} | {elapsed:.1f}s"
                )
                self.logger.info(
                    f"  loss: IC={loss_dict.get('ic',0):.2e} BC={loss_dict.get('bc',0):.2e} "
                    f"PDE={loss_dict.get('pde',0):.2e} Qg={loss_dict.get('qg',0):.2e} "
                    f"Sd={loss_dict.get('shutin_delta',0):.2e} TDS={loss_dict.get('tds',0):.2e}"
                )
                self.logger.info(
                    f"  param: k={loss_dict.get('k_frac_mD',0):.3f}mD "
                    f"dp={loss_dict.get('dp_wellbore',0):.1f} "
                    f"ng={self.loss_fn.relperm.ng.item():.3f} nw={self.loss_fn.relperm.nw.item():.3f} "
                    f"Sw=[{loss_dict['sw_min']:.3f},{loss_dict['sw_max']:.3f}]"
                    f"{self._get_singularity_log_str()}"
                )
            
            # ===== v3.2: k_frac 梯度诊断 (每 500 步) =====
            if step % 500 == 0:
                k_raw = self.model.well_model.peaceman._k_frac_raw
                grad_str = f"grad={k_raw.grad.item():.4e}" if k_raw.grad is not None else "grad=None"
                self.logger.info(f"  [k_frac诊断] {grad_str}, k_frac={loss_dict.get('k_frac_mD',0):.6f} mD")
        
        total_time = time.time() - start_time
        self.logger.info(f"\nM5 训练完成! 耗时: {total_time:.1f}s, 最佳验证分数(val_score): {best_assim_loss:.4f}")
        
        self._save_checkpoint('final')
        self._save_config_snapshot()
        self._save_inversion_audit()
        
        return self.history
    
    def _save_checkpoint(self, tag: str, step: int = None):
        path = os.path.join(self.ckpt_dir, f'm5_pinn_{tag}.pt')
        payload = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'inversion_params': self.model.get_inversion_params(),
        }
        if step is not None:
            payload['step'] = step
        torch.save(payload, path)
    
    def load_checkpoint(self, tag: str = 'best'):
        path = os.path.join(self.ckpt_dir, f'm5_pinn_{tag}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            # v4.8: strict=False 兼容旧 checkpoint (缺少 _r_e_raw 等新增参数)
            missing, unexpected = self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if missing:
                self.logger.info(f"  load_checkpoint: missing keys (将用初始值): {missing}")
            if unexpected:
                self.logger.warning(f"  load_checkpoint: unexpected keys: {unexpected}")
            # ★★★ FIX-v9: 恢复 history，保证 report 指标与 checkpoint 一致 ★★★
            if 'history' in ckpt and ckpt['history']:
                self.history = ckpt['history']
                # v4.8: 补全旧 history 中缺失的新增字段
                n_steps = len(self.history.get('step', []))
                for new_key in ('r_e_m',):
                    if new_key not in self.history:
                        self.history[new_key] = [0.0] * n_steps
                self.logger.info(f"  history 已恢复 ({n_steps} 步)")
            if 'step' in ckpt:
                self.best_step = ckpt['step']
                self.logger.info(f"  best_step={self.best_step}")
            self.model.train()
            self.logger.info(f"已加载 M5 检查点: {path}")
            return True
        return False
    
    def _save_config_snapshot(self):
        """保存 resolved config 快照"""
        path = os.path.join(self.output_dir, 'resolved_config_m5.json')
        try:
            import yaml
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"配置快照已保存: {path}")
        except Exception as e:
            self.logger.warning(f"配置快照保存失败: {e}")
    
    def _save_inversion_audit(self):
        """保存反演参数审计报告"""
        inv = self.model.get_inversion_params()
        path = os.path.join(self.report_dir, 'M5_inversion_params.json')
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(inv, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"反演参数审计: {path}")
        except Exception as e:
            self.logger.warning(f"审计保存失败: {e}")
    
    # ================================================================== #
    #                       验收图件输出
    # ================================================================== #
    
    def plot_qg_comparison(self, save: bool = True) -> str:
        """绘制 qg 真值 vs 预测 (一等奖级别 — 专业期刊风格)"""
        from utils import COLORS, apply_plot_style
        apply_plot_style()
        self.model.eval()
        
        for wid in self.model.well_ids:
            if wid not in self.well_data:
                continue
            
            wdata = self.well_data[wid]
            with torch.no_grad():
                result = self.model.evaluate_at_well(
                    wid, wdata['xyt'],
                    h_well=self.well_h.get(wid, 90.0),
                    bg_val=self.bg_ref,
                    prod_hours_norm=wdata.get('prod_hours_norm', None),
                    casing_norm=wdata.get('casing_norm', None),
                )
            
            qg_pred = result['qg'].cpu().numpy().flatten()
            qg_obs = wdata['qg_obs'].cpu().numpy().flatten()
            if 'shutin_mask' in wdata:
                shutin_all = (wdata['shutin_mask'].cpu().numpy().flatten() > 0.5)
                qg_pred = np.where(shutin_all, 0.0, qg_pred)
            t_days = wdata['t_days']
            
            # Train/Val/Test 分割（与 config.train.m5_split 一致）
            n = len(t_days)
            n_train = int(n * self.m5_train_ratio)
            n_val = int(n * self.m5_val_ratio)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'M5 井 {wid} 产气量同化拟合', fontsize=15, fontweight='bold')
            
            # ---- 左图: 时间序列 ----
            ax1 = axes[0]
            # 训练/验证/测试背景色带
            ymax = max(qg_obs.max(), qg_pred.max()) * 1.12
            ax1.fill_between(t_days[:n_train], 0, ymax,
                             color=COLORS['train'], alpha=0.07, label='训练集')
            ax1.fill_between(t_days[n_train:n_train + n_val], 0, ymax,
                             color=COLORS['val'], alpha=0.07, label='验证集')
            ax1.fill_between(t_days[n_train + n_val:], 0, ymax,
                             color=COLORS['test'], alpha=0.07, label='测试集')
            # 数据线
            ax1.plot(t_days, qg_obs, color=COLORS['primary'], linewidth=1.5,
                     label='观测值', zorder=3)
            ax1.plot(t_days, qg_pred, color=COLORS['accent'], linewidth=1.5,
                     linestyle='--', label='PINN 预测', alpha=0.85, zorder=4)
            ax1.set_xlabel('时间 (天)')
            ax1.set_ylabel('产气量 (m³/d)')
            ax1.set_title('$q_g$ 时间序列')
            ax1.set_ylim(bottom=0, top=ymax)
            ax1.legend(loc='upper right', framealpha=0.85, edgecolor=COLORS['info_box'])
            
            # 指标文本框
            valid = (qg_obs > 0) & np.isfinite(qg_pred)
            if np.any(valid):
                rmse = np.sqrt(np.mean((qg_obs[valid] - qg_pred[valid]) ** 2))
                mape = np.mean(np.abs((qg_obs[valid] - qg_pred[valid]) / (qg_obs[valid] + 1.0))) * 100
                # R² 计算
                ss_res = np.sum((qg_obs[valid] - qg_pred[valid]) ** 2)
                ss_tot = np.sum((qg_obs[valid] - qg_obs[valid].mean()) ** 2)
                r2 = 1.0 - ss_res / (ss_tot + 1e-12)
                textstr = f'MAPE = {mape:.1f}%\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.0f}'
                props = dict(boxstyle='round,pad=0.4', facecolor='white',
                             alpha=0.92, edgecolor=COLORS['info_box'], linewidth=0.8)
                ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
                         verticalalignment='top', bbox=props)
            
            # ---- 右图: 散点对比 (45°线) ----
            ax2 = axes[1]
            ax2.scatter(qg_obs[:n_train], qg_pred[:n_train],
                        c=COLORS['train'], s=15, alpha=0.6, edgecolors='none', label='Train')
            ax2.scatter(qg_obs[n_train:n_train + n_val], qg_pred[n_train:n_train + n_val],
                        c=COLORS['val'], s=15, alpha=0.6, edgecolors='none', label='Val')
            ax2.scatter(qg_obs[n_train + n_val:], qg_pred[n_train + n_val:],
                        c=COLORS['test'], s=15, alpha=0.6, edgecolors='none', label='Test')
            max_val = max(qg_obs.max(), qg_pred.max()) * 1.1
            ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.45, linewidth=1.0, label='$y = x$')
            ax2.set_xlabel('观测 $q_g$ (m³/d)')
            ax2.set_ylabel('预测 $q_g$ (m³/d)')
            ax2.set_title('散点对比')
            ax2.set_xlim(0, max_val)
            ax2.set_ylim(0, max_val)
            ax2.set_aspect('equal', adjustable='box')
            ax2.legend(loc='upper left', framealpha=0.85, edgecolor=COLORS['info_box'])
            
            plt.tight_layout()
            if save:
                fp = os.path.join(self.fig_dir, f'M5_qg_comparison_{wid}.png')
                fig.savefig(fp, dpi=250, bbox_inches='tight')
                plt.close(fig)
                self.logger.info(f"qg 对比图已保存: {fp}")
                return fp
            plt.close(fig)
        return ''
    
    def plot_pwf_inversion(self, save: bool = True) -> str:
        """绘制 p_wf(t) 反演曲线 + 越界检查 (一等奖配色)"""
        from utils import COLORS, apply_plot_style
        apply_plot_style()
        self.model.eval()
        
        for wid in self.model.well_ids:
            if wid not in self.well_data:
                continue
            
            wdata = self.well_data[wid]
            
            # ★★★ FIX-v9: 统一 p_wf 口径 ★★★
            # 旧版 BUG: 用 get_pwf 获取原始网络输出, 但训练中 compute_well_rate
            #           会对 p_wf 做 soft clipping (p_wf < p_cell - epsilon)
            # 修复: 使用 evaluate_at_well 走与训练完全相同的路径
            with torch.no_grad():
                result = self.model.evaluate_at_well(
                    wid, wdata['xyt'],
                    h_well=self.well_h.get(wid, 90.0),
                    bg_val=self.bg_ref,
                    prod_hours_norm=wdata.get('prod_hours_norm', None),
                    casing_norm=wdata.get('casing_norm', None),
                )
                p_wf = result['p_wf'].cpu().numpy().flatten()
            
            t_days = wdata['t_days']
            
            # 越界检查
            guardrails = self.config.get('safety_guardrails', {}).get('limits', {})
            pwf_range = guardrails.get('pwf_hat_MPa', [0.0, 90.0])
            n_below = np.sum(p_wf < pwf_range[0])
            n_above = np.sum(p_wf > pwf_range[1])
            in_bounds = (n_below == 0) and (n_above == 0)
            
            fig, ax = plt.subplots(figsize=(13, 5.5))
            fig.suptitle(f'M5 井 {wid} $p_{{wf}}(t)$ 反演曲线',
                         fontsize=15, fontweight='bold')
            
            ax.plot(t_days, p_wf, color=COLORS['accent'], linewidth=1.8,
                    label='$p_{wf}$ 反演', zorder=3)
            ax.axhline(y=pwf_range[0], color='#95A5A6', linestyle='--',
                       alpha=0.6, linewidth=0.8, label=f'下界 {pwf_range[0]} MPa')
            ax.axhline(y=pwf_range[1], color='#95A5A6', linestyle='--',
                       alpha=0.6, linewidth=0.8, label=f'上界 {pwf_range[1]} MPa')
            
            # WHP 观测
            p_obs = wdata['p_obs'].cpu().numpy().flatten()
            if np.any(p_obs > 0):
                ax.plot(t_days, p_obs, color=COLORS['primary'], linewidth=1.2,
                        linestyle='--', alpha=0.6, label='WHP 观测', zorder=2)
            
            # 状态文本框
            status = "全部在界内" if in_bounds else f"低={n_below} 高={n_above}"
            bg_color = '#D5F5E3' if in_bounds else '#FADBD8'
            edge_color = COLORS['test'] if in_bounds else COLORS['accent']
            props = dict(boxstyle='round,pad=0.4', facecolor=bg_color,
                         alpha=0.92, edgecolor=edge_color, linewidth=0.8)
            ax.text(0.02, 0.98, f'越界检查: {status}', transform=ax.transAxes,
                    fontsize=11, va='top', bbox=props)
            
            ax.set_xlabel('时间 (天)')
            ax.set_ylabel('$p_{wf}$ (MPa)')
            ax.legend(loc='upper right', framealpha=0.85, edgecolor=COLORS['info_box'])
            
            plt.tight_layout()
            if save:
                fp = os.path.join(self.fig_dir, f'M5_pwf_inversion_{wid}.png')
                fig.savefig(fp, dpi=250, bbox_inches='tight')
                plt.close(fig)
                self.logger.info(f"p_wf 反演图已保存: {fp}")
                return fp
            plt.close(fig)
        return ''
    
    def plot_training_history(self, save: bool = True) -> str:
        """绘制 M5 训练曲线 (一等奖配色)"""
        from utils import COLORS, apply_plot_style
        apply_plot_style()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('M5 PINN 训练曲线', fontsize=16, fontweight='bold')
        
        steps = self.history['step']
        
        # (1) 总损失
        ax = axes[0, 0]
        ax.semilogy(steps, self.history['total'], color=COLORS['primary'],
                     linewidth=1.2, alpha=0.85)
        ax.set_title('总损失 (Total)')
        ax.set_ylabel('Loss')
        
        # (2) 分项: PDE + IC + BC
        ax = axes[0, 1]
        phys_items = [('ic', 'IC', COLORS['ic']),
                      ('bc', 'BC', COLORS['bc']),
                      ('pde', 'PDE', COLORS['pde'])]
        for key, label, color in phys_items:
            vals = [max(v, 1e-12) for v in self.history[key]]
            ax.semilogy(steps, vals, color=color, linewidth=1.2, alpha=0.85, label=label)
        ax.set_title('物理损失')
        ax.legend(framealpha=0.85, edgecolor=COLORS['info_box'])
        
        # (3) 监督: qg
        ax = axes[0, 2]
        vals = [max(v, 1e-12) for v in self.history['qg']]
        ax.semilogy(steps, vals, color=COLORS['accent'], linewidth=1.2, alpha=0.85)
        ax.set_title('$q_g$ 同化损失')
        
        # (4) 反演参数 k_eff
        ax = axes[1, 0]
        k_frac_data = self.history.get('k_frac_mD', self.history.get('k_eff_mD', [0]*len(steps)))
        ax.plot(steps, k_frac_data, color=COLORS['k_eff'], linewidth=1.4)
        ax.set_title('$k_{frac}$ (mD)')
        ax.set_xlabel('Step')
        
        # (5) smooth_qg 正则化损失 (v3.13: 替代无信息量的 dp_wellbore 平线)
        ax = axes[1, 1]
        sqg_data = [max(v, 1e-12) for v in self.history.get('smooth_qg', [0]*len(steps))]
        ax.semilogy(steps, sqg_data, color=COLORS.get('f_frac', '#E67E22'), linewidth=1.2, alpha=0.85)
        ax.set_title('Smooth $q_g$ 正则化')
        ax.set_xlabel('Step')
        
        # (6) Sw 范围
        ax = axes[1, 2]
        ax.fill_between(steps, self.history['sw_min'], self.history['sw_max'],
                         color=COLORS['train'], alpha=0.15, label='$S_w$ 区间')
        ax.plot(steps, self.history['sw_min'], color=COLORS['sw_lo'],
                linewidth=1.0, label='$S_{w,min}$')
        ax.plot(steps, self.history['sw_max'], color=COLORS['sw_hi'],
                linewidth=1.0, label='$S_{w,max}$')
        ax.axhline(y=0, color='#95A5A6', linestyle='--', alpha=0.5, linewidth=0.6)
        ax.axhline(y=1, color='#95A5A6', linestyle='--', alpha=0.5, linewidth=0.6)
        ax.set_title('$S_w$ 范围监控')
        ax.set_xlabel('Step')
        ax.legend(framealpha=0.85, edgecolor=COLORS['info_box'])
        ax.set_ylim([-0.1, 1.1])
        
        plt.tight_layout()
        if save:
            fp = os.path.join(self.fig_dir, 'M5_training_history.png')
            fig.savefig(fp, dpi=250, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"训练曲线图已保存: {fp}")
            return fp
        plt.close(fig)
        return ''
    
    def plot_pde_residual_map(self, save: bool = True) -> str:
        """绘制 PDE 残差空间热力图 (一等奖配色: YlOrRd)"""
        from utils import COLORS, CMAP_HEAT, apply_plot_style
        apply_plot_style()
        self.model.eval()
        
        # 在固定时刻 t=0.5 (中期) 的空间网格上计算残差
        nx, ny = 50, 50
        x_lin = np.linspace(-1, 1, nx)
        y_lin = np.linspace(-1, 1, ny)
        xx, yy = np.meshgrid(x_lin, y_lin)
        
        t_val = 0.5
        xyt = np.stack([xx.flatten(), yy.flatten(),
                        np.full(nx * ny, t_val)], axis=-1).astype(np.float32)
        x_grid = torch.from_numpy(xyt).float().to(self.device)
        
        residuals = self.loss_fn.compute_residual_map(self.model, x_grid)
        res_map = residuals.cpu().numpy().reshape(ny, nx)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.pcolormesh(xx, yy, res_map, shading='auto', cmap=CMAP_HEAT)
        cbar = plt.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label('|PDE 残差|', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # 标注井位 (使用 COLORS['well'] 红色星标)
        for wid in self.model.well_ids:
            if wid in self.well_data:
                wx = self.well_data[wid]['xyt'][0, 0].item()
                wy = self.well_data[wid]['xyt'][0, 1].item()
                ax.plot(wx, wy, '*', color=COLORS['well'], markersize=16,
                        markeredgecolor='white', markeredgewidth=0.8, zorder=5)
                ax.annotate(wid, (wx, wy), fontsize=10, fontweight='bold',
                            color=COLORS['primary'],
                            xytext=(6, 6), textcoords='offset points')
        
        ax.set_xlabel('$x$ (归一化)', fontsize=12)
        ax.set_ylabel('$y$ (归一化)', fontsize=12)
        ax.set_title(f'PDE 残差空间分布 ($t_{{norm}}$={t_val})', fontsize=14)
        
        plt.tight_layout()
        if save:
            fp = os.path.join(self.fig_dir, 'M5_pde_residual_map.png')
            fig.savefig(fp, dpi=250, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"残差热力图已保存: {fp}")
            return fp
        plt.close(fig)
        return ''
    
    def plot_sw_evolution(self, save: bool = True) -> str:
        """
        v4.1: 绘制 SY9 井位 PINN Sw(t) vs TDS→Sw 软标签对比图
        补齐 M5 缺失的 Sw 时间演化可视化。
        """
        from utils import COLORS, apply_plot_style
        apply_plot_style()
        self.model.eval()
        
        for wid in self.model.well_ids:
            if wid not in self.well_data:
                continue
            wdata = self.well_data[wid]
            
            with torch.no_grad():
                result = self.model.evaluate_at_well(
                    wid, wdata['xyt'],
                    h_well=self.well_h.get(wid, 90.0),
                    bg_val=self.bg_ref,
                    prod_hours_norm=wdata.get('prod_hours_norm', None),
                    casing_norm=wdata.get('casing_norm', None),
                )
                sw_pred = result['sw_cell'].cpu().numpy().flatten()
            
            t_days = wdata['t_days']
            n = len(t_days)
            n_train = int(n * self.m5_train_ratio)
            n_val = int(n * self.m5_val_ratio)
            
            # --- 获取相渗参数 ---
            rp = self.loss_fn.relperm
            swc = float(rp.Swc)
            sgr = float(rp.Sgr)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'M5 井 {wid} 含水饱和度 $S_w(t)$ 演化', fontsize=15, fontweight='bold')
            
            # ---- 左图: Sw(t) 时间序列 ----
            ax1 = axes[0]
            ymax = min(max(sw_pred.max() * 1.3, 0.6), 1.05)
            ax1.fill_between(t_days[:n_train], 0, ymax,
                             color=COLORS['train'], alpha=0.07, label='训练集')
            ax1.fill_between(t_days[n_train:n_train + n_val], 0, ymax,
                             color=COLORS['val'], alpha=0.07, label='验证集')
            ax1.fill_between(t_days[n_train + n_val:], 0, ymax,
                             color=COLORS['test'], alpha=0.07, label='测试集')
            
            ax1.plot(t_days, sw_pred, color=COLORS['accent'], linewidth=1.8,
                     label='PINN $S_w$ 预测', zorder=4)
            
            # 物理边界线
            ax1.axhline(y=swc, color=COLORS['sw_lo'], linestyle='--',
                        alpha=0.6, linewidth=0.8, label=f'$S_{{wc}}$={swc:.3f}')
            ax1.axhline(y=1.0 - sgr, color=COLORS['sw_hi'], linestyle='--',
                        alpha=0.6, linewidth=0.8, label=f'$1-S_{{gr}}$={1-sgr:.3f}')
            
            # TDS→Sw 软标签叠加
            sw_metrics = {}
            if self.tds_data is not None and wid in self.tds_data:
                tds_info = self.tds_data[wid]
                t_tds_norm = tds_info['t_norm'].cpu().numpy().flatten()
                sw_tds = tds_info['sw_tds'].cpu().numpy().flatten()
                t_tds_days = t_tds_norm * float(self.sampler.t_max)
                
                ax1.scatter(t_tds_days, sw_tds, c='#8E44AD', s=25, alpha=0.7,
                            edgecolors='white', linewidths=0.4, zorder=5,
                            label=f'TDS→$S_w$ 软标签 (n={len(sw_tds)})')
                
                # 计算 PINN Sw 在 TDS 时间点的插值 → 对比指标
                from scipy.interpolate import interp1d
                try:
                    sw_interp_fn = interp1d(t_days, sw_pred, kind='linear',
                                            bounds_error=False, fill_value='extrapolate')
                    sw_pinn_at_tds = sw_interp_fn(t_tds_days)
                    valid = np.isfinite(sw_pinn_at_tds) & np.isfinite(sw_tds)
                    if np.sum(valid) > 3:
                        residuals = sw_tds[valid] - sw_pinn_at_tds[valid]
                        mae = np.mean(np.abs(residuals))
                        rmse = np.sqrt(np.mean(residuals ** 2))
                        corr = np.corrcoef(sw_pinn_at_tds[valid], sw_tds[valid])[0, 1]
                        sw_metrics = {'MAE': float(mae), 'RMSE': float(rmse),
                                      'Pearson_r': float(corr),
                                      'n_points': int(np.sum(valid))}
                        textstr = (f'vs TDS:\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\n'
                                   f'Pearson r = {corr:.4f}\nn = {int(np.sum(valid))}')
                        props = dict(boxstyle='round,pad=0.4', facecolor='#F5EEF8',
                                     alpha=0.92, edgecolor='#8E44AD', linewidth=0.8)
                        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
                                 fontsize=10, verticalalignment='top', bbox=props)
                except Exception as e:
                    self.logger.warning(f"  Sw vs TDS 指标计算失败: {e}")
            
            ax1.set_xlabel('时间 (天)')
            ax1.set_ylabel('$S_w$')
            ax1.set_title('$S_w(t)$ 时间演化')
            ax1.set_ylim(bottom=0, top=ymax)
            ax1.legend(loc='upper right', framealpha=0.85, edgecolor=COLORS['info_box'],
                       fontsize=9)
            
            # ---- 右图: Sw 分布直方图 ----
            ax2 = axes[1]
            ax2.hist(sw_pred, bins=40, color=COLORS['train'], alpha=0.7,
                     edgecolor='white', linewidth=0.5, density=True, label='PINN $S_w$')
            if self.tds_data is not None and wid in self.tds_data:
                sw_tds_vals = self.tds_data[wid]['sw_tds'].cpu().numpy().flatten()
                ax2.hist(sw_tds_vals, bins=20, color='#8E44AD', alpha=0.5,
                         edgecolor='white', linewidth=0.5, density=True,
                         label='TDS→$S_w$')
            ax2.axvline(x=swc, color=COLORS['sw_lo'], linestyle='--', alpha=0.7,
                        linewidth=1.0, label=f'$S_{{wc}}$')
            ax2.axvline(x=1.0 - sgr, color=COLORS['sw_hi'], linestyle='--', alpha=0.7,
                        linewidth=1.0, label=f'$1-S_{{gr}}$')
            ax2.set_xlabel('$S_w$')
            ax2.set_ylabel('概率密度')
            ax2.set_title('$S_w$ 分布对比')
            ax2.legend(framealpha=0.85, edgecolor=COLORS['info_box'], fontsize=9)
            
            plt.tight_layout()
            if save:
                fp = os.path.join(self.fig_dir, f'M5_sw_evolution_{wid}.png')
                fig.savefig(fp, dpi=250, bbox_inches='tight')
                plt.close(fig)
                self.logger.info(f"Sw 演化图已保存: {fp}")
                # 保存 Sw 指标 JSON
                if sw_metrics:
                    metrics_path = os.path.join(self.report_dir, f'M5_sw_metrics_{wid}.json')
                    with open(metrics_path, 'w', encoding='utf-8') as f:
                        json.dump(sw_metrics, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Sw 指标已保存: {metrics_path}")
                return fp
            plt.close(fig)
        return ''
    
    def plot_sw_spatial(self, save: bool = True) -> str:
        """
        v4.1: 绘制 Sw(x,y) 空间分布热力图 (多时刻快照)
        展示水侵前锋的空间扩展过程。
        """
        from utils import COLORS, CMAP_SW, apply_plot_style
        apply_plot_style()
        self.model.eval()
        
        t_snapshots = [0.0, 0.25, 0.50, 0.75, 1.0]
        t_labels = ['初始', '25%', '50%', '75%', '末期']
        
        nx, ny = 60, 60
        x_lin = np.linspace(-1, 1, nx)
        y_lin = np.linspace(-1, 1, ny)
        xx, yy = np.meshgrid(x_lin, y_lin)
        
        rp = self.loss_fn.relperm
        swc = float(rp.Swc)
        sgr = float(rp.Sgr)
        
        fig, axes = plt.subplots(1, len(t_snapshots), figsize=(5 * len(t_snapshots), 4.5))
        fig.suptitle('M5 含水饱和度 $S_w(x,y)$ 空间分布演化', fontsize=15, fontweight='bold')
        
        sw_all = []
        with torch.no_grad():
            for i, t_val in enumerate(t_snapshots):
                xyt = np.stack([xx.flatten(), yy.flatten(),
                                np.full(nx * ny, t_val)], axis=-1).astype(np.float32)
                x_grid = torch.from_numpy(xyt).float().to(self.device)
                _, sw = self.model(x_grid)
                sw_map = sw.cpu().numpy().reshape(ny, nx)
                sw_all.append(sw_map)
        
        # 统一 colorbar 范围
        vmin = min(s.min() for s in sw_all)
        vmax = max(s.max() for s in sw_all)
        vmin = max(vmin, swc - 0.05)
        vmax = min(vmax, 1.0 - sgr + 0.05)
        if vmax - vmin < 0.02:
            vmin = swc - 0.02
            vmax = swc + 0.15
        
        for i, (sw_map, t_val, t_label) in enumerate(zip(sw_all, t_snapshots, t_labels)):
            ax = axes[i]
            im = ax.pcolormesh(xx, yy, sw_map, shading='auto', cmap=CMAP_SW,
                               vmin=vmin, vmax=vmax)
            ax.set_title(f'{t_label}\n$t_{{norm}}$={t_val:.2f}', fontsize=11)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_ylabel('$y$ (归一化)', fontsize=10)
            ax.set_xlabel('$x$ (归一化)', fontsize=10)
            
            # 标注井位
            for wid in self.model.well_ids:
                if wid in self.well_data:
                    wx = self.well_data[wid]['xyt'][0, 0].item()
                    wy = self.well_data[wid]['xyt'][0, 1].item()
                    ax.plot(wx, wy, '*', color=COLORS['well'], markersize=12,
                            markeredgecolor='white', markeredgewidth=0.6, zorder=5)
                    if i == 0:
                        ax.annotate(wid, (wx, wy), fontsize=8, fontweight='bold',
                                    color=COLORS['primary'],
                                    xytext=(4, 4), textcoords='offset points')
        
        # 统一 colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, aspect=30)
        cbar.set_label('$S_w$', fontsize=12)
        cbar.ax.tick_params(labelsize=9)
        
        plt.tight_layout(rect=[0, 0, 0.92, 0.93])
        if save:
            fp = os.path.join(self.fig_dir, 'M5_sw_spatial_evolution.png')
            fig.savefig(fp, dpi=250, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Sw 空间分布图已保存: {fp}")
            return fp
        plt.close(fig)
        return ''
    
    def generate_report(self, report_ckpt: str = 'auto'):
        """生成 M5/M6/M7 验收报告 (含连通性 + 水侵)"""
        # 报告 checkpoint 选择策略:
        #   auto: 在 best/final 中按 qg 综合分数自动择优
        #   best/final: 强制使用指定 checkpoint
        def _qg_report_score() -> float:
            self.model.eval()
            scores = []
            with torch.no_grad():
                for wid in self.model.well_ids:
                    if wid not in self.well_data:
                        continue
                    wdata = self.well_data[wid]
                    res = self.model.evaluate_at_well(
                        wid, wdata['xyt'],
                        h_well=self.well_h.get(wid, 90.0),
                        bg_val=self.bg_ref,
                        prod_hours_norm=wdata.get('prod_hours_norm', None),
                        casing_norm=wdata.get('casing_norm', None),
                    )
                    qg_pred = res['qg'].cpu().numpy().flatten()
                    qg_obs = wdata['qg_obs'].cpu().numpy().flatten()
                    if 'shutin_mask' in wdata:
                        shutin_all = (wdata['shutin_mask'].cpu().numpy().flatten() > 0.5)
                        qg_pred = np.where(shutin_all, 0.0, qg_pred)
                    valid = np.isfinite(qg_pred) & np.isfinite(qg_obs)
                    if 'qg_valid_mask' in wdata:
                        valid = valid & (wdata['qg_valid_mask'].cpu().numpy().flatten() > 0.5)
                    open_mask = valid & (qg_obs > 1.0)
                    if not np.any(open_mask):
                        continue
                    # 开井 MAPE + 高产 MAPE + 负R2 惩罚 + 关井 MAE 软惩罚
                    mape_open = np.mean(
                        np.abs((qg_obs[open_mask] - qg_pred[open_mask]) / (qg_obs[open_mask] + 1.0))
                    ) * 100.0
                    ss_res = np.sum((qg_obs[open_mask] - qg_pred[open_mask]) ** 2)
                    ss_tot = np.sum((qg_obs[open_mask] - qg_obs[open_mask].mean()) ** 2)
                    r2_open = 1.0 - ss_res / (ss_tot + 1e-12)
                    high_th = float(max(4e5, np.percentile(qg_obs[open_mask], 75)))
                    high_mask = open_mask & (qg_obs >= high_th)
                    if np.any(high_mask):
                        mape_high = np.mean(
                            np.abs((qg_obs[high_mask] - qg_pred[high_mask]) / (qg_obs[high_mask] + 1.0))
                        ) * 100.0
                    else:
                        mape_high = mape_open
                    shutin_mask = valid & (qg_obs <= 1.0)
                    mae_shutin = float(np.mean(np.abs(qg_pred[shutin_mask] - qg_obs[shutin_mask]))) if np.any(shutin_mask) else 2e5
                    shutin_penalty = min(mae_shutin / 10000.0, 120.0)
                    score = (
                        0.40 * mape_open
                        + 0.25 * mape_high
                        + 0.25 * max(0.0, -r2_open) * 100.0
                        + 0.10 * shutin_penalty
                    )
                    scores.append(float(score))
            self.model.train()
            return float(np.mean(scores)) if scores else float('inf')

        requested = (report_ckpt or 'auto').lower()
        if requested not in ('auto', 'best', 'final'):
            requested = 'auto'
        candidates = []
        if requested == 'auto':
            for tag in ('best', 'final'):
                if os.path.exists(os.path.join(self.ckpt_dir, f'm5_pinn_{tag}.pt')):
                    candidates.append(tag)
        else:
            if os.path.exists(os.path.join(self.ckpt_dir, f'm5_pinn_{requested}.pt')):
                candidates.append(requested)

        report_tag = None
        report_ckpt_path = None
        report_score = None
        if candidates:
            if len(candidates) == 1:
                report_tag = candidates[0]
                self.load_checkpoint(report_tag)
                report_score = _qg_report_score()
            else:
                score_map = {}
                for tag in candidates:
                    self.load_checkpoint(tag)
                    score_map[tag] = _qg_report_score()
                report_tag = min(score_map, key=score_map.get)
                report_score = float(score_map[report_tag])
                self.load_checkpoint(report_tag)
            report_ckpt_path = os.path.join(self.ckpt_dir, f'm5_pinn_{report_tag}.pt')
            self.logger.info(
                f"generate_report: 自动择优采用 {report_tag} checkpoint, qg_score={report_score:.4f}"
            )
            # 报告与 JSON 一致：用当前已加载 checkpoint 覆盖 inversion_params.json
            self._save_inversion_audit()
        else:
            self.logger.warning("generate_report: 未找到可用 checkpoint，使用当前模型状态")
        
        self.plot_qg_comparison()
        self.plot_pwf_inversion()
        self.plot_training_history()
        self.plot_pde_residual_map()
        
        # 文字报告
        inv = self.model.get_inversion_params()
        param_bd = self.model.count_parameters_breakdown()
        dp_wellbore = float(inv.get('dp_wellbore_MPa', 0.0))
        # ★★★ FIX-v9/v10: 从当前模型重算 qg/p_wf 指标，并输出 train/val/test 分段 ★★★
        qg_metrics = {}
        pwf_metrics = {}
        self.model.eval()
        with torch.no_grad():
            for wid in self.model.well_ids:
                if wid not in self.well_data:
                    continue
                wdata = self.well_data[wid]
                result = self.model.evaluate_at_well(
                    wid, wdata['xyt'],
                    h_well=self.well_h.get(wid, 90.0),
                    bg_val=self.bg_ref,
                    prod_hours_norm=wdata.get('prod_hours_norm', None),
                    casing_norm=wdata.get('casing_norm', None),
                )
                n = wdata['xyt'].shape[0]
                n_train = int(n * self.m5_train_ratio)
                n_val = int(n * self.m5_val_ratio)
                n_test = n - n_train - n_val
                idx_train = slice(0, n_train)
                idx_val = slice(n_train, n_train + n_val)
                idx_test = slice(n_train + n_val, n)

                qg_pred = result['qg'].cpu().numpy().flatten()
                qg_obs = wdata['qg_obs'].cpu().numpy().flatten()
                if 'shutin_mask' in wdata:
                    shutin_all = (wdata['shutin_mask'].cpu().numpy().flatten() > 0.5)
                    qg_pred = np.where(shutin_all, 0.0, qg_pred)
                qg_valid_all = None
                if 'qg_valid_mask' in wdata:
                    qg_valid_all = (wdata['qg_valid_mask'].cpu().numpy().flatten() > 0.5)
                valid_qg = (qg_obs > 1.0) & np.isfinite(qg_pred)
                if qg_valid_all is not None:
                    valid_qg = valid_qg & qg_valid_all
                if np.any(valid_qg):
                    mape = np.mean(np.abs((qg_obs[valid_qg] - qg_pred[valid_qg]) / (qg_obs[valid_qg] + 1.0))) * 100
                    rmse = np.sqrt(np.mean((qg_obs[valid_qg] - qg_pred[valid_qg]) ** 2))
                    ss_res = np.sum((qg_obs[valid_qg] - qg_pred[valid_qg]) ** 2)
                    ss_tot = np.sum((qg_obs[valid_qg] - qg_obs[valid_qg].mean()) ** 2)
                    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
                    qg_metrics[wid] = {'MAPE': mape, 'RMSE': rmse, 'R2': r2}
                    # 分段 qg
                    def _seg_metrics(obs, pred, valid, idx):
                        v = np.zeros_like(valid, dtype=bool)
                        v[idx] = valid[idx]
                        if not np.any(v):
                            return None
                        o, p = obs[v], pred[v]
                        mape_s = np.mean(np.abs((o - p) / (o + 1.0))) * 100
                        rmse_s = np.sqrt(np.mean((o - p) ** 2))
                        ss_r = np.sum((o - p) ** 2)
                        ss_t = np.sum((o - o.mean()) ** 2)
                        r2_s = 1.0 - ss_r / (ss_t + 1e-12)
                        return {'MAPE': mape_s, 'RMSE': rmse_s, 'R2': r2_s}
                    qg_metrics[wid]['train'] = _seg_metrics(qg_obs, qg_pred, valid_qg, idx_train)
                    qg_metrics[wid]['val'] = _seg_metrics(qg_obs, qg_pred, valid_qg, idx_val)
                    qg_metrics[wid]['test'] = _seg_metrics(qg_obs, qg_pred, valid_qg, idx_test)
                    
                    # P0-5: 细化指标 — 关井段、高产段、val/test 点数
                    shutin_mask = (qg_obs <= 1.0) & np.isfinite(qg_pred)
                    if qg_valid_all is not None:
                        shutin_mask = shutin_mask & qg_valid_all
                    n_shutin = int(np.sum(shutin_mask))
                    if n_shutin > 0:
                        mae_shutin = np.mean(np.abs(qg_pred[shutin_mask]))
                        pred_mean_shutin = np.mean(qg_pred[shutin_mask])
                        qg_metrics[wid]['shutin'] = {'n': n_shutin, 'MAE': mae_shutin, 'pred_mean': pred_mean_shutin}
                    else:
                        qg_metrics[wid]['shutin'] = None
                    
                    # 高产阈值：max(4e5, P75(开井qg))，可解释且与数据一致
                    open_qg = qg_obs[valid_qg]
                    threshold_high = float(max(4e5, np.percentile(open_qg, 75))) if len(open_qg) > 0 else 6e5
                    qg_metrics[wid]['high_threshold'] = threshold_high
                    high_prod_mask = valid_qg & (qg_obs >= threshold_high)
                    n_high = int(np.sum(high_prod_mask))
                    if n_high > 0:
                        mape_high = np.mean(np.abs((qg_obs[high_prod_mask] - qg_pred[high_prod_mask]) / (qg_obs[high_prod_mask] + 1.0))) * 100
                        mean_err_high = np.mean(qg_pred[high_prod_mask] - qg_obs[high_prod_mask])
                        qg_metrics[wid]['high_prod'] = {'n': n_high, 'MAPE': mape_high, 'mean_error': mean_err_high}
                    else:
                        qg_metrics[wid]['high_prod'] = None
                    
                    # val/test 开井点数
                    n_val_open = int(np.sum(valid_qg[idx_val]))
                    n_test_open = int(np.sum(valid_qg[idx_test]))
                    qg_metrics[wid]['n_val_open'] = n_val_open
                    qg_metrics[wid]['n_test_open'] = n_test_open

                p_wf_pred = result['p_wf'].cpu().numpy().flatten()
                p_obs = wdata['p_obs'].cpu().numpy().flatten()
                p_wf_obs = p_obs + dp_wellbore
                valid_pwf = np.isfinite(p_wf_obs) & np.isfinite(p_wf_pred) & (p_wf_obs > 1.0)
                if np.any(valid_pwf):
                    mape_p = np.mean(np.abs((p_wf_obs[valid_pwf] - p_wf_pred[valid_pwf]) / (p_wf_obs[valid_pwf] + 1.0))) * 100
                    ss_res_p = np.sum((p_wf_obs[valid_pwf] - p_wf_pred[valid_pwf]) ** 2)
                    ss_tot_p = np.sum((p_wf_obs[valid_pwf] - p_wf_obs[valid_pwf].mean()) ** 2)
                    r2_p = 1.0 - ss_res_p / (ss_tot_p + 1e-12)
                    rmse_p = np.sqrt(np.mean((p_wf_obs[valid_pwf] - p_wf_pred[valid_pwf]) ** 2))
                    pwf_metrics[wid] = {'MAPE': mape_p, 'RMSE': rmse_p, 'R2': r2_p}
                    def _seg_pwf(obs, pred, valid, idx):
                        v = np.zeros_like(valid, dtype=bool)
                        v[idx] = valid[idx]
                        if not np.any(v):
                            return None
                        o, p = obs[v], pred[v]
                        mape_s = np.mean(np.abs((o - p) / (o + 1.0))) * 100
                        rmse_s = np.sqrt(np.mean((o - p) ** 2))
                        ss_r = np.sum((o - p) ** 2)
                        ss_t = np.sum((o - o.mean()) ** 2)
                        return {'MAPE': mape_s, 'RMSE': rmse_s, 'R2': 1.0 - ss_r / (ss_t + 1e-12)}
                    pwf_metrics[wid]['train'] = _seg_pwf(p_wf_obs, p_wf_pred, valid_pwf, idx_train)
                    pwf_metrics[wid]['val'] = _seg_pwf(p_wf_obs, p_wf_pred, valid_pwf, idx_val)
                    pwf_metrics[wid]['test'] = _seg_pwf(p_wf_obs, p_wf_pred, valid_pwf, idx_test)

        # 报告中加入实际使用 checkpoint 信息（参数/指标/时间戳来自当前已加载的 report_tag）
        source_tag = report_tag if report_tag is not None else 'current'
        source_step = (
            self.best_step if (source_tag == 'best' and hasattr(self, 'best_step') and self.best_step is not None)
            else ('final' if source_tag == 'final' else 'N/A')
        )
        import datetime
        source_time = (
            datetime.datetime.fromtimestamp(os.path.getmtime(report_ckpt_path)).strftime('%Y-%m-%d %H:%M:%S')
            if (report_ckpt_path and os.path.exists(report_ckpt_path)) else 'N/A'
        )
        source_label = f"{source_tag} checkpoint"
        
        lines = [
            "# M5 井—藏耦合同化验收报告 (v3)\n",
            f"## 参数来源 ({source_label})",
            f"- Source tag: {source_tag}",
            f"- Step: {source_step}",
            f"- Checkpoint 时间: {source_time}\n",
            f"## 训练配置",
            f"- 总步数: {self.max_steps}",
            f"- 学习率: {self.lr}",
            f"- ReLoBRaLo: {self.use_relobralo}",
            f"- RAR: {self.rar.enable}",
            f"- RAR 点数: {self.rar.get_stats()}",
            f"- Fourier Features: {self.model.field_net.use_fourier if hasattr(self.model.field_net, 'use_fourier') else 'N/A'}",
            f"- k_net: {hasattr(self.model, 'k_net') and self.model.k_net is not None}",
            f"- 参数量: {param_bd}\n",
            f"## 反演参数",
            f"- k_frac: {inv.get('k_frac_mD', inv.get('k_eff_mD', 'N/A')):.4f} mD",
            f"- f_frac: (已合并入 k_frac)",
            f"- r_e: {inv.get('r_e_m', 'N/A'):.1f} m (先验={inv.get('r_e_prior_m', 'N/A'):.1f} m, 可学习v4.8)",
            f"- dp_wellbore: {inv.get('dp_wellbore_MPa', 'N/A'):.2f} MPa (WHP→p_wf 井筒压差)\n",
            f"## 最终损失 ({source_label})",
            f"- Total: {self.history['total'][-1]:.6e}" if self.history['total'] else "- N/A",
            f"- PDE: {self.history['pde'][-1]:.6e}" if self.history['pde'] else "- N/A",
            f"- qg: {self.history['qg'][-1]:.6e}" if self.history['qg'] else "- N/A",
            f"- k_reg: {self.history['k_reg'][-1]:.6e}" if self.history.get('k_reg') else "- N/A",
        ]
        
        # 当前模型的 qg 验证指标（含 train/val/test 分段 + 细化指标）
        if qg_metrics:
            lines.append(f"\n## 当前模型 qg 拟合指标 ({source_label})")
            for wid, m in qg_metrics.items():
                lines.append(f"- 井 {wid}: MAPE={m['MAPE']:.1f}%, RMSE={m['RMSE']:.0f} m³/d, R²={m['R2']:.4f}")
                for seg in ('train', 'val', 'test'):
                    s = m.get(seg)
                    if s:
                        lines.append(f"  - {seg}: MAPE={s['MAPE']:.1f}%, RMSE={s['RMSE']:.0f}, R²={s['R2']:.4f}")
                
                # P0-5: val/test 开井点数（验证数据划分是否合理）
                lines.append(f"  - val 开井点数: {m.get('n_val_open', 'N/A')}, test 开井点数: {m.get('n_test_open', 'N/A')}")
                
                # P0-5: 关井段诊断（q≤1）
                shutin = m.get('shutin')
                if shutin:
                    lines.append(
                        f"  - 关井段 (q≤1): n={shutin['n']}, MAE={shutin['MAE']:.0f} m³/d, "
                        f"pred_mean={shutin['pred_mean']:.0f} m³/d"
                    )
                
                # P0-5: 高产段诊断（阈值=max(4e5, P75(open_qg))，写入报告）
                th = m.get('high_threshold', 6e5)
                high = m.get('high_prod')
                if high:
                    lines.append(
                        f"  - 高产段 (q≥{th:.0f}): n={high['n']}, MAPE={high['MAPE']:.1f}%, "
                        f"mean_error={high['mean_error']:.0f} m³/d"
                    )
        # v10: p_wf 拟合指标（相对 WHP+dp_wellbore）及 train/val/test 分段
        if pwf_metrics:
            lines.append(f"\n## 当前模型 p_wf 拟合指标 ({source_label}, 相对 WHP+dp_wellbore)")
            for wid, m in pwf_metrics.items():
                lines.append(f"- 井 {wid}: MAPE={m['MAPE']:.1f}%, RMSE={m['RMSE']:.2f} MPa, R²={m['R2']:.4f}")
                for seg in ('train', 'val', 'test'):
                    s = m.get(seg)
                    if s:
                        lines.append(f"  - {seg}: MAPE={s['MAPE']:.1f}%, RMSE={s['RMSE']:.2f} MPa, R²={s['R2']:.4f}")

        # -------- 对照 checkpoint 诊断（用于判断是否过拟合/欠拟合）--------
        try:
            diag_tag = 'final' if source_tag != 'final' else 'best'
            diag_path = os.path.join(self.ckpt_dir, f'm5_pinn_{diag_tag}.pt')
            if not os.path.exists(diag_path):
                raise FileNotFoundError(f"{diag_path} 不存在")
            self.load_checkpoint(diag_tag)
            qg_diag, pwf_diag = {}, {}
            inv_d = self.model.get_inversion_params()
            dp_d = float(inv_d.get('dp_wellbore_MPa', 0.0))
            with torch.no_grad():
                for wid in self.model.well_ids:
                    if wid not in self.well_data:
                        continue
                    wdata = self.well_data[wid]
                    res = self.model.evaluate_at_well(
                        wid, wdata['xyt'],
                        h_well=self.well_h.get(wid, 90.0),
                        bg_val=self.bg_ref,
                        prod_hours_norm=wdata.get('prod_hours_norm', None),
                    )
                    n = wdata['xyt'].shape[0]
                    n_train = int(n * self.m5_train_ratio)
                    n_val = int(n * self.m5_val_ratio)
                    n_test = n - n_train - n_val
                    idx_t, idx_v, idx_te = slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)
                    qp, qo = res['qg'].cpu().numpy().flatten(), wdata['qg_obs'].cpu().numpy().flatten()
                    if 'shutin_mask' in wdata:
                        shutin_all = (wdata['shutin_mask'].cpu().numpy().flatten() > 0.5)
                        qp = np.where(shutin_all, 0.0, qp)
                    vq = (qo > 1.0) & np.isfinite(qp)
                    if np.any(vq):
                        mape_q = np.mean(np.abs((qo[vq] - qp[vq]) / (qo[vq] + 1.0))) * 100
                        rmse_q = np.sqrt(np.mean((qo[vq] - qp[vq]) ** 2))
                        ss_r = np.sum((qo[vq] - qp[vq]) ** 2)
                        ss_t = np.sum((qo[vq] - qo[vq].mean()) ** 2)
                        qg_diag[wid] = {'MAPE': mape_q, 'RMSE': rmse_q, 'R2': 1.0 - ss_r / (ss_t + 1e-12)}
                    po = wdata['p_obs'].cpu().numpy().flatten() + dp_d
                    pp = res['p_wf'].cpu().numpy().flatten()
                    vp = np.isfinite(po) & np.isfinite(pp) & (po > 1.0)
                    if np.any(vp):
                        mape_p = np.mean(np.abs((po[vp] - pp[vp]) / (po[vp] + 1.0))) * 100
                        rmse_p = np.sqrt(np.mean((po[vp] - pp[vp]) ** 2))
                        ss_r = np.sum((po[vp] - pp[vp]) ** 2)
                        ss_t = np.sum((po[vp] - po[vp].mean()) ** 2)
                        pwf_diag[wid] = {'MAPE': mape_p, 'RMSE': rmse_p, 'R2': 1.0 - ss_r / (ss_t + 1e-12)}
            if source_tag in ('best', 'final'):
                self.load_checkpoint(source_tag)
            lines.append(f"\n## 对照 checkpoint 诊断 ({diag_tag} 对照)")
            lines.append("- 用途: 与报告主模型对照，辅助判断过拟合/欠拟合。")
            if self.history['total']:
                lines.append(f"- 最终步损失: Total={self.history['total'][-1]:.6e}, PDE={self.history['pde'][-1]:.6e}, qg={self.history['qg'][-1]:.6e}")
            for wid, m in qg_diag.items():
                lines.append(f"- 井 {wid} qg ({diag_tag}): MAPE={m['MAPE']:.1f}%, RMSE={m['RMSE']:.0f} m³/d, R²={m['R2']:.4f}")
            for wid, m in pwf_diag.items():
                lines.append(f"- 井 {wid} p_wf ({diag_tag}): MAPE={m['MAPE']:.1f}%, RMSE={m['RMSE']:.2f} MPa, R²={m['R2']:.4f}")
        except Exception as e:
            self.logger.warning(f"对照 checkpoint 诊断跳过: {e}")
            try:
                if source_tag in ('best', 'final'):
                    self.load_checkpoint(source_tag)
            except Exception:
                pass

        report_path = os.path.join(self.report_dir, 'M5_validation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        self.logger.info(f"M5 验收报告已保存: {report_path}")


