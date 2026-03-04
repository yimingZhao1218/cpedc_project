"""
M4 PINN 训练器
严格按报告 5.7 分阶段训练策略，防翻车

阶段由 YAML m4_config.training_stages 驱动（可与下方默认不同）:
Stage A: 以 IC+BC 为主，可带微弱 PDE/data 防锁死（见 config）
Stage B: 逐步加大 λ_PDE、λ_data、λ_sw_phys
Stage C: 全面同化（物理+数据+先验）
Stage D: 微调，物理主导
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# 修复 OpenMP 库冲突（Windows + Anaconda + PyTorch 常见问题）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise ImportError("M4 PINN 模块需要 PyTorch，请运行: pip install torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger, setup_chinese_support, ensure_dir

# 中文字体必须在 import plt 之前配置，否则会被默认覆盖
setup_chinese_support()
import matplotlib.pyplot as plt


class PINNTrainer:
    """PINN 分阶段训练器"""
    
    def __init__(self, config: dict, model: nn.Module,
                 loss_fn, sampler, device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            config: 全局配置字典
            model: PINNNet 模型
            loss_fn: PINNLoss 损失函数
            sampler: PINNSampler 采样器
            device: 计算设备
        """
        setup_chinese_support()
        self.config = config
        self.device = device
        self.logger = setup_logger('PINNTrainer')
        
        # GPU 优化 & 可复现性（deterministic 与 benchmark 互斥）
        repro_cfg = config.get('reproducibility', {})
        deterministic = repro_cfg.get('deterministic', False)
        if 'cuda' in device:
            if deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
                self.logger.info("  CUDA: deterministic 模式（cudnn.benchmark=False）")
            else:
                torch.backends.cudnn.benchmark = True
                self.logger.info("  CUDA: cudnn.benchmark = True")
        
        # 全局种子
        seed = repro_cfg.get('seed', None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if 'cuda' in device:
                torch.cuda.manual_seed_all(seed)
        
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.sampler = sampler
        
        # 训练参数
        train_cfg = config.get('train', {})
        self.lr = train_cfg.get('learning_rate', 1e-3)
        self.max_steps = train_cfg.get('max_steps', 8000)
        self.batch_size = train_cfg.get('batch_size', 2048)
        
        # Mixed Precision (AMP) — GPU 训练加速（debug_nan 时关闭 AMP 避免 backward 中 NaN）
        runtime_cfg = config.get('runtime', {})
        self.use_amp = (runtime_cfg.get('mixed_precision', False)
                        and 'cuda' in device
                        and not config.get('debug_nan', False))
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            self.logger.info("  Mixed Precision (AMP) 已启用")
        elif config.get('debug_nan', False) and 'cuda' in device:
            self.logger.info("  debug_nan 开启: AMP 已关闭，使用 fp32 避免梯度 NaN")
        
        # 优化器
        opt_cfg = train_cfg.get('optimizer', {})
        wd = opt_cfg.get('weight_decay', 1e-4)
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=wd)
        
        # 学习率调度器 (BUG4: Warmup + Cosine)
        sched_cfg = train_cfg.get('scheduler', {})
        min_lr = sched_cfg.get('min_lr', 1e-6)
        warmup_steps = sched_cfg.get('warmup_steps', 1000)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.max_steps - warmup_steps, 1), eta_min=min_lr
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        self.logger.info(f"  学习率: warmup {warmup_steps} 步, 然后 Cosine 至 min_lr={min_lr}")
        
        # 梯度裁剪
        reg_cfg = config.get('loss', {}).get('regularization', {})
        self.grad_clip = reg_cfg.get('grad_clip_norm', 5.0)
        
        # 训练历史
        self.history = {
            'step': [], 'total': [], 'ic': [], 'bc': [],
            'pde': [], 'data': [], 'sw_phys': [], 'lr': [],
            'sw_min': [], 'sw_max': [], 'stage': []
        }
        
        # Debug NaN：非有限数检测 + autograd 异常检测（默认关闭）
        self.debug_nan = config.get('debug_nan', False)
        
        # 输出目录
        output_dir = config['paths']['outputs']
        self.output_dir = output_dir
        self.fig_dir = config['paths'].get('figures', os.path.join(output_dir, 'figs'))
        self.ckpt_dir = config['paths'].get('checkpoints', os.path.join(output_dir, 'ckpt'))
        ensure_dir(self.output_dir)
        ensure_dir(self.fig_dir)
        ensure_dir(self.ckpt_dir)
        
        # 加载数据锚点
        self._load_data_anchors()
        
        # 计算物理缩放系数 α_phys → αx, αy 并注入 loss_fn
        self._compute_pde_scaling()
        
        self.logger.info(
            f"PINNTrainer 初始化: device={device}, lr={self.lr}, "
            f"max_steps={self.max_steps}, batch_size={self.batch_size}, "
            f"params={model.count_parameters()}"
        )
    
    def _load_data_anchors(self):
        """加载 SY9 数据锚点（训练仅用压力有效点，过滤 p_obs 含 NaN 的行；绘图用全序列与缺失区间）"""
        well_data = self.sampler.sample_well_data('SY9')
        if well_data:
            # P0: 仅用压力有效点进 data_loss，避免 NaN 导致 loss=nan
            p_valid_mask = well_data.get('p_valid_mask_local')
            if p_valid_mask is not None:
                p_valid_mask = p_valid_mask.astype(bool) if getattr(p_valid_mask, 'dtype', None) != np.dtype('bool') else p_valid_mask
                n_valid = int(np.sum(p_valid_mask))
            else:
                p_obs = well_data['p_obs']
                p_valid_mask = np.isfinite(p_obs) & ~np.isnan(p_obs)
                n_valid = int(np.sum(p_valid_mask))
            if n_valid > 0:
                self.data_xyt = torch.from_numpy(well_data['xyt'][p_valid_mask]).float().to(self.device)
                self.data_p_obs = torch.from_numpy(well_data['p_obs'][p_valid_mask]).float().to(self.device)
                self.data_t_days = well_data['t_days'][p_valid_mask]
            else:
                self.data_xyt = None
                self.data_p_obs = None
                self.data_t_days = None
            self.logger.info(f"  SY9 数据锚点（压力有效）: {n_valid} 条")
            # 全序列与缺失信息（仅用于绘图与报告，不改变训练/验收指标）
            self.data_t_days_full = well_data.get('t_days_full')
            self.data_p_obs_full = well_data.get('p_obs_full')
            self.data_valid_mask = well_data.get('valid_mask')
            self.data_missing_runs = well_data.get('missing_runs', [])
            self.data_xyt_full = well_data.get('xyt_full')
            if self.data_xyt_full is not None:
                self.data_xyt_full = torch.from_numpy(self.data_xyt_full).float().to(self.device)
            self.data_nan_ratio = well_data.get('nan_ratio')
        else:
            self.data_xyt = None
            self.data_p_obs = None
            self.data_t_days = None
            self.data_t_days_full = None
            self.data_p_obs_full = None
            self.data_valid_mask = None
            self.data_missing_runs = []
            self.data_xyt_full = None
            self.data_nan_ratio = None
            self.logger.warning("  未能加载 SY9 数据锚点")
    
    def _compute_pde_scaling(self):
        """
        计算 PDE 物理缩放系数并注入 loss_fn。
        
        α_phys = k / (φ * μ_ref * c_t_ref)    [m²/s]
        c_t_ref = c_g_ref + c_r
        
        归一化坐标下:
            αx = α_phys * t_max_s * (2/Δx)²
            αy = α_phys * t_max_s * (2/Δy)²
        """
        physics_cfg = self.config.get('physics', {})
        priors = physics_cfg.get('priors', {})
        pde_cfg = physics_cfg.get('pde', {})
        domain_cfg = pde_cfg.get('domain', {})
        
        # --- 读取物理先验 ---
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
        
        # --- 尝试用 M3 PVT 查询 cg 覆盖先验 ---
        cg_source = "prior"
        try:
            from physics.pvt import GasPVT
            gas_pvt = GasPVT(config=self.config)
            cg_queried = float(gas_pvt.cg(p_ref, T_ref).item())
            self.logger.info(
                f"  M3 PVT 查询 cg({p_ref} MPa, {T_ref} ℃) = {cg_queried:.6f} 1/MPa "
                f"→ 覆盖先验 {cg_1_per_MPa:.4f}"
            )
            cg_1_per_MPa = cg_queried
            cg_source = f"M3 PVT @ (p={p_ref}, T={T_ref})"
        except Exception as e:
            self.logger.warning(f"  M3 PVT 查询 cg 失败 ({e})，使用先验 {cg_1_per_MPa}")
        
        # --- 域参数 ---
        x_min = domain_cfg.get('x_min_m', self.sampler.x_min)
        x_max = domain_cfg.get('x_max_m', self.sampler.x_max)
        y_min = domain_cfg.get('y_min_m', self.sampler.y_min)
        y_max = domain_cfg.get('y_max_m', self.sampler.y_max)
        t_max_d = domain_cfg.get('t_max_d', self.sampler.t_max)
        
        dx = x_max - x_min                   # m
        dy = y_max - y_min                    # m
        t_max_s = t_max_d * 86400.0           # s
        
        # --- 单位换算（全 SI）---
        k_SI = k_mD * 9.869233e-16           # mD → m²
        mu_SI = mu_mPa_s * 1e-3              # mPa·s → Pa·s
        cg_SI = cg_1_per_MPa * 1e-6          # 1/MPa → 1/Pa
        cr_SI = cr_1_per_MPa * 1e-6          # 1/MPa → 1/Pa
        ct_SI = cg_SI + cr_SI                # 1/Pa
        
        # --- α_phys ---
        alpha_phys = k_SI / (phi * mu_SI * ct_SI)  # m²/s
        
        # --- 归一化系数 ---
        sx = (2.0 / dx) ** 2                 # 1/m²
        sy = (2.0 / dy) ** 2                 # 1/m²
        alpha_x = alpha_phys * t_max_s * sx  # 无量纲
        alpha_y = alpha_phys * t_max_s * sy  # 无量纲
        
        # --- 注入 loss_fn ---
        self.loss_fn.alpha_x = alpha_x
        self.loss_fn.alpha_y = alpha_y
        
        # --- 井周加密比例 ---
        near_well_ratio = (self.config.get('m4_config', {})
                           .get('sampling', {})
                           .get('near_well_ratio', 0.0))
        
        # --- 打印（答辩友好；alpha 为诊断参数，pde_loss 内部使用自洽的 sx/sy 归一化）---
        self.logger.info("=" * 60)
        self.logger.info("PDE 物理缩放参数（诊断参数）")
        self.logger.info("=" * 60)
        self.logger.info(f"  域: Δx = {dx:.0f} m, Δy = {dy:.0f} m, t_max = {t_max_d:.0f} d = {t_max_s:.0f} s")
        self.logger.info(f"  sx = (2/Δx)² = {sx:.6e} 1/m², sy = (2/Δy)² = {sy:.6e} 1/m²")
        self.logger.info(f"  p_ref = {p_ref} MPa, T_ref = {T_ref} ℃")
        self.logger.info(f"  k = {k_mD} mD = {k_SI:.6e} m²")
        self.logger.info(f"  φ = {phi}")
        self.logger.info(f"  μ_ref = {mu_mPa_s} mPa·s = {mu_SI:.6e} Pa·s")
        self.logger.info(f"  c_g_ref = {cg_1_per_MPa:.6f} 1/MPa = {cg_SI:.6e} 1/Pa  (来源: {cg_source})")
        self.logger.info(f"  c_r = {cr_1_per_MPa} 1/MPa = {cr_SI:.6e} 1/Pa")
        self.logger.info(f"  c_t = c_g + c_r = {ct_SI:.6e} 1/Pa")
        self.logger.info(f"  α_phys = k/(φ·μ·c_t) = {alpha_phys:.6f} m²/s")
        self.logger.info(f"  αx = α_phys·t_max_s·sx = {alpha_x:.6f}")
        self.logger.info(f"  αy = α_phys·t_max_s·sy = {alpha_y:.6f}")
        self.logger.info(f"  near_well_ratio = {near_well_ratio}")
        # 厚度场均值注入 loss_fn，使 PDE 残差与地质域一致（替代 loss 内硬编码 90.0）
        if hasattr(self.sampler, 'h_mean') and np.isfinite(self.sampler.h_mean):
            self.loss_fn.h_mean = float(self.sampler.h_mean)
            self.logger.info(f"  h_mean 已从 sampler 注入 loss_fn: {self.loss_fn.h_mean:.2f} m")
        self.logger.info("-" * 60)
        
        # --- 敏感性分析: μ_ref / cg_ref 区间端点 ---
        mu_range = priors.get('mu_ref_mPa_s', {}).get('range', [mu_mPa_s, mu_mPa_s]) \
            if isinstance(priors.get('mu_ref_mPa_s'), dict) else [mu_mPa_s, mu_mPa_s]
        cg_range = priors.get('cg_ref_1_per_MPa', {}).get('range', [cg_1_per_MPa, cg_1_per_MPa]) \
            if isinstance(priors.get('cg_ref_1_per_MPa'), dict) else [cg_1_per_MPa, cg_1_per_MPa]
        
        # α_phys 最大 = k / (φ * μ_min * (cg_min + cr))
        mu_min_SI = mu_range[0] * 1e-3
        mu_max_SI = mu_range[1] * 1e-3
        cg_min_SI = cg_range[0] * 1e-6
        cg_max_SI = cg_range[1] * 1e-6
        
        alpha_max = k_SI / (phi * mu_min_SI * (cg_min_SI + cr_SI))
        alpha_min = k_SI / (phi * mu_max_SI * (cg_max_SI + cr_SI))
        
        self.logger.info("敏感性分析 (μ_ref × cg_ref 区间端点):")
        self.logger.info(f"  μ_ref ∈ [{mu_range[0]}, {mu_range[1]}] mPa·s")
        self.logger.info(f"  cg_ref ∈ [{cg_range[0]}, {cg_range[1]}] 1/MPa")
        self.logger.info(f"  α_phys ∈ [{alpha_min:.6f}, {alpha_max:.6f}] m²/s")
        self.logger.info(f"  αx ∈ [{alpha_min * t_max_s * sx:.6f}, {alpha_max * t_max_s * sx:.6f}]")
        self.logger.info(f"  αy ∈ [{alpha_min * t_max_s * sy:.6f}, {alpha_max * t_max_s * sy:.6f}]")
        
        # --- 2.5D 厚度场审计 ---
        if hasattr(self.sampler, 'h_min'):
            self.logger.info("-" * 60)
            self.logger.info("2.5D 厚度场审计:")
            self.logger.info(f"  h_min = {self.sampler.h_min:.2f} m, "
                             f"h_max = {self.sampler.h_max:.2f} m, "
                             f"h_mean = {self.sampler.h_mean:.2f} m")
            gx_a = np.abs(self.sampler.collocation_gx)
            gy_a = np.abs(self.sampler.collocation_gy)
            gx_m = float(np.nanmax(gx_a)) if np.any(np.isfinite(gx_a)) else 0.0
            gy_m = float(np.nanmax(gy_a)) if np.any(np.isfinite(gy_a)) else 0.0
            self.logger.info(f"  log-thickness gradient |gx|_max = {gx_m:.4f}, |gy|_max = {gy_m:.4f}")
            self.logger.info("  PDE 残差形式: dp/dt_n = αx(d²p/dx_n² + gx·dp/dx_n) "
                             "+ αy(d²p/dy_n² + gy·dp/dy_n)")
            n_oor = int(getattr(self.sampler, 'collocation_is_oor', np.zeros(1)).sum())
            if n_oor > 0:
                pct = 100.0 * n_oor / len(self.sampler.collocation_xy)
                self.logger.info(f"  PDE 域外点过滤: {n_oor}/{len(self.sampler.collocation_xy)} = {pct:.2f}% 不参与 PDE loss")
        self.logger.info("=" * 60)
    
    def _sample_batch(self, seed_offset: int = 0, training_progress: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        生成一个 batch 的采样点。
        采样数量由 YAML m4_config.sampling.{n_domain, n_boundary, n_initial} 驱动。
        training_progress: [0,1]，用于时间采样分布（前期偏早期、后期混合均匀，保证后期压降段被采样）。
        """
        samp_cfg = self.config.get('m4_config', {}).get('sampling', {})
        n_domain = samp_cfg.get('n_domain', self.batch_size // 2)
        n_boundary = samp_cfg.get('n_boundary', self.batch_size // 4)
        n_initial = samp_cfg.get('n_initial', self.batch_size // 4)
        
        x_ic = self.sampler.sample_initial(n_initial, seed=seed_offset)
        x_bc = self.sampler.sample_boundary(n_boundary, seed=seed_offset + 1)
        x_pde = self.sampler.sample_domain(n_domain, seed=seed_offset + 2,
                                           training_progress=training_progress)
        
        batch = {
            'x_ic': torch.from_numpy(x_ic).float().to(self.device),
            'x_bc': torch.from_numpy(x_bc).float().to(self.device),
            'x_pde': torch.from_numpy(x_pde).float().to(self.device),
        }
        
        # 取出 PDE 点对应的 log-thickness 梯度（2.5D 厚度加权）
        gx_np, gy_np = self.sampler.get_last_h_grad()
        if gx_np is not None:
            batch['h_grad'] = {
                'gx': torch.from_numpy(gx_np).float().to(self.device).unsqueeze(-1),
                'gy': torch.from_numpy(gy_np).float().to(self.device).unsqueeze(-1),
            }
        
        # P1: 域外点 PDE 掩码（1=有效，0=过滤），可选
        pde_mask_np = self.sampler.get_last_pde_mask()
        if pde_mask_np is not None:
            batch['pde_mask'] = torch.from_numpy(pde_mask_np).float().to(self.device).unsqueeze(-1)
        
        return batch
    
    def _train_step(self, step: int, weights: Dict[str, float],
                    total_steps: int = 1, stage: str = '?') -> Dict[str, float]:
        """
        执行一步训练（AMP 拆段策略）:
        - IC / BC / Data loss: 可在 autocast(fp16) 下计算
        - PDE loss（二阶导）: 必须在 fp32 下计算
        - backward 统一在 autocast 外执行
        total_steps: 用于计算 training_progress，使时间采样后期覆盖压降段。
        stage: 当前阶段 (A/B/C/D)，供 debug_nan dump 使用。
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        training_progress = step / total_steps if total_steps else 0.0
        batch = self._sample_batch(seed_offset=step, training_progress=training_progress)
        
        # 数据锚点
        x_data = self.data_xyt if weights.get('data', 0) > 0 else None
        p_obs = self.data_p_obs if weights.get('data', 0) > 0 else None
        
        # 2.5D 厚度梯度（若 sampler 提供）
        h_grad = batch.get('h_grad', None)
        
        # Debug NaN: 每步注入 step/stage/batch 供 loss 内 isfinite 失败时 dump
        if self.debug_nan:
            self.loss_fn.set_debug_context(step, stage, batch, x_data, p_obs, h_grad)
        
        w_ic = weights.get('ic', 1.0)
        w_bc = weights.get('bc', 1.0)
        w_pde = weights.get('pde', 0.0)
        w_data = weights.get('data', 0.0)
        w_sw_phys = weights.get('sw_phys', 0.0)
        
        if self.use_amp:
            # --- AMP 拆段: IC/BC/Data 用 autocast, PDE 保持 fp32 ---
            with torch.amp.autocast('cuda'):
                loss_ic = self.loss_fn.ic_loss(self.model, batch['x_ic'])
                loss_bc = self.loss_fn.bc_loss(self.model, batch['x_bc'])
                loss_data = (self.loss_fn.data_loss(self.model, x_data, p_obs)
                             if w_data > 0 and x_data is not None and p_obs is not None
                             else torch.tensor(0.0, device=self.device))
            
            # PDE loss 在 fp32 下（二阶导需要完整精度）+ 厚度加权
            if w_pde > 0:
                loss_pde = self.loss_fn.pde_loss(self.model, batch['x_pde'],
                                                  h_grad=h_grad, pde_mask=batch.get('pde_mask'))
            else:
                loss_pde = torch.tensor(0.0, device=self.device)
            if w_sw_phys > 0:
                loss_sw_phys = self.loss_fn.sw_physics_loss(self.model, batch['x_pde'])
            else:
                loss_sw_phys = torch.tensor(0.0, device=self.device)
            
            total = (w_ic * loss_ic + w_bc * loss_bc
                     + w_pde * loss_pde + w_data * loss_data
                     + w_sw_phys * loss_sw_phys)
            
            # backward 在 autocast 外（PyTorch AMP 文档建议）
            if self.debug_nan:
                torch.autograd.set_detect_anomaly(True)
            try:
                self.scaler.scale(total).backward()
            finally:
                if self.debug_nan:
                    torch.autograd.set_detect_anomaly(False)
            self.scaler.unscale_(self.optimizer)
            if self.debug_nan:
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            losses = {'total': total, 'ic': loss_ic, 'bc': loss_bc,
                      'pde': loss_pde, 'data': loss_data, 'sw_phys': loss_sw_phys}
        else:
            # 全 fp32（也传入 h_grad 用于厚度加权 PDE）
            loss_ic = self.loss_fn.ic_loss(self.model, batch['x_ic'])
            loss_bc = self.loss_fn.bc_loss(self.model, batch['x_bc'])
            loss_data = (self.loss_fn.data_loss(self.model, x_data, p_obs)
                         if w_data > 0 and x_data is not None and p_obs is not None
                         else torch.tensor(0.0, device=self.device))
            if w_pde > 0:
                loss_pde = self.loss_fn.pde_loss(self.model, batch['x_pde'],
                                                  h_grad=h_grad, pde_mask=batch.get('pde_mask'))
            else:
                loss_pde = torch.tensor(0.0, device=self.device)
            if w_sw_phys > 0:
                loss_sw_phys = self.loss_fn.sw_physics_loss(self.model, batch['x_pde'])
            else:
                loss_sw_phys = torch.tensor(0.0, device=self.device)
            
            total = (w_ic * loss_ic + w_bc * loss_bc
                     + w_pde * loss_pde + w_data * loss_data
                     + w_sw_phys * loss_sw_phys)
            losses = {'total': total, 'ic': loss_ic, 'bc': loss_bc,
                      'pde': loss_pde, 'data': loss_data, 'sw_phys': loss_sw_phys}
            if self.debug_nan:
                torch.autograd.set_detect_anomaly(True)
            try:
                losses['total'].backward()
            finally:
                if self.debug_nan:
                    torch.autograd.set_detect_anomaly(False)
            if self.debug_nan:
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        self.scheduler.step()
        
        # 监控 Sw 范围
        with torch.no_grad():
            x_check = batch['x_pde'][:100]
            _, sw_check = self.model(x_check)
            sw_min_val = sw_check.min().item()
            sw_max_val = sw_check.max().item()
        
        # 记录
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['sw_min'] = sw_min_val
        loss_dict['sw_max'] = sw_max_val
        loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
        
        return loss_dict
    
    def train(self) -> Dict[str, list]:
        """
        执行完整的分阶段训练
        
        Returns:
            训练历史字典
        """
        self.logger.info("=" * 60)
        self.logger.info("M4 PINN 基线训练开始")
        self.logger.info("=" * 60)
        
        total_steps = self.max_steps
        
        # 阶段划分 —— 从 YAML m4_config.training_stages 读取
        stages_cfg = self.config.get('m4_config', {}).get('training_stages', {})
        sa = stages_cfg.get('stage_a', {})
        sb = stages_cfg.get('stage_b', {})
        sc = stages_cfg.get('stage_c', {})
        sd = stages_cfg.get('stage_d', {})
        
        frac_a = sa.get('fraction', 0.20)
        frac_b = sb.get('fraction', 0.30)
        frac_c = sc.get('fraction', 0.30)
        # stage_d takes the rest
        
        stage_a_end = int(total_steps * frac_a)
        stage_b_end = int(total_steps * (frac_a + frac_b))
        stage_c_end = int(total_steps * (frac_a + frac_b + frac_c))
        
        # 各阶段权重（从 YAML 读取）
        w_a = sa.get('weights', {'ic': 10.0, 'bc': 5.0, 'pde': 0.0, 'data': 0.0, 'sw_phys': 0.0})
        w_b_start = sb.get('weights_start', {'ic': 5.0, 'bc': 3.0, 'pde': 0.001, 'data': 0.0, 'sw_phys': 0.5})
        w_b_end = sb.get('weights_end', {'ic': 5.0, 'bc': 3.0, 'pde': 0.1, 'data': 0.5, 'sw_phys': 1.0})
        w_c_start = sc.get('weights_start', {'ic': 3.0, 'bc': 2.0, 'pde': 0.1, 'data': 1.0, 'sw_phys': 1.5})
        w_c_end = sc.get('weights_end', {'ic': 3.0, 'bc': 2.0, 'pde': 0.5, 'data': 3.0, 'sw_phys': 2.0})
        w_d = sd.get('weights', {'ic': 1.0, 'bc': 1.0, 'pde': 1.0, 'data': 5.0, 'sw_phys': 2.0})
        
        self.logger.info(
            f"训练阶段: A[0-{stage_a_end}] B[{stage_a_end}-{stage_b_end}] "
            f"C[{stage_b_end}-{stage_c_end}] D[{stage_c_end}-{total_steps}]"
        )
        
        start_time = time.time()
        best_loss = float('inf')
        
        def _lerp_weights(w_start, w_end, progress):
            """线性插值两组权重"""
            return {k: w_start.get(k, 0) + progress * (w_end.get(k, 0) - w_start.get(k, 0))
                    for k in set(w_start) | set(w_end)}
        
        for step in range(total_steps):
            # 确定当前阶段和权重（YAML 驱动）
            if step < stage_a_end:
                stage = 'A'
                weights = w_a
            elif step < stage_b_end:
                stage = 'B'
                progress = (step - stage_a_end) / max(stage_b_end - stage_a_end, 1)
                weights = _lerp_weights(w_b_start, w_b_end, progress)
            elif step < stage_c_end:
                stage = 'C'
                progress = (step - stage_b_end) / max(stage_c_end - stage_b_end, 1)
                weights = _lerp_weights(w_c_start, w_c_end, progress)
            else:
                stage = 'D'
                weights = w_d
            
            # 训练一步
            loss_dict = self._train_step(step, weights, total_steps=total_steps, stage=stage)
            # P0: fail-fast — 非有限损失立即终止，不保存当前步
            loss_keys = ['total', 'ic', 'bc', 'pde', 'data', 'sw_phys']
            if not all(np.isfinite(loss_dict.get(k, 0)) for k in loss_keys):
                self.logger.error(
                    f"M4 训练出现非有限损失 (step={step}, stage={stage}): "
                    f"total={loss_dict.get('total')}, data={loss_dict.get('data')}, pde={loss_dict.get('pde')}"
                )
                raise RuntimeError(
                    "M4 训练出现 NaN/Inf 损失，已终止（未保存当前步）。请检查数据锚点与 PDE 参数。"
                )
            
            # 记录历史
            self.history['step'].append(step)
            self.history['total'].append(loss_dict['total'])
            self.history['ic'].append(loss_dict['ic'])
            self.history['bc'].append(loss_dict['bc'])
            self.history['pde'].append(loss_dict['pde'])
            self.history['data'].append(loss_dict['data'])
            self.history['sw_phys'].append(loss_dict.get('sw_phys', 0.0))
            self.history['lr'].append(loss_dict['lr'])
            self.history['sw_min'].append(loss_dict['sw_min'])
            self.history['sw_max'].append(loss_dict['sw_max'])
            self.history['stage'].append(stage)
            
            # 保存最佳模型：仅在 Stage C/D（有 Data 和 PDE 约束后）才更新 best，避免 Stage A 总损失更小被误选
            if step >= stage_c_end and loss_dict['total'] < best_loss:
                best_loss = loss_dict['total']
                self._save_checkpoint('best')
            
            # 日志（每 100 步一次，便于确认 GPU 在跑）
            if step % 100 == 0 or step == total_steps - 1:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"[Step {step:5d}/{total_steps}] Stage {stage} | "
                    f"Loss: {loss_dict['total']:.4e} "
                    f"(IC={loss_dict['ic']:.3e}, BC={loss_dict['bc']:.3e}, "
                    f"PDE={loss_dict['pde']:.3e}, Data={loss_dict['data']:.3e}, sw_phys={loss_dict.get('sw_phys', 0):.3e}) | "
                    f"Sw=[{loss_dict['sw_min']:.4f}, {loss_dict['sw_max']:.4f}] | "
                    f"lr={loss_dict['lr']:.2e} | {elapsed:.1f}s"
                )
        
        total_time = time.time() - start_time
        self.logger.info(f"\n训练完成! 总耗时: {total_time:.1f}s, 最佳损失: {best_loss:.4e}")
        
        # 保存最终模型
        self._save_checkpoint('final')
        
        return self.history
    
    def _save_checkpoint(self, tag: str):
        """保存检查点"""
        path = os.path.join(self.ckpt_dir, f'pinn_{tag}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, tag: str = 'best'):
        """加载检查点"""
        path = os.path.join(self.ckpt_dir, f'pinn_{tag}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.logger.info(f"已加载检查点: {path}")
            return True
        return False
    
    def plot_training_history(self, save: bool = True) -> str:
        """绘制训练曲线"""
        self.logger.info("生成训练曲线图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('M4 PINN 基线训练曲线', fontsize=16, fontweight='bold')
        
        steps = self.history['step']
        
        # 标注阶段分界
        stage_changes = []
        for i in range(1, len(self.history['stage'])):
            if self.history['stage'][i] != self.history['stage'][i-1]:
                stage_changes.append((steps[i], self.history['stage'][i]))
        
        # (1) 总损失
        ax1 = axes[0, 0]
        ax1.semilogy(steps, self.history['total'], 'b-', linewidth=0.8, alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('总损失')
        ax1.grid(True, alpha=0.3)
        for s, label in stage_changes:
            ax1.axvline(x=s, color='red', linestyle='--', alpha=0.3)
            ax1.text(s, ax1.get_ylim()[1], f'Stage {label}', fontsize=8, color='red')
        
        # (2) 分项损失
        ax2 = axes[0, 1]
        ax2.semilogy(steps, self.history['ic'], label='IC', linewidth=0.8, alpha=0.7)
        ax2.semilogy(steps, self.history['bc'], label='BC', linewidth=0.8, alpha=0.7)
        pde_vals = [max(v, 1e-12) for v in self.history['pde']]
        data_vals = [max(v, 1e-12) for v in self.history['data']]
        ax2.semilogy(steps, pde_vals, label='PDE', linewidth=0.8, alpha=0.7)
        ax2.semilogy(steps, data_vals, label='Data', linewidth=0.8, alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('分项损失')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # (3) Sw 范围监控
        ax3 = axes[1, 0]
        ax3.plot(steps, self.history['sw_min'], 'b-', linewidth=0.8, label='Sw_min')
        ax3.plot(steps, self.history['sw_max'], 'r-', linewidth=0.8, label='Sw_max')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Sw 范围')
        ax3.set_title('含水饱和度范围监控')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([-0.1, 1.1])
        
        # (4) 学习率
        ax4 = axes[1, 1]
        ax4.semilogy(steps, self.history['lr'], 'g-', linewidth=0.8)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('学习率调度')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.fig_dir, 'M4_training_history.png')
            fig.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"训练曲线图已保存: {filepath}")
            return filepath
        
        plt.close(fig)
        return ''
    
    def plot_pressure_comparison(self, save: bool = True, show_obs_line: bool = False) -> str:
        """
        绘制 SY9 压力预测 vs 观测对比图（M4 验收核心图件）。
        观测用散点避免跨缺失段伪连续；缺失区间灰色阴影；指标仅在有效观测点计算。
        """
        if self.data_xyt is None:
            self.logger.warning("无数据锚点，跳过压力对比图")
            return ''
        
        self.logger.info("生成 SY9 压力对比图...")
        
        dp_wellbore = getattr(self.loss_fn, 'dp_wellbore', 18.0)
        p_obs = self.data_p_obs.cpu().numpy().flatten()
        t_days = self.data_t_days  # 仅有效点，用于指标与 Sw 子图
        
        # 全时间网格与缺失信息（若有）
        t_all = getattr(self, 'data_t_days_full', None)
        p_obs_full = getattr(self, 'data_p_obs_full', None)
        valid_mask = getattr(self, 'data_valid_mask', None)
        missing_runs = getattr(self, 'data_missing_runs', []) or []
        nan_ratio = getattr(self, 'data_nan_ratio', None)
        xyt_full = getattr(self, 'data_xyt_full', None)
        
        self.model.eval()
        with torch.no_grad():
            p_pred_form, sw_pred = self.model(self.data_xyt)
            p_pred_form = p_pred_form.cpu().numpy().flatten()
            sw_pred = sw_pred.cpu().numpy().flatten()
        p_pred_whp = p_pred_form - dp_wellbore  # 仅在有效点，用于指标
        
        if xyt_full is not None:
            with torch.no_grad():
                p_pred_form_all, sw_pred_all = self.model(xyt_full)
                p_pred_form_all = p_pred_form_all.cpu().numpy().flatten()
                sw_pred_all = sw_pred_all.cpu().numpy().flatten()
            p_pred_whp_all = p_pred_form_all - dp_wellbore
            t_plot = t_all
            sw_plot = sw_pred_all
        else:
            p_pred_whp_all = p_pred_whp
            t_plot = t_days
            sw_plot = sw_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('M4 PINN 基线: SY9 井验收', fontsize=14, fontweight='bold')
        
        ax1 = axes[0]
        # 缺失区间灰色阴影
        for run in missing_runs:
            start_idx, end_idx = run[0], run[1]
            if t_plot is not None and end_idx < len(t_plot):
                t_start = t_plot[start_idx]
                t_end = t_plot[end_idx]
                ax1.axvspan(t_start, t_end, alpha=0.15, color='gray', zorder=0)
        # 观测：散点（仅有效点），避免伪连续
        ax1.scatter(t_days, p_obs, c='blue', s=8, alpha=0.7, label='观测值', zorder=2)
        if show_obs_line and p_obs_full is not None and t_plot is not None and len(p_obs_full) == len(t_plot):
            # 带 NaN 的折线（matplotlib 在 NaN 处自动断线，不 dropna 连线）
            ax1.plot(t_plot, p_obs_full, 'b-', linewidth=0.8, alpha=0.4, label='观测(含缺失)')
        # PINN 预测：连续线
        ax1.plot(t_plot, p_pred_whp_all, 'r--', linewidth=1.5, label='PINN 预测', alpha=0.7)
        ax1.set_xlabel('时间 (天)', fontsize=12)
        ax1.set_ylabel('压力 (MPa)', fontsize=12)
        ax1.set_title('SY9 压力时间序列', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 指标仅在有效观测点计算
        valid = np.isfinite(p_obs) & np.isfinite(p_pred_whp) & (p_obs > 0)
        lines_annot = []
        if np.any(valid):
            _eps_mape = 1.0
            mape = np.mean(np.abs((p_obs[valid] - p_pred_whp[valid]) / (p_obs[valid] + _eps_mape))) * 100
            rmse = np.sqrt(np.mean((p_obs[valid] - p_pred_whp[valid]) ** 2))
            lines_annot.append(f'MAPE={mape:.2f}%\nRMSE={rmse:.2f} MPa')
        if nan_ratio is not None:
            lines_annot.append(f'nan_ratio={nan_ratio:.2%}')
        if missing_runs:
            longest_days = max(r[4] for r in missing_runs)
            lines_annot.append(f'最长缺失={longest_days}天')
        if lines_annot:
            ax1.text(0.05, 0.05, '\n'.join(lines_annot), transform=ax1.transAxes, fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2 = axes[1]
        ax2.plot(t_plot, sw_plot, 'g-', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('时间 (天)', fontsize=12)
        ax2.set_ylabel('含水饱和度 Sw', fontsize=12)
        ax2.set_title('SY9 Sw 演化', fontsize=13)
        ax2.set_ylim([-0.05, 1.05])
        ax2.grid(True, alpha=0.3)
        
        sw_in_range = np.all((sw_plot >= 0) & (sw_plot <= 1))
        status = "Sw ∈ [0,1] ✅" if sw_in_range else "Sw 越界 ❌"
        ax2.text(0.05, 0.95, status, transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round',
                          facecolor='lightgreen' if sw_in_range else 'lightsalmon',
                          alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.fig_dir, 'M4_pressure_comparison.png')
            fig.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"压力对比图已保存: {filepath}")
            return filepath
        
        plt.close(fig)
        return ''
