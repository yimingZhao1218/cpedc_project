"""
M5 增强 PINN 网络 (v3 — 三箭修复版)
======================================
修复记录:
    P1-FIX: get_inversion_param_tensors 返回 k_frac_mD (替代 k_eff_mD + f_frac)
    P1-FIX: get_inversion_params 审计输出同步更新
    P1-FIX: count_parameters_breakdown 不再统计 wi_factor

在 M4 PINNNet 基础上增加:
    1. 可训练物理反演参数 (k_frac) — 通过 WellModel 管理
    2. 每口井独立 p_wf(t) 隐变量网络 — 通过 WellModel 管理
    3. 井—藏耦合产量计算
    4. k(x,y) 空间渗透率场子网络 — 计划书核心要求
    5. 场网络保持 M4 结构 (输入 x,y,t → 输出 p, Sw)
    6. 油管压力→井底流压转换 (不直接用 WHP 约束 p_cell)

不破坏 M4 基线:
    - 继承 PINNNet 的结构
    - 扩展 forward / forward_with_grad 接口
"""

import os
import sys
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("m5_model 需要 PyTorch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger
from pinn.model import PINNNet, PermeabilityNet
from pinn.well_model import WellModel, PeacemanWI


class M5PINNNet(nn.Module):
    """
    M5 增强 PINN 网络 (v3)

    组成:
        - field_net: 场网络 (x,y,t) → (p, Sw)，继承自 PINNNet
        - k_net: 渗透率子网络 (x,y) → log_k(x,y)
        - well_model: 井模型 (Peaceman WI + p_wf + source term)
        - dp_wellbore: 井筒压差可学习参数 (油管压力→井底流压)
    """

    def __init__(self, config: dict, well_ids: Optional[List[str]] = None):
        super().__init__()
        self.logger = setup_logger('M5PINNNet')
        self.config = config

        # 场网络 (复用 M4 架构, 含 Fourier Features)
        self.field_net = PINNNet(config)

        # k(x,y) 空间渗透率子网络
        self.use_k_net = config.get('model', {}).get('architecture', {}).get('use_k_net', True)
        if self.use_k_net:
            self.k_net = PermeabilityNet(config)
            self.logger.info(f"  k_net 启用: {sum(p.numel() for p in self.k_net.parameters())} 参数")
        else:
            self.k_net = None

        # 井列表
        if well_ids is None:
            data_cfg = config.get('data', {})
            if data_cfg.get('mode') == 'single_well':
                well_ids = [data_cfg.get('primary_well', 'SY9')]
            else:
                well_ids = data_cfg.get('wells', ['SY9'])
        self.well_ids = well_ids

        # 井模型 (v3: 使用 k_frac, 无 wi_factor)
        device = config.get('runtime', {}).get('device', 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.well_model = WellModel(config, well_ids, device=device)

        # --- 井筒压差 (v3.7: 冻结为试油实测值, 不再可学习) ---
        # v3.6 证实: dp prior 贡献仅占总损失 0.0003% (168,000:1),
        # 无论 prior 权重如何调整都无法阻止 dp 从 13.3 漂移到 16.24 MPa.
        # WHP loss 把 dp 当逃逸通道: dp↑ → p_wf↑ → 驱动压差↓ → qg 系统性低估.
        # 冻结 dp=13.3 (试油实测: WHP=57.93, BHP=71.23, Δp=13.3 MPa) 是唯一可靠方案.
        # 驱动压差从 0.76 MPa (dp=16.24) 恢复到 3.70 MPa (dp=13.3), 提升 4.9x.
        dp_init = 13.3  # MPa (试油实测值)
        self.register_buffer('_dp_wellbore_raw', torch.tensor(dp_init))

        # --- v3.17: 井眼奇异性分解 (Well Singularity Decomposition) ---
        # p(x,y,t) = p_nn(x,y,t) + A(t) × φ(r)
        # φ(r) = log(r + r_w) / log(r_w)  → 1.0 at well, ~0 far away
        # A(t) 由小 MLP 学习, 表示时变泄油漏斗振幅 (MPa)
        # 物理依据: ∇²log(r) = 0 (r>0), 修正项不增加 PDE 域内残差
        # 参考: Tancik et al. (2020) + Wang et al. (2021) PINN well treatment
        self.use_well_singularity = config.get('model', {}).get(
            'architecture', {}).get('use_well_singularity', True)
        if self.use_well_singularity:
            self.well_log_amp_net = nn.Sequential(
                nn.Linear(1, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, 1)
            )
            # 近零初始化: 训练初期 p_correction ≈ 0, 不扰动 Stage A
            for m in self.well_log_amp_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)
            # 井位坐标 buffer (归一化空间, 由 trainer 注入)
            self.register_buffer('_well_xy_norm', torch.zeros(1, 2))
            # r_w 归一化: 物理井径 ~0.1m, 域 ~17400m, r_w_norm ≈ 1e-5
            # 用稍大值 1e-4 保证数值稳定
            self._r_w_norm = 1e-4
            self._log_r_w = math.log(self._r_w_norm)  # ≈ -9.21
            self._well_xy_set = False
        else:
            self._well_xy_set = False

        self.logger.info(
            f"M5PINNNet v3.17 初始化: field_params={self.field_net.count_parameters()}, "
            f"well_model 含 {len(well_ids)} 口井, "
            f"k_net={'启用' if self.use_k_net else '关闭'}, "
            f"k_frac_init={self.well_model.peaceman.k_frac_mD.item():.2f} mD, "
            f"dp_wellbore_init={self._dp_wellbore_raw.item():.1f} MPa, "
            f"well_singularity={'启用' if self.use_well_singularity else '关闭'}"
        )

    @property
    def dp_wellbore(self) -> torch.Tensor:
        """井筒压差 (MPa), 固定为试油实测值 13.3 MPa (v3.7: 不再可学习)"""
        return self._dp_wellbore_raw

    def set_well_xy_norm(self, well_xy_norm: torch.Tensor):
        """设置归一化井位坐标 (由 trainer 在加载数据后调用)"""
        if not self.use_well_singularity:
            return
        if well_xy_norm.dim() == 1:
            well_xy_norm = well_xy_norm.unsqueeze(0)
        self._well_xy_norm.copy_(well_xy_norm[:1].detach())  # 取第一口井
        self._well_xy_set = True
        self.logger.info(
            f"  v3.17 well singularity: well_xy_norm = "
            f"({self._well_xy_norm[0,0].item():.6f}, {self._well_xy_norm[0,1].item():.6f})"
        )

    def _well_singularity_correction(self, xyt: torch.Tensor) -> torch.Tensor:
        """
        计算井眼奇异性压力修正项
        
        p_correction = A(t) × φ(r)
        φ(r) = log(max(r, r_w)) / log(r_w)  ∈ [0, 1]
        A(t) = MLP(t_norm)  ∈ ℝ (MPa)
        """
        # 到井的归一化距离
        dx = xyt[:, 0:1] - self._well_xy_norm[0, 0]
        dy = xyt[:, 1:2] - self._well_xy_norm[0, 1]
        r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-12)
        
        # 归一化对数基函数: 井点=1, 远处→0
        log_r = torch.log(r.clamp(min=self._r_w_norm))
        phi_r = (log_r / self._log_r_w).clamp(0.0, 1.0)
        
        # 时变振幅 A(t)
        t_norm = xyt[:, 2:3]
        A_t = self.well_log_amp_net(t_norm)  # (B, 1) MPa
        
        return A_t * phi_r  # (B, 1)

    def forward(self, xyt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v3.17: 场网络前向传播 (含井眼奇异性修正)
        p = p_nn(x,y,t) + A(t)×φ(r)
        """
        p_nn, sw = self.field_net(xyt)
        
        if self.use_well_singularity and self._well_xy_set:
            p_correction = self._well_singularity_correction(xyt)
            p = p_nn + p_correction
            p = torch.clamp(p, 1.0, 150.0)
        else:
            p = p_nn
        
        return p, sw

    def forward_with_grad(self, xyt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        v3.17: 带梯度前向 (含井眼奇异性修正)
        autograd 自动传播 log(r) 修正项的解析梯度
        """
        # 无修正时委托给 field_net (更高效)
        if not (self.use_well_singularity and self._well_xy_set):
            return self.field_net.forward_with_grad(xyt)
        
        xyt = xyt.requires_grad_(True)
        p, sw = self.forward(xyt)  # 含修正项
        
        grad_p = torch.autograd.grad(
            p, xyt,
            grad_outputs=torch.ones_like(p),
            create_graph=True, retain_graph=True
        )[0]
        dp_dx = grad_p[:, 0:1]
        dp_dy = grad_p[:, 1:2]
        dp_dt = grad_p[:, 2:3]
        
        grad_sw = torch.autograd.grad(
            sw, xyt,
            grad_outputs=torch.ones_like(sw),
            create_graph=True, retain_graph=True
        )[0]
        dsw_dx = grad_sw[:, 0:1]
        dsw_dy = grad_sw[:, 1:2]
        dsw_dt = grad_sw[:, 2:3]
        
        return {
            'p': p, 'sw': sw,
            'dp_dx': dp_dx, 'dp_dy': dp_dy, 'dp_dt': dp_dt,
            'dsw_dx': dsw_dx, 'dsw_dy': dsw_dy, 'dsw_dt': dsw_dt,
            'xyt': xyt,
        }

    def get_k_field(self, xy: torch.Tensor) -> Optional[torch.Tensor]:
        """获取渗透率场 k(x,y) in mD"""
        if self.k_net is not None:
            return self.k_net.get_k_mD(xy)
        return None

    def evaluate_at_well(self,
                         well_id: str,
                         well_xyt_norm: torch.Tensor,
                         h_well: float = 90.0,
                         bg_val: float = 0.002577,
                         krg_val: Optional[torch.Tensor] = None,
                         prod_hours_norm: Optional[torch.Tensor] = None,
                         casing_norm: Optional[torch.Tensor] = None
                         ) -> Dict[str, torch.Tensor]:
        """
        在井位处评估场变量并计算产量
        """
        p_cell, sw_cell = self.forward(well_xyt_norm)  # v3.17: 含井眼奇异性修正

        # k_frac 梯度贯通: k_net 仅用于 PDE 空间渗透率场,
        # Peaceman WI 始终使用 k_frac (近井裂缝增效渗透率)
        k_local_mD = None

        t_norm = well_xyt_norm[:, 2:3]
        well_result = self.well_model.compute_well_rate(
            well_id=well_id,
            p_cell=p_cell,
            sw_cell=sw_cell,
            t_norm=t_norm,
            h_well=h_well,
            bg_val=bg_val,
            krg_val=krg_val,
            k_local_mD=k_local_mD,
            prod_hours_norm=prod_hours_norm,
            casing_norm=casing_norm,
        )

        result = {
            'p_cell': p_cell,
            'sw_cell': sw_cell,
            'p_wf': well_result['p_wf'],
            'qg': well_result['qg'],
            'WI': well_result['WI'],
            'lambda_g': well_result['lambda_g'],
        }
        # v3.14: 透传水相产量 qw (如果 well_model 计算了)
        if 'qw' in well_result:
            result['qw'] = well_result['qw']
        return result

    def convert_whp_to_pwf(self, whp: torch.Tensor) -> torch.Tensor:
        """油管压力 → 井底流压转换"""
        return whp + self.dp_wellbore

    def get_pwf(self, well_id: str, t_norm: torch.Tensor) -> torch.Tensor:
        """获取指定井的 p_wf(t)；v10 压差参数化下为 p_ref - delta（无 p_cell 时的参考值）"""
        return self.well_model.get_all_pwf(t_norm)[well_id]

    def get_all_pwf(self, t_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取所有井的 p_wf(t)"""
        return self.well_model.get_all_pwf(t_norm)

    def count_parameters(self) -> int:
        """统计总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_breakdown(self) -> Dict[str, int]:
        """分类统计参数量"""
        field_params = sum(
            p.numel() for p in self.field_net.parameters() if p.requires_grad
        )
        well_params = sum(
            p.numel() for p in self.well_model.parameters() if p.requires_grad
        )
        k_params = sum(
            p.numel() for p in self.k_net.parameters() if p.requires_grad
        ) if self.k_net is not None else 0

        singularity_params = sum(
            p.numel() for p in self.well_log_amp_net.parameters() if p.requires_grad
        ) if self.use_well_singularity else 0

        return {
            'field_net': field_params,
            'well_model': well_params,
            'k_net': k_params,
            'well_singularity': singularity_params,  # v3.17
            'dp_wellbore': 0,  # v3.7: frozen buffer, not trainable
            'total': field_params + well_params + k_params + singularity_params,
        }

    def get_inversion_params(self) -> Dict[str, float]:
        """获取当前反演参数值 (v3: k_frac 替代 k_eff + f_frac)"""
        audit = self.well_model.get_audit_dict()
        audit['dp_wellbore_MPa'] = self.dp_wellbore.item()
        # v3 向后兼容: 仍输出 k_eff_mD 键 (= k_frac_mD, 用于日志)
        audit['k_eff_mD'] = audit.get('k_frac_mD', 0.0)
        return audit

    def get_inversion_param_tensors(self) -> Dict[str, torch.Tensor]:
        """
        获取反演参数的张量 (用于先验正则化)

        v3 变更:
            - 'k_frac_mD' 替代 'k_eff_mD' + 'f_frac'
            - 保留 'k_eff_mD' 键 (= k_frac_mD) 供 PDE loss 中 k_eff_mD_tensor 路径
        """
        result = {
            'k_frac_mD': self.well_model.peaceman.k_frac_mD,
            'k_eff_mD': self.well_model.peaceman.k_frac_mD,  # 向后兼容
            'dp_wellbore': self.dp_wellbore,
        }
        return result
