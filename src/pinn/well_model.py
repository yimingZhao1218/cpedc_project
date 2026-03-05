"""
M5 井模型模块 (Well Model) — v3 修复版
========================================
修复记录 (v3 — 三箭修复):
    P1-FIX: 合并 k_eff × f_frac 为单一乘积参数 k_frac (消除参数不可辨识性)
    P1-FIX: 删除 wi_factor (安全阀副作用导致 k_eff 失去调节动力)
    P2-FIX: 缩窄 p_wf sigmoid 范围 [30, 80] MPa (由 config 控制)

核心功能:
    1. WellModel: 井—藏耦合关系 q_g = WI · λ_g · (p_cell - p_wf)
    2. SourceTerm: 将井流量通过高斯核分配到 PDE 配点
    3. PwfHiddenVariable: p_wf(t) 可学习隐变量（有界 + 平滑）

依据:
    - Peaceman 模型: WI = 2πkh / (ln(r_e/r_w) + s)
    - 有限支撑核避免理想点源不稳定
"""

import os
import sys
import math
import numpy as np
from typing import Dict, Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("well_model 需要 PyTorch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger
from pinn.torch_physics import TorchPVT


class PeacemanWI(nn.Module):
    """
    Peaceman 井生产指数 (Well Index) 计算 — v3 重参数化版

    ===== P1-FIX: 消除参数不可辨识性 =====

    旧版 WI = 2π · k_eff · h · f_frac / (ln(r_e/r_w) + skin)
    问题: k_eff 和 f_frac 以乘积形式出现, 任意 (k₁,f₁) 和 (k₂,f₂)
          只要 k₁f₁ = k₂f₂ 就产生相同 WI.
          → 优化器只关心乘积, 不关心个体 → k_eff 僵死.

    新版 WI = 2π · k_frac · h / (ln(r_e/r_w) + skin)
    其中 k_frac = k_eff × f_frac 是唯一可训练标量 (mD),
    物理意义: "等效裂缝增强渗透率" (fracture-enhanced permeability).

    初始值: k_frac_init = k_eff × f_frac (由 config 指定, 默认 5.0 × 8.0 = 40.0 mD)
    合理范围: [0.1, 500] mD (覆盖从低渗基质到高导裂缝)

    ===== P1-FIX: 删除 wi_factor =====
    wi_factor 是每口井的 WI 修正因子, 原意是微调.
    但训练中 wi_factor=0.0854 击穿下界 clamp(0.1),
    说明它成为 k_eff 僵死后优化器的唯一"逃生通道",
    引入了非物理的自由度. 删除后 k_frac 成为唯一控制 WI 的参数.
    """

    def __init__(self, config: dict, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.logger = setup_logger('PeacemanWI')

        physics_cfg = config.get('physics', {})
        priors = physics_cfg.get('priors', {})

        # ===== P1-FIX: 单一乘积参数 k_frac (mD) =====
        # 从 config 读取 k_eff 和 f_frac 先验, 计算乘积作为初始值
        k_eff_cfg = priors.get('k_eff_mD', {})
        k_eff_init = float(k_eff_cfg.get('value', 5.0)) if isinstance(k_eff_cfg, dict) else 5.0

        f_frac_cfg = priors.get('frac_conductivity_factor', {})
        f_frac_init = float(f_frac_cfg.get('value', 10.0)) if isinstance(f_frac_cfg, dict) else 10.0

        # k_frac = k_eff × f_frac
        k_frac_init = k_eff_init * f_frac_init  # config: k_eff × f_frac

        # v3.3-FIX: sigmoid 参数化 (替代 softplus + clamp)
        # k = k_min + (k_max - k_min) * sigmoid(raw)
        # 反推初始值: sigmoid(raw) = (k_init - k_min) / (k_max - k_min)
        #             raw = logit(ratio) = log(ratio / (1 - ratio))
        self.k_frac_bounds_mD = [0.1, 100.0]  # v3.4: 缩窄上界 500→100, 消除低 k_init 的 sigmoid ratio clamping
        k_min, k_max = self.k_frac_bounds_mD
        k_range = k_max - k_min
        
        # 确保初始值在范围内
        k_frac_init = max(k_min + 0.01, min(k_max - 0.01, k_frac_init))
        ratio = (k_frac_init - k_min) / k_range  # ∈ (0, 1)
        ratio = max(0.001, min(0.999, ratio))  # v3.4: 0.01→0.001 避免小 k_init 被截断
        raw_init = math.log(ratio / (1.0 - ratio))  # logit
        
        self._k_frac_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # 保留原始先验值用于审计和先验正则化
        self._k_eff_prior = k_eff_init
        self._f_frac_prior = f_frac_init

        # --- 固定几何参数 ---
        pde_domain = physics_cfg.get('pde', {}).get('domain', {})
        dx = pde_domain.get('x_max_m', 18555317.9) - pde_domain.get('x_min_m', 18537917.9)
        dy = pde_domain.get('y_max_m', 3357340.4) - pde_domain.get('y_min_m', 3346340.4)

        m5_cfg = config.get('m5_config', {})
        well_cfg = m5_cfg.get('well_model', {})
        self.r_w = well_cfg.get('r_w_m', 0.1)
        self.skin = well_cfg.get('skin', 0.0)

        # r_e: 等效排泄半径 (可学习, sigmoid约束)
        # v4.8: Peaceman公式r_e≈0.28√(Δx²+Δy²)在PINN无网格框架下缺乏严格依据,
        #       将r_e升级为可学习参数, 让产量数据驱动排泄半径反演.
        #       初始值仍用Peaceman估算作为合理先验.
        n_colloc_est = 2000
        n_side = max(math.sqrt(n_colloc_est), 10)
        dx_colloc = dx / n_side
        dy_colloc = dy / n_side
        r_e_init = 0.28 * math.sqrt(dx_colloc**2 + dy_colloc**2)  # ≈128.9m

        # sigmoid参数化: r_e = r_e_min + (r_e_max - r_e_min) * sigmoid(raw)
        self.r_e_bounds_m = [50.0, 500.0]  # 物理合理范围: 裂缝尺度~排泄半径
        r_e_min, r_e_max = self.r_e_bounds_m
        r_e_range = r_e_max - r_e_min
        r_e_init = max(r_e_min + 1.0, min(r_e_max - 1.0, r_e_init))
        r_e_ratio = (r_e_init - r_e_min) / r_e_range
        r_e_ratio = max(0.01, min(0.99, r_e_ratio))
        r_e_raw_init = math.log(r_e_ratio / (1.0 - r_e_ratio))  # logit

        self._r_e_raw = nn.Parameter(torch.tensor(r_e_raw_init, dtype=torch.float32))
        self._r_e_prior = r_e_init  # 保留先验值用于审计和正则化

        self.logger.info(
            f"PeacemanWI v4.8 初始化: k_frac_init={k_frac_init:.2f} mD "
            f"(= k_eff {k_eff_init} × f_frac {f_frac_init}), "
            f"r_e_init={r_e_init:.1f} m (可学习, 范围[{r_e_min:.0f},{r_e_max:.0f}]m), "
            f"r_w={self.r_w} m, skin={self.skin}"
        )

    @property
    def k_frac_mD(self) -> torch.Tensor:
        """
        等效裂缝增强渗透率 (mD), 正值, 有界
        
        v3.3-FIX: 使用 sigmoid 软约束替代 clamp 硬截断
        - 硬截断在边界处梯度=0, 导致优化停滞
        - sigmoid 在整个区间内保持梯度连续
        """
        k_min, k_max = self.k_frac_bounds_mD
        k_range = k_max - k_min
        # sigmoid 将 _k_frac_raw 从 (-∞, +∞) 映射到 (k_min, k_max)
        # sigmoid(0) = 0.5 → k = k_min + 0.5 * k_range = 中点
        k = k_min + k_range * torch.sigmoid(self._k_frac_raw)
        return k

    @property
    def k_frac_SI(self) -> torch.Tensor:
        """等效裂缝增强渗透率 (m²)"""
        return self.k_frac_mD * 9.869233e-16

    @property
    def r_e(self) -> torch.Tensor:
        """
        等效排泄半径 (m), 正值, 有界
        
        v4.8: sigmoid 软约束 [50, 500] m
        - Peaceman公式初始值≈128.9m
        - 训练中由产量数据驱动收敛
        """
        r_min, r_max = self.r_e_bounds_m
        r_range = r_max - r_min
        return r_min + r_range * torch.sigmoid(self._r_e_raw)

    # ===== 向后兼容属性 (供 PDE 中 k_eff 引用路径) =====
    @property
    def k_eff_mD(self) -> torch.Tensor:
        """向后兼容: 返回 k_frac_mD (供 PDE loss 中 k_eff_mD_tensor 引用)"""
        return self.k_frac_mD

    @property
    def k_eff_SI(self) -> torch.Tensor:
        """向后兼容"""
        return self.k_frac_SI

    def compute_WI(self,
                   h_well: torch.Tensor,
                   k_SI_override: Optional[torch.Tensor] = None
                   ) -> torch.Tensor:
        """
        计算井生产指数

        v3: WI = 2π · k_frac · h_well / (ln(r_e/r_w) + skin)
        注意: 不再包含 f_frac (已合并入 k_frac)

        Args:
            h_well: 井段厚度 (m), shape (n_wells,) or scalar
            k_SI_override: 可选, 井位局部渗透率 (m²).
                          当 k_net 启用时, 传入 k_net(well_xy) 替代 k_frac.
                          ★ 注意: k_net 输出已包含裂缝增效 (由 k_net 自行学习),
                          因此直接使用, 不再乘以 f_frac.

        Returns:
            WI: shape same as h_well, 单位 m³ (SI)
        """
        # v4.8: r_e 现为 tensor, 用 torch.log 保持 autograd 图连通
        ln_ratio = torch.log(self.r_e / self.r_w) + self.skin
        ln_ratio = torch.clamp(ln_ratio, min=0.1)  # 防零 (实际范围[50,500]/0.1→ln∈[6.2,8.5], 不会触发)

        # v3: 直接用 k_frac (已含裂缝增效), 不再乘 f_frac
        k_SI = k_SI_override if k_SI_override is not None else self.k_frac_SI
        WI = (2.0 * math.pi * k_SI * h_well) / ln_ratio
        return WI

    def get_audit_dict(self) -> Dict[str, float]:
        """返回反演参数当前值（审计用）"""
        return {
            'k_frac_mD': self.k_frac_mD.item(),
            'k_frac_SI': self.k_frac_SI.item(),
            'k_eff_prior_mD': self._k_eff_prior,
            'f_frac_prior': self._f_frac_prior,
            'r_e_m': self.r_e.item(),
            'r_e_prior_m': self._r_e_prior,
            'r_w_m': self.r_w,
            'skin': self.skin,
        }


class PwfHiddenVariable(nn.Module):
    """
    p_wf(t, prod_hours) 隐变量参数化 — v16: 时间+工况双输入

    用小型 MLP 将 (t_norm, prod_hours_norm) 映射到 压差 δ ∈ [dp_min, dp_max] (MPa)，
    井底流压 p_wf = p_cell - δ。prod_hours_norm 编码了当日生产时长 (0=关井, 1=全天开井),
    使 δ 能感知"开关井"和"限产"工况，打破纯时间映射下的动态范围压缩。

    约束:
        - δ = softplus(raw) + dp_min, 再 clamp 到 [dp_min, dp_max]
        - 平滑: 训练时施加 |dδ/dt| 惩罚
    """

    def __init__(self, config: dict, well_id: str = 'default', device: str = 'cpu'):
        super().__init__()
        self.well_id = well_id
        self.device = device

        m5_cfg = config.get('m5_config', {})
        pwf_cfg = m5_cfg.get('pwf_network', {})
        guardrails = config.get('safety_guardrails', {}).get('limits', {})
        drawdown_range = guardrails.get('drawdown_MPa', [2.0, 65.0])
        self.dp_min = float(pwf_cfg.get('dp_min_MPa', pwf_cfg.get('min_drawdown_MPa', drawdown_range[0])))
        self.dp_max = float(pwf_cfg.get('dp_max_MPa', drawdown_range[1]))
        if self.dp_max <= self.dp_min + 1e-6:
            self.dp_max = self.dp_min + 1.0

        hidden = pwf_cfg.get('hidden', [32, 32])

        # v16: 输入维度 2 = (t_norm, prod_hours_norm)
        in_dim = 2
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t_norm: torch.Tensor,
                prod_hours_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            t_norm: (N, 1) or (N,) 归一化时间 [0, 1]
            prod_hours_norm: (N, 1) or (N,) or None, 归一化生产时间 [0, 1]
        Returns:
            delta: (N, 1) 压差 δ (MPa), p_wf = p_cell - delta
        """
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(-1)

        if prod_hours_norm is None:
            prod_hours_norm = torch.ones_like(t_norm)
        if prod_hours_norm.dim() == 1:
            prod_hours_norm = prod_hours_norm.unsqueeze(-1)

        net_input = torch.cat([t_norm, prod_hours_norm], dim=-1)  # (N, 2)
        raw = self.net(net_input)
        delta = torch.nn.functional.softplus(raw) + self.dp_min
        delta = torch.clamp(delta, min=self.dp_min, max=self.dp_max)
        return delta

    def compute_smoothness(self, t_norm: torch.Tensor,
                           prod_hours_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算 δ(t) 时间平滑度惩罚: mean(|dδ/dt|²)
        """
        t_norm = t_norm.requires_grad_(True)
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(-1)

        delta = self.forward(t_norm, prod_hours_norm)

        d_delta_dt = torch.autograd.grad(
            delta, t_norm,
            grad_outputs=torch.ones_like(delta),
            create_graph=True, retain_graph=True
        )[0]

        return torch.mean(d_delta_dt ** 2)


class GaussianSourceTerm(nn.Module):
    """
    有限支撑高斯核源项分配

    将井流量 q 通过高斯核分配到 PDE 配点:
        source(x, y) = Σ_j q_j · K(x - x_j, y - y_j) / Σ K(...)

    其中 K(dx, dy) = exp(-(dx² + dy²) / (2σ²))
    σ 由井周影响半径控制
    """

    def __init__(self, config: dict, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.logger = setup_logger('SourceTerm')

        m5_cfg = config.get('m5_config', {})
        src_cfg = m5_cfg.get('source_term', {})

        self.sigma = src_cfg.get('sigma_norm', 0.05)
        self.cutoff_factor = src_cfg.get('cutoff_factor', 3.0)

        self.logger.info(f"GaussianSourceTerm: σ={self.sigma}, cutoff={self.cutoff_factor}σ")

    def compute_source(self,
                       x_colloc: torch.Tensor,
                       well_xy_norm: torch.Tensor,
                       well_rates: torch.Tensor
                       ) -> torch.Tensor:
        """
        在配点处计算井源项强度

        Args:
            x_colloc: (N, 2) 配点归一化坐标 [x_n, y_n]
            well_xy_norm: (n_wells, 2) 井位归一化坐标
            well_rates: (n_wells,) 各井总流量 (正为注入，负为生产)

        Returns:
            source: (N, 1) 各配点处的源项强度
        """
        N = x_colloc.shape[0]
        n_wells = well_xy_norm.shape[0]

        source = torch.zeros(N, 1, device=x_colloc.device)
        cutoff_r2 = (self.cutoff_factor * self.sigma) ** 2

        for j in range(n_wells):
            dx = x_colloc[:, 0] - well_xy_norm[j, 0]
            dy = x_colloc[:, 1] - well_xy_norm[j, 1]
            r2 = dx ** 2 + dy ** 2

            mask = r2 < cutoff_r2
            kernel = torch.zeros(N, device=x_colloc.device)
            kernel[mask] = torch.exp(-r2[mask] / (2.0 * self.sigma ** 2))

            kernel_sum = kernel.sum() + 1e-12
            kernel_normalized = kernel / kernel_sum

            source[:, 0] += well_rates[j] * kernel_normalized

        return source


class WellModel(nn.Module):
    """
    完整井模型 v3：组合 Peaceman WI + PwfHiddenVariable + SourceTerm

    ===== v3 变更 =====
    - PeacemanWI 使用 k_frac (合并参数)
    - 删除 wi_factors (非物理自由度)
    - 其余逻辑不变

    对每口井:
        1. 从场网络获取井处 p_cell, Sw
        2. 从 PwfHiddenVariable 获取 p_wf(t)
        3. 计算 q_g = WI · λ_g · (p_cell - p_wf)
        4. 通过高斯核将 q 写入 PDE 源项
    """

    def __init__(self, config: dict, well_ids: List[str], device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.logger = setup_logger('WellModel')
        self.well_ids = well_ids
        self.config = config

        # 共享参数: Peaceman WI 计算器 (k_frac 全井共享)
        self.peaceman = PeacemanWI(config, device=device)

        # 井专属: 每口井的 δ(t) 网络 (压差, p_wf = p_cell - δ)
        self.pwf_nets = nn.ModuleDict()
        for wid in well_ids:
            self.pwf_nets[wid] = PwfHiddenVariable(config, well_id=wid, device=device)
        # 用于 get_all_pwf 的参考压力 (无 p_cell 时展示用)
        mk = config.get('mk_formation', {})
        p_avg = mk.get('avg_pressure_MPa', {})
        self.p_ref = p_avg.get('value', 76.0) if isinstance(p_avg, dict) else float(p_avg)

        # ===== P1-FIX: 删除 wi_factors =====
        # 旧版: self.wi_factors = nn.ParameterDict({wid: Parameter(1.0) for wid})
        # 问题: wi_factor=0.0854 击穿 clamp(0.1), 成为非物理逃生通道
        # 现在 k_frac 是唯一控制 WI 的参数, 无需额外修正因子

        # 源项分配器
        self.source_term = GaussianSourceTerm(config, device=device)

        # 最小压差 (MPa)：p_wf 上限 = p_cell - min_drawdown，减小关井流量地板
        pwf_cfg = config.get('m5_config', {}).get('pwf_network', {})
        self.min_drawdown_MPa = float(pwf_cfg.get('min_drawdown_MPa', 0.01))

        # 可微分 PVT 物性
        self.pvt = TorchPVT(config)

        # v3.14: 共享相渗模块 (由 trainer 注入 loss_fn.relperm, 保证 ng/nw 梯度一致)
        self.relperm = None  # placeholder, injected by M5Trainer

        self.logger.info(
            f"WellModel v3 初始化: {len(well_ids)} 口井 {well_ids}, "
            f"k_frac={self.peaceman.k_frac_mD.item():.2f} mD, "
            f"wi_factor=已删除, PVT=TorchPVT"
        )

    def compute_well_rate(self,
                          well_id: str,
                          p_cell: torch.Tensor,
                          sw_cell: torch.Tensor,
                          t_norm: torch.Tensor,
                          h_well: float,
                          bg_val: float = 0.002577,
                          krg_val: Optional[torch.Tensor] = None,
                          k_local_mD: Optional[torch.Tensor] = None,
                          prod_hours_norm: Optional[torch.Tensor] = None,
                          casing_norm: Optional[torch.Tensor] = None
                          ) -> Dict[str, torch.Tensor]:
        """
        计算单口井的产气量 (地面标准条件)

        v16 变更: pwf_net 接受 (t_norm, prod_hours_norm) 双输入
        casing_norm: 可选，预留供 pwf_net 扩展使用，当前未使用

        Peaceman 标准公式:
            q_g,surface = WI · (krg / μ_g) · (p_cell - p_wf) / Bg
        """
        # v16: 压差参数化 — δ(t, prod_hours), p_wf = p_cell - δ
        delta = self.pwf_nets[well_id](t_norm, prod_hours_norm)
        p_wf = p_cell - delta
        # 最小压差从 config 读取（默认 0.01 MPa），减小关井流量地板
        # v12: 低压保护下限保持 5 MPa（v13 曾放宽到 1 MPa）
        min_pwf = torch.tensor(5.0, device=p_wf.device, dtype=p_wf.dtype)
        min_dd = torch.tensor(self.min_drawdown_MPa, device=p_wf.device, dtype=p_wf.dtype)
        p_wf = torch.clamp(p_wf, min=min_pwf, max=p_cell - min_dd)

        # WI — 当 k_net 提供局部渗透率时用 k_local 替代 k_frac
        h_tensor = torch.tensor(h_well, dtype=torch.float32, device=p_cell.device)
        k_SI_override = None
        if k_local_mD is not None:
            k_SI_override = k_local_mD * 9.869233e-16
        WI = self.peaceman.compute_WI(h_tensor, k_SI_override=k_SI_override)
        # ===== P1-FIX: 不再乘以 wi_factor =====
        # 旧版: WI = WI_base * torch.clamp(self.wi_factors[well_id], 0.1, 10.0)

        # --- 气相相渗 krg(Sw) ---
        if krg_val is None:
            sw_clamped = torch.clamp(sw_cell, 0.0, 1.0)
            if self.relperm is not None:
                krg_val = self.relperm.krg(sw_clamped)
            else:
                krg_val = self._compute_krg_torch(sw_clamped)

        # --- 可微分 μ_g(p) ---
        mu_g = self.pvt.mu_g(p_cell)  # (N, 1) Pa·s

        # --- 可微分 Bg(p) ---
        bg_val_t = self.pvt.bg(p_cell)  # (N, 1)

        # --- 气相流度 λ_g = krg / μ_g ---
        lambda_g = krg_val / (mu_g + 1e-20)  # 1/(Pa·s)

        # 压差 (转 Pa)
        dp = (p_cell - p_wf) * 1e6  # MPa → Pa

        # --- Peaceman 产量公式 (单位审计) ---
        # WI: m³ (from compute_WI: 2π·k_SI·h/ln_ratio, k_SI=m², h=m)
        # λ_g: 1/(Pa·s),  dp: Pa  →  WI·λ_g·dp = m³/s
        # qg_surface = qg_reservoir/Bg [m³/s],  qg_m3d = qg_surface*86400 [m³/d]
        qg_reservoir = WI * lambda_g * dp
        qg_surface = qg_reservoir / (bg_val_t + 1e-20)

        # 转为 m³/d
        qg_m3d = qg_surface * 86400.0

        # --- v3.14: 水相产量 qw (t/d) ---
        qw_td = None
        if self.relperm is not None:
            krw_val = self.relperm.krw(torch.clamp(sw_cell, 0.0, 1.0))
            mu_w = self.pvt.mu_w    # Pa·s (地层水粘度, 来自 TorchPVT)
            Bw = self.pvt.Bw        # 水体积系数 (来自 TorchPVT)
            rho_w = self.pvt.rho_w  # kg/m³ (地层水密度, 来自 TorchPVT)
            lambda_w = krw_val / (mu_w + 1e-20)
            qw_reservoir = WI * lambda_w * dp   # m³/s
            qw_surface = qw_reservoir / Bw       # m³/s (地面)
            qw_td = qw_surface * 86400.0 * rho_w / 1000.0  # t/d

        result = {
            'qg': qg_m3d,
            'p_wf': p_wf,
            'WI': WI,
            'lambda_g': lambda_g,
        }
        if qw_td is not None:
            result['qw'] = qw_td
        return result

    def _compute_krg_torch(self, sw: torch.Tensor) -> torch.Tensor:
        """Corey-Brooks 气相相渗 krg(Sw) — 回退方法 (self.relperm 未注入时使用)"""
        self.logger.warning(
            "_compute_krg_torch fallback 被调用: self.relperm 未注入, "
            "使用硬编码 Corey 参数 (ng=1.0846). 请检查 trainer 是否正确注入了 relperm."
        )
        Swc = 0.26
        Sgr = 0.062
        krg_max = 0.675
        ng = 1.0846  # v3.14: 附表7 SY13 MK组 21点最小二乘拟合 (R²=0.9945)

        Se = torch.clamp((1.0 - sw - Sgr) / (1.0 - Swc - Sgr), 0.0, 1.0)
        return krg_max * Se ** ng + 1e-4

    def get_all_pwf(self, t_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取所有井的 p_wf(t)；v10 为 p_ref - delta（无 p_cell 时的参考展示）"""
        result = {}
        p_ref = torch.tensor(self.p_ref, device=t_norm.device, dtype=t_norm.dtype)
        for wid in self.well_ids:
            delta = self.pwf_nets[wid](t_norm)
            result[wid] = p_ref - delta
        return result

    def get_audit_dict(self) -> Dict[str, object]:
        """审计输出 (v3: 不再包含 wi_factor)"""
        audit = self.peaceman.get_audit_dict()
        # v3: 无 wi_factor
        return audit
