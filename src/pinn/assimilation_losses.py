"""
M5/M6 同化损失函数模块 (v2 — 增强版)
========================================
核心改进:
    1. PDE 升级为含 k(x,y) 变系数方程 + 两相守恒
    2. 井底流压转换: 油管压力不再直接约束 p_cell
    3. k_net 正则化: TV + Laplacian + 井点先验
    4. 改进的 qg 损失: 过滤关井期, 加权开井生产期

监督损失:
    L_qg:  产气量监督 (必须) — 只对开井期计算
    L_whp: 油管压力监督 → 转换为 p_wf 约束 (若有)
    L_qw:  产水量监督 (若有)

物理损失:
    L_PDE: 含 k(x,y) 变系数 PDE 残差 (两相版本)
    L_IC:  初始条件 (分区 Sw 初始化)
    L_BC:  边界条件 (含水层压力梯度)

正则/先验:
    L_smooth_pwf: p_wf(t) 平滑惩罚
    L_monotonic:  q_g vs p_wf 单调性约束
    L_prior_k:    k_eff 先验正则
    L_prior_f:    f_frac 先验正则
    L_sw_bounds:  Sw 有界性惩罚
    L_k_tv:       k(x,y) Total Variation 正则
    L_k_lap:      k(x,y) Laplacian 正则
    L_k_well:     k(x,y) 井点先验约束

总损失通过 ReLoBRaLo 或手动权重平衡。
"""

import os
import sys
import math
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("assimilation_losses 需要 PyTorch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger
from pinn.torch_physics import TorchPVT, TorchRelPerm


class AssimilationLoss:
    """
    M5/M6 完整同化损失函数 (v2)
    
    整合监督、物理、正则三大类损失，
    并提供统一的 total_loss 接口供 trainer 调用。
    """
    
    def __init__(self, config: dict, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.logger = setup_logger('AssimilationLoss')
        
        # --- 从 M4 继承的物理参数 ---
        mk = config.get('mk_formation', {})
        p_avg = mk.get('avg_pressure_MPa', {})
        self.p_init = p_avg.get('value', 76.0) if isinstance(p_avg, dict) else float(p_avg)
        self.p_boundary = self.p_init
        self.sw_init = 0.15  # v3.14: SY9 测井解释加权平均 (附表8: 13.8%×16.3+15.54%×32.1)/(48.4)
        
        # 气水界面
        gwc = config.get('coordinate_system', {}).get('gas_water_contact', {})
        self.gwc_z = gwc.get('value', -4385.0) if isinstance(gwc, dict) else -4385.0
        
        # PDE 缩放系数 (由 trainer 在初始化后注入)
        self.alpha_x = 1.0
        self.alpha_y = 1.0
        
        # 物理常数
        physics_priors = config.get('physics', {}).get('priors', {})
        phi_cfg = physics_priors.get('phi', {})
        self.phi = phi_cfg.get('value', 0.0216) if isinstance(phi_cfg, dict) else 0.0216
        
        # --- 可微分物性模块 (两相 PDE 核心) ---
        self.pvt = TorchPVT(config)
        self.relperm = TorchRelPerm(config)
        
        # 域参数 (由 trainer 注入)
        self.dx = 17400.0   # m
        self.dy = 11000.0   # m
        self.t_max_s = 1331 * 86400.0  # s
        self.h_mean = 90.0  # m (平均厚度, 由 trainer 注入)
        
        # --- 损失开关与权重 ---
        loss_cfg = config.get('loss', {})
        sup_cfg = loss_cfg.get('supervised', {})
        phys_cfg = loss_cfg.get('physics', {})
        
        self.sup_enable = sup_cfg.get('enable', True)
        self.phys_enable = phys_cfg.get('enable', True)
        
        # 监督权重
        sup_weights = sup_cfg.get('weights', {})
        self.w_qg = sup_weights.get('qg', 1.0)
        self.w_whp = sup_weights.get('whp', 0.3)
        self.w_qw = sup_weights.get('qw', 0.7)
        # v11: qg 混合损失权重（归一化后等权）
        self.w_qg_smape = sup_weights.get('qg_smape', 0.5)
        self.w_qg_log1p = sup_weights.get('qg_log1p', 0.5)
        # v11: log1p 归一化基准（由 trainer 注入 qg_rms，默认 430000 m³/d）
        self._log1p_norm = None  # 延迟注入
        
        # v13+: near-zero 参数从 config 读取
        m5_cfg = config.get('m5_config', {})
        nearzero_cfg = m5_cfg.get('qg_nearzero', {})
        self.nearzero_enable = nearzero_cfg.get('enable', True)
        self.nearzero_threshold = nearzero_cfg.get('threshold_m3d', 500.0)
        self.nearzero_qscale = nearzero_cfg.get('q_scale_m3d', 5e4)
        self.nearzero_min_points = nearzero_cfg.get('min_points', 8)
        shutin_cfg = m5_cfg.get('shutin_delta', {})
        self.shutin_target_mpa = float(shutin_cfg.get('target_MPa', 0.1))
        self.shutin_min_points = int(shutin_cfg.get('min_points', 20))
        
        # 约束开关
        constraints = config.get('physics', {}).get('constraints', {})
        self.monotonic_enable = constraints.get('monotonic_qg_vs_pwf', {}).get('enable', True)
        self.smooth_pwf_enable = constraints.get('smooth_pwf_time', {}).get('enable', True)
        self.sw_bounds_enable = constraints.get('sw_bounds', {}).get('enable', True)
        
        # 先验参数 (正则化)
        priors = config.get('physics', {}).get('priors', {})
        k_cfg = priors.get('k_eff_mD', {})
        self.k_eff_prior = k_cfg.get('value', 5.0) if isinstance(k_cfg, dict) else 5.0
        f_cfg = priors.get('frac_conductivity_factor', {})
        self.f_frac_prior = f_cfg.get('value', 10.0) if isinstance(f_cfg, dict) else 10.0
        # m5_config.inversion_prior: k/dp 先验权重与中心（可配并加强）
        inv_prior = m5_cfg.get('inversion_prior', {})
        self._inv_prior = inv_prior  # v3.14: 保存供 loss_prior_params 读取 corey_weight
        self.prior_k_weight = float(inv_prior.get('k_weight', 0.1))  # v3.5: 1.0→0.1, 1.0太强锚死k_frac=3.08, 0.1仍引导但不压制寻优
        self.prior_dp_center_MPa = float(inv_prior.get('dp_center_MPa', 13.3))  # v3.6: 12→13.3, 试油实测 WHP=57.93 BHP=71.23 Δp=13.3
        self.prior_dp_weight = float(inv_prior.get('dp_weight', 0.5))  # v3.6: 0.01→0.5, dp漂移到17→驱动压差坍缩, 需强约束
        
        self._loss_qg_shape_logged = False
        self._loss_whp_shape_logged = False  # 仅首次打印 shape 诊断
        self.logger.info(
            f"AssimilationLoss v2 初始化: p_init={self.p_init} MPa, "
            f"sw_init={self.sw_init}, monotonic={self.monotonic_enable}, "
            f"smooth_pwf={self.smooth_pwf_enable}"
        )
    
    # ================================================================== #
    #                          监督损失
    # ================================================================== #
    
    def loss_qg(self,
                qg_pred: torch.Tensor,
                qg_obs: torch.Tensor,
                scale: float = 1.0,
                producing_mask: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        产气量监督损失 (v12，见 docs/v12_changes_summary.md)
        分段加权：低产(<5万 m³/d)=2.0、中产(5~20万)=1.5、高产=6.0；
        同时强调低产与超高产两端，抑制“中段正确、两端塌缩”的动态范围压缩。
        """
        if qg_obs.dim() == 1:
            qg_obs = qg_obs.unsqueeze(-1)
        
        # 过滤关井期（> 1 m³/d 视为开井）
        if producing_mask is not None:
            if producing_mask.dim() == 1:
                producing_mask = producing_mask.unsqueeze(-1)
            mask = producing_mask
            n_producing = producing_mask.sum().clamp(min=1)
        else:
            mask = (qg_obs.abs() > 1.0).float()
            n_producing = mask.sum().clamp(min=1)

        if valid_mask is not None:
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(-1)
            mask = mask * (valid_mask > 0.5).float()
            n_producing = mask.sum().clamp(min=1)
        
        # 分段 低(<5万)/中(5~20万)/高(≥20万)，权重 2.0/4.0/6.0
        # v3.8: 中产段 1.5→4.0 (中产段 APE 贡献最大: 5万误差/10万观测=50% MAPE)
        LOW, MID = 5e4, 20e4  # m³/d
        seg_w = torch.ones_like(qg_obs, device=qg_obs.device, dtype=qg_obs.dtype)
        seg_w = torch.where(qg_obs < LOW, torch.full_like(seg_w, 2.0), seg_w)
        seg_w = torch.where((qg_obs >= LOW) & (qg_obs < MID), torch.full_like(seg_w, 4.0), seg_w)
        seg_w = torch.where(qg_obs >= MID, torch.full_like(seg_w, 6.0), seg_w)
        seg_w = seg_w * mask
        sum_w = seg_w.sum().clamp(min=1.0)
        
        epsilon = 1e3  # m³/d 防止除零
        abs_error = torch.abs(qg_pred - qg_obs) * mask
        denominator = (torch.abs(qg_pred) + torch.abs(qg_obs) + epsilon) * mask
        safe_denom = denominator + (1.0 - mask) * 1.0
        smape = abs_error / safe_denom
        
        L_smape = (smape ** 2 * seg_w).sum() / sum_w
        
        log1p_err = torch.log1p(abs_error + (1.0 - mask) * 1.0)
        if self._log1p_norm is not None:
            norm_const = self._log1p_norm
        else:
            with torch.no_grad():
                qg_rms = torch.sqrt(torch.sum(qg_obs ** 2 * mask) / n_producing + 1.0)
                norm_const = float(torch.log1p(qg_rms).item() ** 2)
                norm_const = max(norm_const, 1.0)
        L_log1p = ((log1p_err ** 2 * seg_w).sum() / sum_w) / norm_const
        
        # v15: 高产段绝对误差 MSE
        HIGH_ABS_THRESHOLD = 2e5  # v3.8: 3e5→2e5, 覆盖更多高产点
        Q_REF = 5e5  # 归一化基准
        high_abs_mask = (qg_obs >= HIGH_ABS_THRESHOLD).float() * mask
        n_high_abs = high_abs_mask.sum().clamp(min=1)
        L_mse_high = (((qg_pred - qg_obs) / Q_REF) ** 2 * high_abs_mask).sum() / n_high_abs
        w_mse_high = 0.80  # v3.10: 回退到v3.8值 (v3.9的1.00导致k_frac崩塌)

        # v16: 高产段偏置惩罚 — 专门消除系统性低估
        mean_bias_high = ((qg_pred - qg_obs) * high_abs_mask).sum() / (n_high_abs * Q_REF)
        L_bias_high = mean_bias_high ** 2
        w_bias_high = 0.60  # v3.10: 回退到v3.8值 (v3.9的0.80过激)

        # v3.6: 全局 MSE — 不限于高产段, 直接优化整体 R²
        L_mse_global = (((qg_pred - qg_obs) / Q_REF) ** 2 * mask).sum() / n_producing
        w_mse_global = 0.60  # v3.10: 回退到v3.8值

        # v3.8: 直接 MAPE 损失 — SMAPE 对低估惩罚不足, 需直接优化 MAPE
        MAPE_THRESHOLD = 1e4  # m³/d, 避免近零除法
        mape_mask = (qg_obs > MAPE_THRESHOLD).float() * mask
        n_mape = mape_mask.sum().clamp(min=1)
        L_mape = (torch.abs(qg_pred - qg_obs) / qg_obs.clamp(min=MAPE_THRESHOLD) * mape_mask).sum() / n_mape
        w_mape = 0.40

        # v3.17: 超高产峰值强化 — 聚焦>350k最高产段, 用相对误差²
        # 与 HIGH_ABS_THRESHOLD=200k 互补: 200k覆盖范围广, 350k精准打击峰值低估
        PEAK_THRESHOLD = 3.5e5  # m³/d
        peak_mask = (qg_obs >= PEAK_THRESHOLD).float() * mask
        n_peak = peak_mask.sum().clamp(min=1)
        rel_err_peak = torch.abs(qg_pred - qg_obs) / qg_obs.clamp(min=1e4) * peak_mask
        L_peak = (rel_err_peak ** 2).sum() / n_peak
        w_peak = 0.80  # 与w_mse_high同量级, 避免过强导致k_frac不稳定

        if not self._loss_qg_shape_logged:
            self.logger.info(
                f"  [loss_qg] v3.17 分段加权: 低(<5万)=2, 中(5~20万)=4, 高=6; "
                f"w_smape={self.w_qg_smape}, w_log1p={self.w_qg_log1p}, "
                f"w_mse_high={w_mse_high}, w_bias={w_bias_high}, "
                f"w_global={w_mse_global}, w_mape={w_mape}, w_peak={w_peak}, "
                f"n_prod={n_producing.item():.0f}, n_high={n_high_abs.item():.0f}, "
                f"n_peak={n_peak.item():.0f}"
            )
            self._loss_qg_shape_logged = True
        
        loss = (self.w_qg_smape * L_smape + self.w_qg_log1p * L_log1p
                + w_mse_high * L_mse_high + w_bias_high * L_bias_high
                + w_mse_global * L_mse_global
                + w_mape * L_mape
                + w_peak * L_peak)  # v3.17: +L_peak超高产峰值强化
        return loss
    
    def loss_qg_nearzero(self,
                         qg_pred: torch.Tensor,
                         qg_obs: torch.Tensor,
                         threshold: float = 500.0,
                         q_scale: float = 5e4,
                         valid_mask: Optional[torch.Tensor] = None,
                         min_points: int = 8
                         ) -> torch.Tensor:
        """
        v13+: 仅对真实观测的关井点监督（qg_obs <= threshold 且 valid_mask==1）。
        对 qg_obs ∈ [0, threshold] 做归一化 MSE: ((qg_pred - qg_obs) / q_scale)^2。
        threshold 默认 500 m³/d（真关井/极低产）；min_points 默认 8（至少要有 8 个真实关井点才监督）。
        """
        if qg_obs.dim() == 1:
            qg_obs = qg_obs.unsqueeze(-1)
        mask = (qg_obs >= 0) & (qg_obs <= threshold) & torch.isfinite(qg_pred) & torch.isfinite(qg_obs)
        # v13+: 乘上 valid_mask，只监督真实观测点
        if valid_mask is not None:
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(-1)
            mask = mask & (valid_mask > 0.5)
        n = mask.sum()
        if n < min_points:
            return torch.tensor(0.0, device=qg_obs.device, dtype=qg_obs.dtype)
        err = (qg_pred - qg_obs).float() * mask.float()
        scale = max(float(q_scale), 1.0)
        return ((err / scale) ** 2).sum() / n.clamp(min=1)
    
    def loss_whp(self,
                 p_wf_pred: torch.Tensor,
                 whp_obs: torch.Tensor,
                 dp_wellbore: torch.Tensor
                 ) -> torch.Tensor:
        """
        油管压力监督损失 (v2: 转换为 p_wf 约束)
        
        不直接用 WHP 约束 p_cell，而是:
        p_wf_pred ≈ WHP_obs + Δp_wellbore
        
        Args:
            p_wf_pred: (N, 1) 反演的井底流压 (MPa)
            whp_obs: (N,) 或 (N, 1) 观测油管压力 (MPa)
            dp_wellbore: scalar 井筒压差 (MPa, 可学习)
        """
        # ★ 统一为 (N, 1)，避免 (N,) - (N,1) 广播成 (N,N)
        if p_wf_pred.dim() == 1:
            p_wf_pred = p_wf_pred.unsqueeze(-1)
        if whp_obs.dim() == 1:
            whp_obs = whp_obs.unsqueeze(-1)
        
        # 估算的井底流压 = 油管压力 + 井筒压差
        p_wf_from_whp = whp_obs + dp_wellbore
        delta = p_wf_pred - p_wf_from_whp
        if not self._loss_whp_shape_logged:
            self.logger.info(
                f"  [loss_whp] p_wf_pred={tuple(p_wf_pred.shape)}, whp_obs={tuple(whp_obs.shape)}, "
                f"delta={tuple(delta.shape)}"
            )
            self._loss_whp_shape_logged = True
        return torch.mean(delta ** 2)
    
    def loss_qw(self,
                qw_pred: torch.Tensor,
                qw_obs: torch.Tensor,
                scale: float = 1.0
                ) -> torch.Tensor:
        """产水量监督损失"""
        if qw_obs.dim() == 1:
            qw_obs = qw_obs.unsqueeze(-1)
        eps = scale + 1e-6
        return torch.mean(((qw_pred - qw_obs) / eps) ** 2)
    
    # ================================================================== #
    #                          物理损失
    # ================================================================== #
    
    def loss_ic(self, model, x_ic: torch.Tensor) -> torch.Tensor:
        """
        初始条件损失 L_IC (v2: 支持分区 Sw 初始化)
        
        气区 (大部分域): Sw ≈ 0.15 (SY9 测井, 附表8)
        过渡带/水区 (接近气水界面): Sw 更高
        """
        p_pred, sw_pred = model(x_ic)
        
        # 压力初始值 (均匀场)
        p_target = torch.full_like(p_pred, self.p_init)
        loss_p = torch.mean((p_pred - p_target) ** 2)
        
        # 饱和度初始值 — 全域用束缚水 (简化)
        # TODO: 后续可按 z 坐标分区
        sw_target = torch.full_like(sw_pred, self.sw_init)
        loss_sw = torch.mean((sw_pred - sw_target) ** 2)
        
        return loss_p + loss_sw
    
    def loss_bc(self, model, x_bc: torch.Tensor) -> torch.Tensor:
        """
        边界条件损失 L_BC
        外边界定压 (含水层准稳态)
        """
        p_pred, _ = model(x_bc)
        p_target = torch.full_like(p_pred, self.p_boundary)
        return torch.mean((p_pred - p_target) ** 2)
    
    def loss_pde(self,
                 model,
                 x_pde: torch.Tensor,
                 h_grad: Optional[Dict[str, torch.Tensor]] = None,
                 source_g: Optional[torch.Tensor] = None,
                 source_w: Optional[torch.Tensor] = None,
                 k_net=None,
                 k_eff_mD_tensor: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """
        完整两相守恒 PDE 残差 (一等奖版 — 梯度贯通)
        
        ===== 气相质量守恒 =====
        φ·h · [-ρ_g · ∂Sw/∂t + (1-Sw) · ∂ρ_g/∂t]
            = ∇·(k · krg · h / μ_g · ∇p) + q_g
        
        ===== 水相质量守恒 =====
        φ·h · ρ_w · ∂Sw/∂t
            = ∇·(k · krw · h / μ_w · ∇p) + q_w
        
        在归一化坐标下:
            ∂/∂x → (2/Δx) ∂/∂x_n
            ∂/∂t → (1/t_max) ∂/∂t_n
        
        两个残差各自求 MSE, 以 1:λ_sw 加权求和。
        
        梯度贯通设计:
            k_net 和 k_eff_mD_tensor 在 loss_pde 内部用 xyt 计算 k_field，
            确保 autograd.grad(k_field, xyt) 在同一张计算图中。
        
        Args:
            model: PINN 模型 (需有 forward_with_grad)
            x_pde: (N, 3) PDE 配点 [x_n, y_n, t_n]
            h_grad: {'gx': (N,1), 'gy': (N,1)} log-thickness 梯度
            source_g: (N, 1) 气相井源项 (可选)
            source_w: (N, 1) 水相井源项 (可选, 通常为 0)
            k_net: PermeabilityNet 子网络引用 (在内部用 xyt 计算 k_field)
            k_eff_mD_tensor: 可训练标量张量 (无 k_net 时使用)
        """
        grads = model.forward_with_grad(x_pde)
        
        p = grads['p']           # (N,1) MPa
        sw = grads['sw']         # (N,1) 含水饱和度
        dp_dx = grads['dp_dx']   # ∂p/∂x_n
        dp_dy = grads['dp_dy']   # ∂p/∂y_n
        dp_dt = grads['dp_dt']   # ∂p/∂t_n
        dsw_dx = grads['dsw_dx']
        dsw_dy = grads['dsw_dy']
        dsw_dt = grads['dsw_dt'] # ∂Sw/∂t_n
        xyt = grads['xyt']
        
        # ============================================
        #  物性计算 (全部可微分, 支持 autograd)
        # ============================================
        rho_g = self.pvt.rho_g(p)       # (N,1) kg/m³
        mu_g  = self.pvt.mu_g(p)        # (N,1) Pa·s
        Bg    = self.pvt.bg(p)          # (N,1) m³/m³
        
        krg   = self.relperm.krg(sw)    # (N,1)
        krw   = self.relperm.krw(sw)    # (N,1)
        
        rho_w = self.pvt.rho_w          # scalar kg/m³
        mu_w  = self.pvt.mu_w           # scalar Pa·s
        phi   = self.phi                # scalar
        
        # ∂ρ_g/∂t = (dρ_g/dp) · (∂p/∂t)
        # dρ_g/dp 通过链式法则: ρ_g = f(p), 已由 autograd 图自动传递
        # 但这里我们需要显式值, 用解析近似:
        # ρ_g = p·M/(Z·R·T), dρ_g/dp ≈ ρ_g · (1/p - (1/Z)(dZ/dp))
        # = ρ_g · cg(p)
        cg_val = self.pvt.cg(p)  # (N,1) 1/MPa
        
        # 归一化坐标导数 → 物理坐标导数 的缩放因子
        sx = 2.0 / self.dx        # ∂/∂x = sx · ∂/∂x_n
        sy = 2.0 / self.dy
        st = 1.0 / self.t_max_s   # ∂/∂t = st · ∂/∂t_n (秒)
        
        # 物理坐标梯度
        dp_dx_phys = sx * dp_dx   # MPa/m
        dp_dy_phys = sy * dp_dy
        dp_dt_phys = st * dp_dt   # MPa/s
        dsw_dt_phys = st * dsw_dt # 1/s
        
        # 二阶导 (归一化坐标)
        d2p_dx2 = torch.autograd.grad(
            dp_dx, xyt,
            grad_outputs=torch.ones_like(dp_dx),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]
        
        d2p_dy2 = torch.autograd.grad(
            dp_dy, xyt,
            grad_outputs=torch.ones_like(dp_dy),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]
        
        # 物理坐标二阶导
        d2p_dx2_phys = sx * sx * d2p_dx2  # MPa/m²
        d2p_dy2_phys = sy * sy * d2p_dy2
        
        # ============================================
        #  渗透率场 k(x,y) — 在 loss_pde 内部计算
        #  [护栏1] 用 loss_pde 自己的 xyt 调用 k_net，
        #          确保 autograd.grad(k_field, xyt) 在同一张计算图
        # ============================================
        if k_net is not None:
            # 在内部用 xyt 的空间分量调用 k_net（同一个 Tensor 对象！）
            xy_for_k = xyt[:, :2]   # 切片 = 视图，不断图
            k_field = k_net.get_k_mD(xy_for_k)  # (N,1) mD, 在计算图中
            k_SI = k_field * 9.869233e-16  # mD → m²
            
            # ∇k 梯度 — allow_unused=False: 图断了直接报错，不静默置零
            dk_grad = torch.autograd.grad(
                k_field, xyt,
                grad_outputs=torch.ones_like(k_field),
                create_graph=True, retain_graph=True,
                allow_unused=False  # [护栏1] 显式报错而非静默 None
            )[0]
            # [护栏2] 一阶导用 sx (不是 sx²)
            dk_dx_phys = sx * dk_grad[:, 0:1] * 9.869233e-16  # m²/m
            dk_dy_phys = sy * dk_grad[:, 1:2] * 9.869233e-16
            
        elif k_eff_mD_tensor is not None:
            # 无 k_net: 用可训练标量 k_eff_mD（仍在计算图中！）
            k_field = k_eff_mD_tensor.expand(p.shape[0], 1)  # 广播，不断图
            k_SI = k_field * 9.869233e-16
            # 标量 k 的空间梯度 = 0（均匀场，物理正确）
            dk_dx_phys = torch.zeros_like(dp_dx)
            dk_dy_phys = torch.zeros_like(dp_dy)
            
        else:
            # Fallback: 不应到达（trainer 已保证传入其一）
            import warnings
            warnings.warn("loss_pde: k_net 和 k_eff_mD_tensor 均为 None! 梯度断开!")
            k_mD_val = self.k_eff_prior  # 至少不是硬编码 5.0
            k_SI = torch.full_like(p, k_mD_val * 9.869233e-16)
            dk_dx_phys = torch.zeros_like(dp_dx)
            dk_dy_phys = torch.zeros_like(dp_dy)
        
        h = self.h_mean  # m (后续可替换为配点处 h(x,y))
        
        # ============================================
        #  气相方程残差 R_g
        # ============================================
        # 左端: φ·h · [-ρ_g · ∂Sw/∂t + (1-Sw) · (dρ_g/dp) · (∂p/∂t)]
        #      = φ·h · [-ρ_g · ∂Sw/∂t + (1-Sw) · ρ_g · cg · (∂p/∂t)]
        # 单位: kg/m² · 1/s = kg/(m²·s)
        # 
        # 注意: cg 单位 1/MPa, dp/dt 单位 MPa/s
        #       ρ_g·cg·dp/dt → kg/m³ · 1/MPa · MPa/s = kg/(m³·s) ✓
        
        Sg = 1.0 - sw
        accumulation_g = phi * h * (
            -rho_g * dsw_dt_phys
            + Sg * rho_g * cg_val * dp_dt_phys
        )
        
        # 右端 flux (质量守恒口径): ∇·(k·krg·h·ρ_g / μ_g · ∇p)
        # 展开: = (k·krg·h·ρ_g/μ_g) · ∇²p
        #       + ∇(k·krg·h·ρ_g/μ_g) · ∇p
        #
        # ∇(k·krg·h·ρ_g/μ_g) 主要项:
        # ≈ (krg·h·ρ_g/μ_g)·∇k + (k·h·ρ_g/μ_g)·(dkrg/dSw)·∇Sw
        #   + k·krg·ρ_g/(μ_g)·∇h  (已通过 h_grad 处理)
        #   + (k·krg·h/μ_g)·∇ρ_g  (密度梯度, 含在 T_g 定义中)
        #
        # 与 M4 losses.py 保持一致的质量守恒口径:
        
        # p 单位转换: ∇p 是 MPa/m, k·krg/μ_g·∇p 要得到 m/s (Darcy velocity)
        # Darcy: v = -k·kr/(μ) · ∇p  → k[m²] · ∇p[Pa/m] / μ[Pa·s] = [m/s]
        # 所以 ∇p 需要转为 Pa/m: ×1e6
        dp_dx_Pa = dp_dx_phys * 1e6  # Pa/m
        dp_dy_Pa = dp_dy_phys * 1e6
        d2p_dx2_Pa = d2p_dx2_phys * 1e6  # Pa/m²
        d2p_dy2_Pa = d2p_dy2_phys * 1e6
        
        # 流动系数 (气相, 质量守恒口径 — 含 ρ_g)
        T_g = k_SI * krg * h * rho_g / (mu_g + 1e-20)  # kg·m/(Pa·s)
        
        # 主扩散项: T_g · ∇²p
        flux_g_diffusion = T_g * (d2p_dx2_Pa + d2p_dy2_Pa)
        
        # ∇k 项: (krg·h·ρ_g/μ_g) · ∇k · ∇p
        flux_g_dk = (krg * h * rho_g / (mu_g + 1e-20)) * (
            dk_dx_phys * dp_dx_Pa + dk_dy_phys * dp_dy_Pa
        )
        
        # ∇krg 项: (k·h·ρ_g/μ_g) · (dkrg/dSw) · (∇Sw · ∇p)
        dkrg_dsw = self.relperm.dkrg_dSw(sw)  # (N,1)
        dsw_dx_phys = sx * dsw_dx  # 1/m
        dsw_dy_phys = sy * dsw_dy
        
        flux_g_dkr = (k_SI * h * rho_g / (mu_g + 1e-20)) * dkrg_dsw * (
            dsw_dx_phys * dp_dx_Pa + dsw_dy_phys * dp_dy_Pa
        )
        
        # 厚度修正: T_g · (1/h)(∇h) · ∇p
        # gx = (1/h)(∂h/∂x_n) 是归一化坐标下的对数厚度梯度
        # 物理坐标转换: (1/h)(∂h/∂x) = gx · sx,  其中 sx = 2/Δx  [1/m]
        # dp_dx_Pa 已是物理坐标梯度 [Pa/m]
        # 量纲: T_g [kg·m/(Pa·s)] × (gx·sx) [1/m] × dp_dx_Pa [Pa/m] = [kg/(m²·s)] ✓
        flux_g_h = torch.zeros_like(p)
        if h_grad is not None:
            gx = h_grad['gx']  # (N,1) = (1/h)(∂h/∂x_n)
            gy = h_grad['gy']
            flux_g_h = T_g * (
                gx * sx * dp_dx_Pa + gy * sy * dp_dy_Pa
            )
        
        # v3.20-FIX: 删除旧版 flux_g_compress 项
        # 旧版用体积口径 T_g (无 ρ_g) + 单独压缩修正, 量纲不自洽
        # 现已改为质量口径 T_g (含 ρ_g), ∇·(ρ_g·v·h) 的密度梯度效应
        # 自然包含在 autograd 对 T_g 的求导链中, 无需额外修正项
        
        # 气相源项 (kg/(m²·s)) — 产气率转换
        source_g_val = torch.zeros_like(p)
        if source_g is not None:
            source_g_val = source_g
        
        # 气相残差: accumulation = flux + source
        # 量纲: kg/(m²·s) (质量守恒口径, 与 M4 losses.py 一致)
        R_gas = accumulation_g - (flux_g_diffusion + flux_g_dk + flux_g_dkr + flux_g_h) \
            - source_g_val
        
        # 归一化: 除以特征值避免量级问题
        # 特征累积: φ·h·ρ_g_ref/t_max ≈ 0.02·90·200 / 1.15e8 ≈ 3e-6
        scale_g = phi * h * 200.0 * st + 1e-12
        R_gas_norm = R_gas / scale_g
        R_gas_norm = torch.clamp(R_gas_norm, -50.0, 50.0)  # v3.20: 残差截断防梯度爆炸 (同 M4)
        
        # ============================================
        #  水相方程残差 R_w
        # ============================================
        # 左端: φ·h·ρ_w · ∂Sw/∂t
        accumulation_w = phi * h * rho_w * dsw_dt_phys  # kg/(m²·s)
        
        # 右端 flux (质量守恒口径): ∇·(k·krw·h·ρ_w / μ_w · ∇p)
        T_w = k_SI * krw * h * rho_w / (mu_w + 1e-20)  # kg·m/(Pa·s)
        
        flux_w_diffusion = T_w * (d2p_dx2_Pa + d2p_dy2_Pa)
        
        flux_w_dk = (krw * h * rho_w / (mu_w + 1e-20)) * (
            dk_dx_phys * dp_dx_Pa + dk_dy_phys * dp_dy_Pa
        )
        
        dkrw_dsw = self.relperm.dkrw_dSw(sw)
        flux_w_dkr = (k_SI * h * rho_w / (mu_w + 1e-20)) * dkrw_dsw * (
            dsw_dx_phys * dp_dx_Pa + dsw_dy_phys * dp_dy_Pa
        )
        
        # 水相厚度修正 (同气相, 质量守恒口径 kg/(m²·s))
        flux_w_h = torch.zeros_like(p)
        if h_grad is not None:
            flux_w_h = T_w * (
                gx * sx * dp_dx_Pa + gy * sy * dp_dy_Pa
            )
        
        source_w_val = torch.zeros_like(p)
        if source_w is not None:
            source_w_val = source_w
        
        R_water = accumulation_w - (flux_w_diffusion + flux_w_dk + flux_w_dkr + flux_w_h) - source_w_val
        
        scale_w = phi * h * rho_w * st + 1e-12
        R_water_norm = R_water / scale_w
        R_water_norm = torch.clamp(R_water_norm, -50.0, 50.0)  # v3.20: 残差截断防梯度爆炸 (同 M4)
        
        # ============================================
        #  总 PDE 损失: 气相 + λ_sw · 水相
        # ============================================
        loss_gas = torch.mean(R_gas_norm ** 2)
        loss_water = torch.mean(R_water_norm ** 2)
        
        # 水相方程权重 (水相方程提供 Sw 演化约束)
        lambda_sw = 0.5
        
        return loss_gas + lambda_sw * loss_water
    
    # ================================================================== #
    #                     正则/先验/约束损失
    # ================================================================== #
    
    def loss_smooth_qg(self,
                       qg_pred: torch.Tensor,
                       qg_obs: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """
        qg_pred 时间光滑惩罚 (v3.9)
        
        只在连续开井段内惩罚相邻时间步差分 |Δqg|²，
        跨关井段不惩罚（避免模糊开/关井跳变）。
        """
        if qg_pred.dim() == 2:
            qg_pred = qg_pred.squeeze(-1)
        if qg_obs.dim() == 2:
            qg_obs = qg_obs.squeeze(-1)
        
        n = qg_pred.shape[0]
        if n < 3:
            return torch.tensor(0.0, device=qg_pred.device)
        
        # 识别开井点
        SHUTIN_THRESHOLD = 1.0  # m³/d
        producing = (qg_obs.abs() > SHUTIN_THRESHOLD).float()
        if valid_mask is not None:
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.squeeze(-1)
            producing = producing * (valid_mask > 0.5).float()
        
        # 只在连续开井段计算 (两个相邻点都开井)
        both_producing = producing[:-1] * producing[1:]
        n_pairs = both_producing.sum().clamp(min=1)
        
        # 归一化差分惩罚
        Q_REF = 5e5  # m³/d
        dqg = (qg_pred[1:] - qg_pred[:-1]) / Q_REF
        L_smooth = (dqg ** 2 * both_producing).sum() / n_pairs
        
        return L_smooth
    
    def loss_smooth_pwf(self,
                        pwf_net,
                        t_norm: torch.Tensor,
                        prod_hours_norm=None
                        ) -> torch.Tensor:
        """p_wf(t) 平滑惩罚: mean(|dp_wf/dt|²)"""
        return pwf_net.compute_smoothness(t_norm, prod_hours_norm)
    
    def loss_monotonic_qg_pwf(self,
                              qg_pred: torch.Tensor,
                              p_wf: torch.Tensor
                              ) -> torch.Tensor:
        """q_g vs p_wf 单调性约束"""
        if len(qg_pred) < 2:
            return torch.tensor(0.0, device=qg_pred.device)
        
        dqg = qg_pred[1:] - qg_pred[:-1]
        dpwf = p_wf[1:] - p_wf[:-1]
        
        anomaly = torch.relu(dqg) * torch.relu(dpwf)
        return torch.mean(anomaly)
    
    def loss_shutin_delta(self,
                          p_cell: torch.Tensor,
                          p_wf: torch.Tensor,
                          qg_obs: torch.Tensor,
                          valid_mask: Optional[torch.Tensor] = None,
                          target_MPa: float = 0.1,
                          min_points: int = 20) -> torch.Tensor:
        """
        P0-1: 关井压差损失 — 关井时 p_wf 应接近 p_cell，压差趋近 0。
        对 qg_obs<=1 且 valid 的点，罚 (p_cell - p_wf)，目标 0~target_MPa。
        直击「关井仍维持大压差、qg 下不去」的根因。
        """
        if p_cell.dim() == 1:
            p_cell = p_cell.unsqueeze(-1)
        if p_wf.dim() == 1:
            p_wf = p_wf.unsqueeze(-1)
        if qg_obs.dim() == 1:
            qg_obs = qg_obs.unsqueeze(-1)
        shutin = (qg_obs <= 1.0) & torch.isfinite(p_cell) & torch.isfinite(p_wf)
        if valid_mask is not None:
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(-1)
            shutin = shutin & (valid_mask > 0.5)
        n = shutin.sum()
        if n < max(int(min_points), 1):
            return torch.tensor(0.0, device=p_cell.device, dtype=p_cell.dtype)
        delta = (p_cell - p_wf).float() * shutin.float()
        # 目标 0，允许 target_MPa 容差：只罚超过容差部分
        over = torch.clamp(delta - target_MPa, min=0.0)
        return (over ** 2).sum() / n.clamp(min=1)
    
    def pwf_physical_constraint_loss(self,
                                     p_cell: torch.Tensor,
                                     p_wf: torch.Tensor) -> torch.Tensor:
        """
        井底流压物理约束损失 (v3.2新增)

        对于生产井: p_wf 必须 < p_cell - margin
        惩罚违反约束的情况

        Args:
            p_cell: 地层压力 (MPa), shape (N, 1)
            p_wf: 井底流压 (MPa), shape (N, 1)

        Returns:
            constraint_loss: 标量张量
        """
        margin = 2.0  # MPa (要求至少 2 MPa 的压差)
        violation = p_wf - p_cell + margin
        loss = torch.mean(torch.relu(violation) ** 2)
        return loss
    
    def loss_prior_params(self,
                          inversion_params: Dict[str, torch.Tensor]
                          ) -> torch.Tensor:
        """
        反演参数先验正则化 (对数空间, 软约束)
        
        使用 log(k / k_prior)² 替代 ((k - k_prior) / k_prior)²:
        - 对数空间对跨数量级变化更宽容 (k 可能从 5 mD 变到 50 mD)
        - 系数从 0.01 降至 0.001, 避免在训练前期 qg loss 尚大时
          先验约束成为主导力把参数"拉回"先验值
        """
        loss = torch.tensor(0.0, device=self.device)
        
        # v3: k_frac = k_eff × f_frac 合并参数, 先验 = k_eff_prior × f_frac_prior
        k_frac_prior = self.k_eff_prior * self.f_frac_prior  # config: k_eff × f_frac
        if 'k_frac_mD' in inversion_params:
            k = inversion_params['k_frac_mD']
            loss = loss + self.prior_k_weight * (torch.log(k / (k_frac_prior + 1e-12))) ** 2
        elif 'k_eff_mD' in inversion_params:
            k = inversion_params['k_eff_mD']
            loss = loss + self.prior_k_weight * (torch.log(k / (k_frac_prior + 1e-12))) ** 2
        
        # dp_wellbore 先验: 中心与权重来自 m5_config.inversion_prior（默认 12 MPa, 权重 0.01）
        if 'dp_wellbore' in inversion_params:
            dp = inversion_params['dp_wellbore']
            scale = max(self.prior_dp_center_MPa, 1.0)
            loss = loss + self.prior_dp_weight * ((dp - self.prior_dp_center_MPa) / scale) ** 2
        
        # v3.14: Corey 指数先验 — 防止 ng/nw 偏离 SY13 拟合值过远
        # log 空间正则, 权重 0.05 (温和: 允许 ±30% 微调, 防止 weight_decay 拉崩)
        rp = self.relperm
        if hasattr(rp, '_ng_log'):
            corey_w = float(self._inv_prior.get('corey_weight', 0.1))
            loss = loss + corey_w * (torch.log(rp.ng / rp.ng_prior)) ** 2
            loss = loss + corey_w * (torch.log(rp.nw / rp.nw_prior)) ** 2
        
        return loss
    
    def loss_sw_bounds(self,
                       sw: torch.Tensor
                       ) -> torch.Tensor:
        """
        Sw 有界性惩罚 v5 (最终版: 对称屏障, 参数有物理依据)
        
        所有阈值的依据:
          · gas_floor = sw_init−0.06 = 0.09 (SY9 初始 Sw≈0.15, 附表8)
          · gas_ceiling = 0.65 (相渗等渗点附近, 附表7 krg≈krw 在 Sw≈80%)
          · hard_lower = 0.05 (tanh 下界 + margin)
          · hard_upper = 0.85 (tanh 上界 − margin)
        
        设计原则:
          · 对称: 上下等强度, 不造成单向漂移
          · 初始值 Sw=0.15 处零惩罚
          · 不含 anchor 项 (anchor 会制造持续单向推力)
        """
        # 基于 M3 相渗端点参数 (附表7)
        gas_floor = self.sw_init - 0.06     # = 0.09 (v3.14: sw_init=0.15)
        gas_ceiling = 0.65                  # 相渗等渗区附近
        hard_lower = 0.05                   # tanh 下界 + margin
        hard_upper = 0.85                   # tanh 上界 − margin
        
        # 对称双侧物理屏障
        penalty_gas_floor = torch.mean(
            torch.relu(gas_floor - sw) ** 2
        )
        penalty_gas_ceiling = torch.mean(
            torch.relu(sw - gas_ceiling) ** 2
        )
        
        # 硬边界兜底
        penalty_hard_low = torch.mean(torch.relu(hard_lower - sw) ** 2)
        penalty_hard_high = torch.mean(torch.relu(sw - hard_upper) ** 2)
        
        return (penalty_hard_low + penalty_hard_high
                + 15.0 * penalty_gas_floor
                + 15.0 * penalty_gas_ceiling)
    
    def loss_k_net_regularization(self,
                                  k_net,
                                  xy_sample: torch.Tensor,
                                  well_xy: Optional[torch.Tensor] = None,
                                  well_k_obs: Optional[torch.Tensor] = None
                                  ) -> Dict[str, torch.Tensor]:
        """
        k_net 正则化损失
        
        Args:
            k_net: PermeabilityNet
            xy_sample: (N, 2) 随机采样的空间点
            well_xy: (n_wells, 2) 井位归一化坐标
            well_k_obs: (n_wells,) 井位观测渗透率 log(mD)
        """
        result = {}
        
        # TV 正则
        result['k_tv'] = k_net.compute_tv_regularization(xy_sample) * 0.01
        
        # Laplacian 正则
        result['k_lap'] = k_net.compute_laplacian_regularization(xy_sample) * 0.001
        
        # 井点先验
        if well_xy is not None and well_k_obs is not None:
            log_k_pred = k_net(well_xy)
            result['k_well'] = torch.mean((log_k_pred.squeeze() - well_k_obs) ** 2)
        else:
            result['k_well'] = torch.tensor(0.0, device=xy_sample.device)
        
        return result
    
    # ================================================================== #
    #                    PDE 残差空间热力图辅助
    # ================================================================== #
    
    @torch.no_grad()
    def compute_residual_map(self,
                             model,
                             x_grid: torch.Tensor,
                             h_grad: Optional[Dict[str, torch.Tensor]] = None
                             ) -> torch.Tensor:
        """在给定网格上计算 PDE 残差 (用于可视化)"""
        x_grid = x_grid.requires_grad_(True)
        
        with torch.enable_grad():
            grads = model.forward_with_grad(x_grid)
            dp_dx = grads['dp_dx']
            dp_dy = grads['dp_dy']
            dp_dt = grads['dp_dt']
            xyt = grads['xyt']
            
            d2p_dx2 = torch.autograd.grad(
                dp_dx, xyt,
                grad_outputs=torch.ones_like(dp_dx),
                create_graph=False, retain_graph=True
            )[0][:, 0:1]
            
            d2p_dy2 = torch.autograd.grad(
                dp_dy, xyt,
                grad_outputs=torch.ones_like(dp_dy),
                create_graph=False
            )[0][:, 1:2]
        
        residual = dp_dt - self.alpha_x * d2p_dx2 - self.alpha_y * d2p_dy2
        
        if h_grad is not None:
            gx = h_grad['gx']
            gy = h_grad['gy']
            residual = residual - self.alpha_x * gx * dp_dx - self.alpha_y * gy * dp_dy
        
        return residual.abs().squeeze(-1)
    
    # ================================================================== #
    #                      总损失组装
    # ================================================================== #
    
    def total_loss(self,
                   model,
                   batch: Dict[str, torch.Tensor],
                   well_outputs: Optional[Dict[str, Dict]] = None,
                   weights: Optional[Dict[str, float]] = None,
                   k_net=None,
                   k_eff_mD_tensor: Optional[torch.Tensor] = None
                   ) -> Dict[str, torch.Tensor]:
        """
        计算完整的 M5 总损失 (v2 — 梯度贯通版)
        
        Args:
            model: M5PINNNet 模型
            batch: {
                'x_ic': IC 点, 'x_bc': BC 点, 'x_pde': PDE 点,
                'h_grad': {gx, gy}, 'source': 井源项 (可选)
            }
            well_outputs: {well_id: {qg, p_wf, qg_obs, whp_obs, ...}}
            weights: 各损失项权重
            k_net: PermeabilityNet 子网络引用 (梯度贯通)
            k_eff_mD_tensor: 可训练标量张量 (梯度贯通)
            
        Returns:
            {loss_name: value, 'total': weighted_sum}
        """
        if weights is None:
            weights = {}
        
        losses = {}
        
        # --- IC 损失 ---
        # phys_enable 为 False 时 (如 pure_ml 消融组) 跳过所有物理损失
        w_ic = weights.get('ic', 1.0)
        if self.phys_enable and w_ic > 0 and 'x_ic' in batch:
            losses['ic'] = self.loss_ic(model, batch['x_ic'])
        else:
            losses['ic'] = torch.tensor(0.0, device=self.device)
        
        # --- BC 损失 ---
        w_bc = weights.get('bc', 1.0)
        if self.phys_enable and w_bc > 0 and 'x_bc' in batch:
            losses['bc'] = self.loss_bc(model, batch['x_bc'])
        else:
            losses['bc'] = torch.tensor(0.0, device=self.device)
        
        # --- PDE 损失 (两相守恒) ---
        w_pde = weights.get('pde', 1.0)
        if self.phys_enable and w_pde > 0 and 'x_pde' in batch:
            losses['pde'] = self.loss_pde(
                model, batch['x_pde'],
                h_grad=batch.get('h_grad'),
                source_g=batch.get('source'),
                source_w=batch.get('source_w'),
                k_net=k_net,
                k_eff_mD_tensor=k_eff_mD_tensor,
            )
        else:
            losses['pde'] = torch.tensor(0.0, device=self.device)
        
        # --- 监督损失 (产量/压力) ---
        losses['qg'] = torch.tensor(0.0, device=self.device)
        losses['qg_nearzero'] = torch.tensor(0.0, device=self.device)
        losses['shutin_delta'] = torch.tensor(0.0, device=self.device)
        losses['whp'] = torch.tensor(0.0, device=self.device)
        losses['qw'] = torch.tensor(0.0, device=self.device)
        
        if well_outputs:
            qg_loss_total = torch.tensor(0.0, device=self.device)
            qg_nearzero_total = torch.tensor(0.0, device=self.device)
            shutin_delta_total = torch.tensor(0.0, device=self.device)
            whp_loss_total = torch.tensor(0.0, device=self.device)
            n_wells = 0
            
            for wid, wdata in well_outputs.items():
                n_wells += 1
                
                if 'qg' in wdata and 'qg_obs' in wdata:
                    qg_obs = wdata['qg_obs']
                    qg_valid = wdata.get('qg_valid_mask', None)
                    scale = qg_obs[qg_obs.abs() > 1.0].std().item() \
                        if (qg_obs.abs() > 1.0).sum() > 1 else 1.0
                    qg_loss_total = qg_loss_total + self.loss_qg(
                        wdata['qg'], qg_obs, scale=scale, valid_mask=qg_valid
                    )

                if self.nearzero_enable and 'qg' in wdata and 'qg_obs' in wdata:
                    qg_obs = wdata['qg_obs']
                    qg_valid = wdata.get('qg_valid_mask', None)
                    qg_nearzero_total = qg_nearzero_total + self.loss_qg_nearzero(
                        wdata['qg'],
                        qg_obs,
                        threshold=float(self.nearzero_threshold),
                        q_scale=float(self.nearzero_qscale),
                        valid_mask=qg_valid,
                        min_points=int(self.nearzero_min_points)
                    )
                
                if 'p_cell' in wdata and 'p_wf' in wdata and 'qg_obs' in wdata:
                    qg_valid = wdata.get('qg_valid_mask', None)
                    shutin_delta_total = shutin_delta_total + self.loss_shutin_delta(
                        wdata['p_cell'], wdata['p_wf'], wdata['qg_obs'],
                        valid_mask=qg_valid,
                        target_MPa=self.shutin_target_mpa,
                        min_points=self.shutin_min_points
                    )
                
                # v2: WHP 转换为 p_wf 约束
                if 'p_wf' in wdata and 'p_obs' in wdata and 'dp_wellbore' in wdata:
                    valid_mask = wdata['p_obs'].abs() > 0.1
                    if valid_mask.any():
                        whp_loss_total = whp_loss_total + self.loss_whp(
                            wdata['p_wf'][valid_mask],
                            wdata['p_obs'][valid_mask],
                            wdata['dp_wellbore']
                        )
            
            # v3.14: 产水量同化 — qw_pred vs qw_obs
            qw_loss_total = torch.tensor(0.0, device=self.device)
            for wid, wdata in well_outputs.items():
                if 'qw_pred' in wdata and 'qw_obs' in wdata:
                    qw_obs = wdata['qw_obs']
                    # scale = qw 标准差 (量级 ~2-6 t/d, 远小于 qg)
                    qw_nonzero = qw_obs[qw_obs.abs() > 0.01]
                    scale = qw_nonzero.std().item() if len(qw_nonzero) > 5 else 1.0
                    qw_loss_total = qw_loss_total + self.loss_qw(
                        wdata['qw_pred'], qw_obs, scale=max(scale, 0.1)
                    )
            
            if n_wells > 0:
                losses['qg'] = qg_loss_total / n_wells
                losses['qg_nearzero'] = qg_nearzero_total / n_wells
                losses['shutin_delta'] = shutin_delta_total / n_wells
                losses['whp'] = whp_loss_total / n_wells
                losses['qw'] = qw_loss_total / n_wells
        
        # --- 正则/约束损失 ---
        losses['smooth_pwf'] = torch.tensor(0.0, device=self.device)
        losses['smooth_qg'] = torch.tensor(0.0, device=self.device)
        losses['pwf_constraint'] = torch.tensor(0.0, device=self.device)
        losses['monotonic'] = torch.tensor(0.0, device=self.device)
        losses['prior'] = torch.tensor(0.0, device=self.device)
        losses['sw_bounds'] = torch.tensor(0.0, device=self.device)
        losses['k_reg'] = torch.tensor(0.0, device=self.device)
        
        if well_outputs and weights.get('pwf_constraint', 0) > 0:
            constraint_total = torch.tensor(0.0, device=self.device)
            for wid, wdata in well_outputs.items():
                if 'p_cell' in wdata and 'p_wf' in wdata:
                    constraint_total = constraint_total + self.pwf_physical_constraint_loss(
                        wdata['p_cell'], wdata['p_wf']
                    )
            losses['pwf_constraint'] = constraint_total
        
        if well_outputs and self.smooth_pwf_enable:
            smooth_total = torch.tensor(0.0, device=self.device)
            for wid, wdata in well_outputs.items():
                if 'p_wf' in wdata and 't_norm' in wdata:
                    pwf_net = model.well_model.pwf_nets[wid]
                    smooth_total = smooth_total + self.loss_smooth_pwf(
                        pwf_net, wdata['t_norm'],
                        prod_hours_norm=wdata.get('prod_hours_norm', None),
                    )
            losses['smooth_pwf'] = smooth_total
        
        # v3.9: qg_pred 时间光滑正则 (抑制 Peaceman 乘法链放大的高频抖动)
        if well_outputs:
            smooth_qg_total = torch.tensor(0.0, device=self.device)
            for wid, wdata in well_outputs.items():
                if 'qg' in wdata and 'qg_obs' in wdata:
                    qg_valid = wdata.get('qg_valid_mask', None)
                    smooth_qg_total = smooth_qg_total + self.loss_smooth_qg(
                        wdata['qg'], wdata['qg_obs'],
                        valid_mask=qg_valid,
                    )
            losses['smooth_qg'] = smooth_qg_total
        
        if well_outputs and self.monotonic_enable:
            mono_total = torch.tensor(0.0, device=self.device)
            for wid, wdata in well_outputs.items():
                if 'qg' in wdata and 'p_wf' in wdata:
                    mono_total = mono_total + self.loss_monotonic_qg_pwf(
                        wdata['qg'], wdata['p_wf']
                    )
            losses['monotonic'] = mono_total
        
        # 反演参数先验
        if hasattr(model, 'get_inversion_param_tensors'):
            inv_params = model.get_inversion_param_tensors()
            losses['prior'] = self.loss_prior_params(inv_params)
        
        # k_net 正则化
        if hasattr(model, 'k_net') and model.k_net is not None and 'x_pde' in batch:
            xy_sample = batch['x_pde'][:, :2].detach().requires_grad_(True)
            k_reg = self.loss_k_net_regularization(model.k_net, xy_sample)
            losses['k_reg'] = k_reg['k_tv'] + k_reg['k_lap'] + k_reg['k_well']
        
        # ★ Sw 对称屏障约束
        w_sw = weights.get('sw_bounds', 0.0)
        if self.sw_bounds_enable and w_sw > 0 and 'x_pde' in batch:
            _, sw_pde = model(batch['x_pde'])
            losses['sw_bounds'] = self.loss_sw_bounds(sw_pde)
        
        # --- 加权求和 ---
        total = torch.tensor(0.0, device=self.device)
        for key, val in losses.items():
            if key == 'total':
                continue
            w = weights.get(key, 1.0)
            total = total + w * val
        
        losses['total'] = total
        return losses

