"""
M4 PINN 损失函数
按报告 5.7 分阶段设计，最小闭环版本

损失组成:
    L_IC   — 初始条件损失 (t=0 时压力/饱和度)
    L_BC   — 边界条件损失 (定压边界)
    L_data — 数据锚点损失 (SY9 压力趋势)
    L_PDE  — PDE 残差损失 (两相守恒方程)

总损失:
    L = λ_IC * L_IC + λ_BC * L_BC + λ_data * L_data + λ_PDE * L_PDE

Debug NaN: 当 config.debug_nan 或 --debug-nan 开启时，对每个子损失做 torch.isfinite 断言，
若发现非有限数则打印 step/stage、dump 到 outputs/debug_nan/step_{step}.pt 并 raise RuntimeError。
"""

import sys
import os
from typing import Dict, Optional, Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("M4 PINN 模块需要 PyTorch，请运行: pip install torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger, ensure_dir

try:
    from .torch_physics import TorchPVT, TorchRelPerm
except ImportError:
    from torch_physics import TorchPVT, TorchRelPerm


class PINNLoss:
    """PINN 多目标损失函数"""
    
    def __init__(self, config: dict, device: str = 'cpu'):
        """
        初始化损失函数
        
        Args:
            config: 全局配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device
        self.logger = setup_logger('PINNLoss')
        
        # 物理参数
        mk = config.get('mk_formation', {})
        p_avg = mk.get('avg_pressure_MPa', {})
        self.p_init = p_avg.get('value', 76.0) if isinstance(p_avg, dict) else float(p_avg)
        self.p_boundary = self.p_init  # 外边界定压（近似初始地层压力）
        
        # 初始含水饱和度（气藏初始 Sw 较低）
        self.sw_init = 0.15  # v3.14: SY9 测井解释 (附表8 加权平均 ≈14.95%)
        
        # 气水界面标高
        gwc = config.get('coordinate_system', {}).get('gas_water_contact', {})
        self.gwc_z = gwc.get('value', -4385.0) if isinstance(gwc, dict) else -4385.0
        
        # 井筒压差：模型输出地层压力，data_loss 中减去 dp_wellbore 后与井口压力 p_obs 对比（与 validate_m4 一致）
        dp_wb_cfg = mk.get('dp_wellbore_MPa', {})
        self.dp_wellbore = (dp_wb_cfg.get('value', 18.0) if isinstance(dp_wb_cfg, dict) else float(dp_wb_cfg))
        
        # 归一化 PDE 缩放系数（由 trainer 在初始化后注入，默认 1.0 兼容旧行为；用于诊断）
        self.alpha_x = 1.0
        self.alpha_y = 1.0
        
        # === 可微分物性模块 (Bug2 进阶: 替代 Corey 硬编码) ===
        self.pvt = TorchPVT(config)
        self.relperm = TorchRelPerm(config)
        pde_domain = config.get('physics', {}).get('pde', {}).get('domain', {})
        self.dx = pde_domain.get('x_max_m', 18555317.902) - pde_domain.get('x_min_m', 18537917.902)
        self.dy = pde_domain.get('y_max_m', 3357340.360) - pde_domain.get('y_min_m', 3346340.360)
        t_max_d = pde_domain.get('t_max_d', 1331)
        self.t_max_s = t_max_d * 86400.0
        physics_priors = config.get('physics', {}).get('priors', {})
        phi_cfg = physics_priors.get('phi', {})
        self.phi = phi_cfg.get('value', 0.0216) if isinstance(phi_cfg, dict) else 0.0216
        self.h_mean = 90.0
        if device != 'cpu':
            self.pvt = self.pvt.to(device)
            self.relperm = self.relperm.to(device)
        
        # Debug NaN: 非有限数检测与 dump（默认关闭）
        self.debug_nan = config.get('debug_nan', False)
        out_root = config.get('paths', {}).get('outputs', 'outputs')
        self.debug_nan_dump_dir = os.path.join(out_root, 'debug_nan')
        self._last_pde_debug = None
        self._debug_step = None
        self._debug_stage = None
        self._debug_batch = None
        self._debug_x_data = None
        self._debug_p_obs = None
        self._debug_h_grad = None
        
        self.logger.info(f"PINNLoss 初始化: p_init={self.p_init} MPa, "
                         f"sw_init={self.sw_init}, p_boundary={self.p_boundary} MPa"
                         + (f", debug_nan=True dump_dir={self.debug_nan_dump_dir}" if self.debug_nan else ""))
    
    def set_debug_context(self, step: int, stage: str, batch: Dict[str, Any],
                          x_data: Optional[torch.Tensor] = None,
                          p_obs: Optional[torch.Tensor] = None,
                          h_grad: Optional[dict] = None):
        """由 Trainer 在每步开始时调用，用于 NaN 时 dump 完整信息。"""
        self._debug_step = step
        self._debug_stage = stage
        self._debug_batch = batch
        self._debug_x_data = x_data
        self._debug_p_obs = p_obs
        self._debug_h_grad = h_grad
    
    def _dump_nan_and_raise(self, stage_name: str, loss_value: torch.Tensor, model: nn.Module):
        """发现非有限损失时：补全 PDE debug（若尚未有）、构建 dump、保存并 raise RuntimeError。"""
        step = self._debug_step
        stage = self._debug_stage
        batch = self._debug_batch
        x_data = self._debug_x_data
        p_obs = self._debug_p_obs
        h_grad = self._debug_h_grad
        # 若尚未计算过 PDE 本步，补算一次仅用于 dump（不参与 backward）
        if self._last_pde_debug is None and batch is not None and 'x_pde' in batch:
            try:
                self.pde_loss(model, batch['x_pde'], h_grad=h_grad, store_debug_only=True)
            except Exception:
                pass
        dump = {
            'step': step,
            'stage': stage,
            'failed_stage': stage_name,
            'loss_value': loss_value.detach().cpu() if isinstance(loss_value, torch.Tensor) else loss_value,
        }
        # batch 输入 (x,y,t)
        if batch:
            for k in ('x_ic', 'x_bc', 'x_pde', 'x_data'):
                if k in batch and batch[k] is not None:
                    dump[f'batch_{k}'] = batch[k].detach().cpu()
        if x_data is not None:
            dump['batch_x_data'] = x_data.detach().cpu()
        if p_obs is not None:
            dump['batch_p_obs'] = p_obs.detach().cpu()
        if h_grad is not None:
            dump['batch_h_grad'] = {k: v.detach().cpu() for k, v in h_grad.items()}
        # 模型输出
        with torch.no_grad():
            if batch and 'x_ic' in batch:
                p_ic, sw_ic = model(batch['x_ic'])
                dump['model_out_ic_p'] = p_ic.cpu()
                dump['model_out_ic_sw'] = sw_ic.cpu()
            if batch and 'x_bc' in batch:
                p_bc, sw_bc = model(batch['x_bc'])
                dump['model_out_bc_p'] = p_bc.cpu()
                dump['model_out_bc_sw'] = sw_bc.cpu()
            if batch and 'x_pde' in batch:
                p_pde, sw_pde = model(batch['x_pde'])
                dump['model_out_pde_p'] = p_pde.cpu()
                dump['model_out_pde_sw'] = sw_pde.cpu()
            if x_data is not None:
                p_data, sw_data = model(x_data)
                dump['model_out_data_p'] = p_data.cpu()
                dump['model_out_data_sw'] = sw_data.cpu()
        # PDE 物性/厚度/渗透率、关键分母项、pde residual（来自 _last_pde_debug）
        if self._last_pde_debug is not None:
            for k, v in self._last_pde_debug.items():
                dump[f'pde_{k}'] = v
        ensure_dir(self.debug_nan_dump_dir)
        path = os.path.join(self.debug_nan_dump_dir, f'step_{step}.pt')
        torch.save(dump, path)
        self.logger.error(
            f"[debug_nan] non-finite loss in {stage_name} at step={step} stage={stage}. "
            f"Dumped to {path}"
        )
        raise RuntimeError(
            f"PINNLoss: non-finite loss in {stage_name} at step={step} stage={stage}. Dumped to {path}"
        )
    
    def ic_loss(self, model: nn.Module, x_ic: torch.Tensor) -> torch.Tensor:
        """
        初始条件损失 L_IC
        t=0 时: p ≈ p_init, Sw ≈ Sw_init
        
        Args:
            model: PINN 模型
            x_ic: (N, 3) 初始条件点 [x, y, t=0]
        """
        p_pred, sw_pred = model(x_ic)
        
        # 压力初始值（先简化为均匀场，后续可加压力梯度）
        p_target = torch.full_like(p_pred, self.p_init)
        loss_p = torch.mean((p_pred - p_target) ** 2)
        
        # 饱和度初始值
        sw_target = torch.full_like(sw_pred, self.sw_init)
        loss_sw = torch.mean((sw_pred - sw_target) ** 2)
        
        loss = loss_p + loss_sw
        if self.debug_nan and getattr(self, '_debug_step', None) is not None:
            if not torch.isfinite(loss).item():
                self._dump_nan_and_raise('IC', loss, model)
        return loss
    
    def bc_loss(self, model: nn.Module, x_bc: torch.Tensor) -> torch.Tensor:
        """
        边界条件损失 L_BC
        外边界定压: p|∂Ω = p_boundary
        
        Args:
            model: PINN 模型
            x_bc: (N, 3) 边界点 [x, y, t]
        """
        p_pred, sw_pred = model(x_bc)
        
        # 定压边界
        p_target = torch.full_like(p_pred, self.p_boundary)
        loss_p = torch.mean((p_pred - p_target) ** 2)
        
        if self.debug_nan and getattr(self, '_debug_step', None) is not None:
            if not torch.isfinite(loss_p).item():
                self._dump_nan_and_raise('BC', loss_p, model)
        return loss_p
    
    def data_loss(self, model: nn.Module, x_data: torch.Tensor,
                  p_obs: torch.Tensor) -> torch.Tensor:
        """
        数据锚点损失 L_data
        模型输出地层压力，减去井筒压差后与观测井口压力 p_obs (tubing_p_avg) 对比，训练/验收口径一致。
        P0: 仅对 p_obs 有限点计算损失，避免 NaN 污染（训练端已过滤，此处双重防御）。
        
        Args:
            model: PINN 模型
            x_data: (N, 3) 数据点 [x, y, t]
            p_obs: (N,) 或 (N, 1) 观测井口压力 (MPa)，可能含 NaN
        """
        if p_obs.dim() == 1:
            p_obs = p_obs.unsqueeze(-1)
        valid = torch.isfinite(p_obs) & (p_obs > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device, dtype=x_data.dtype)
        x_data = x_data[valid.squeeze(-1)]
        p_obs = p_obs[valid].reshape(-1, 1)
        p_pred_form, _ = model(x_data)
        p_pred_whp = p_pred_form - self.dp_wellbore  # 地层压力 → 井口压力
        loss = torch.mean((p_pred_whp - p_obs) ** 2)
        if self.debug_nan and getattr(self, '_debug_step', None) is not None:
            if not torch.isfinite(loss).item():
                self._dump_nan_and_raise('Data', loss, model)
        return loss
    
    def pde_loss(self, model: nn.Module, x_pde: torch.Tensor,
                 h_grad: dict = None, pde_mask: torch.Tensor = None,
                 store_debug_only: bool = False) -> torch.Tensor:
        """
        两相守恒 PDE 残差 (Bug2 方案B进阶版 + Bug5 残差截断)
        
        气相: φ·h·[(1-Sw)·ρ_g·c_g·∂p/∂t - ρ_g·∂Sw/∂t] = ∇·(k·krg·h/μ_g·∇p) + 交叉项
        水相: φ·h·ρ_w·∂Sw/∂t = ∇·(k·krw·h/μ_w·∇p) + 交叉项
        
        归一化坐标缩放，两残差各自归一化后加权求和，残差截断防爆炸。
        pde_mask: 可选 (N,1)，1=有效 0=域外过滤；域外点不参与 PDE 残差均值。
        store_debug_only: 为 True 时仅填充 _last_pde_debug 并返回 0（用于 NaN dump 时补全 PDE 信息）。
        """
        grads = model.forward_with_grad(x_pde)
        p = grads['p']
        sw = grads['sw']
        dp_dx = grads['dp_dx']
        dp_dy = grads['dp_dy']
        dp_dt = grads['dp_dt']
        dsw_dx = grads['dsw_dx']
        dsw_dy = grads['dsw_dy']
        dsw_dt = grads['dsw_dt']
        xyt = grads['xyt']
        
        rho_g = self.pvt.rho_g(p)
        mu_g = self.pvt.mu_g(p)
        cg_val = self.pvt.cg(p)
        krg = self.relperm.krg(sw)
        krw = self.relperm.krw(sw)
        dkrg_dsw = self.relperm.dkrg_dSw(sw)
        dkrw_dsw = self.relperm.dkrw_dSw(sw)
        rho_w = self.pvt.rho_w
        mu_w = self.pvt.mu_w
        phi = self.phi
        h = self.h_mean
        
        sx = 2.0 / self.dx
        sy = 2.0 / self.dy
        st = 1.0 / self.t_max_s
        dp_dx_phys = sx * dp_dx
        dp_dy_phys = sy * dp_dy
        dp_dt_phys = st * dp_dt
        dsw_dt_phys = st * dsw_dt
        dsw_dx_phys = sx * dsw_dx
        dsw_dy_phys = sy * dsw_dy
        dp_dx_Pa = dp_dx_phys * 1e6
        dp_dy_Pa = dp_dy_phys * 1e6
        
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
        d2p_dx2_Pa = sx * sx * d2p_dx2 * 1e6
        d2p_dy2_Pa = sy * sy * d2p_dy2 * 1e6
        
        physics_priors = self.config.get('physics', {}).get('priors', {})
        k_cfg = physics_priors.get('k_eff_mD', {})
        k_mD = k_cfg.get('value', 0.5) if isinstance(k_cfg, dict) else 0.5
        f_cfg = physics_priors.get('frac_conductivity_factor', {})
        f_frac = f_cfg.get('value', 16.0) if isinstance(f_cfg, dict) else 16.0
        k_eff_mD = k_mD * f_frac
        k_SI = k_eff_mD * 9.869233e-16
        
        gx = h_grad['gx'] if h_grad is not None else None
        gy = h_grad['gy'] if h_grad is not None else None
        # 若厚度梯度含 NaN/Inf（如厚度场未正确加载），则不用厚度项，避免 PDE loss 爆炸
        if gx is not None and (not torch.isfinite(gx).all()):
            gx = None
        if gy is not None and (not torch.isfinite(gy).all()):
            gy = None
        
        Sg = 1.0 - sw
        accumulation_g = phi * h * (
            -rho_g * dsw_dt_phys
            + Sg * rho_g * cg_val * dp_dt_phys
        )
        # 质量守恒口径：通量项含密度 ∇·(k·krg·h·ρ_g/μ_g·∇p)
        T_g = k_SI * krg * h * rho_g / (mu_g + 1e-20)
        flux_g_diffusion = T_g * (d2p_dx2_Pa + d2p_dy2_Pa)
        flux_g_dkr = (k_SI * h * rho_g / (mu_g + 1e-20)) * dkrg_dsw * (
            dsw_dx_phys * dp_dx_Pa + dsw_dy_phys * dp_dy_Pa
        )
        flux_g_h = torch.zeros_like(p)
        if h_grad is not None and gx is not None and gy is not None:
            flux_g_h = T_g * (gx * sx * dp_dx_Pa + gy * sy * dp_dy_Pa)
        R_gas = accumulation_g - (flux_g_diffusion + flux_g_dkr + flux_g_h)
        scale_g = phi * h * 200.0 * st + 1e-12
        R_gas_norm = R_gas / scale_g
        R_gas_norm = torch.clamp(R_gas_norm, -50.0, 50.0)
        
        accumulation_w = phi * h * rho_w * dsw_dt_phys
        # 质量守恒口径：通量项含密度 ∇·(k·krw·h·ρ_w/μ_w·∇p)
        T_w = k_SI * krw * h * rho_w / (mu_w + 1e-20)
        flux_w_diffusion = T_w * (d2p_dx2_Pa + d2p_dy2_Pa)
        flux_w_dkr = (k_SI * h * rho_w / (mu_w + 1e-20)) * dkrw_dsw * (
            dsw_dx_phys * dp_dx_Pa + dsw_dy_phys * dp_dy_Pa
        )
        flux_w_h = torch.zeros_like(p)
        if h_grad is not None and gx is not None and gy is not None:
            flux_w_h = T_w * (gx * sx * dp_dx_Pa + gy * sy * dp_dy_Pa)
        R_water = accumulation_w - (flux_w_diffusion + flux_w_dkr + flux_w_h)
        scale_w = phi * h * rho_w * st + 1e-12
        R_water_norm = R_water / scale_w
        R_water_norm = torch.clamp(R_water_norm, -50.0, 50.0)
        
        lambda_sw = 0.5
        if pde_mask is not None and pde_mask.numel() > 0:
            m = pde_mask.expand_as(R_gas_norm)
            denom = m.sum().clamp(min=1e-12)
            loss_gas = (R_gas_norm ** 2 * m).sum() / denom
            loss_water = (R_water_norm ** 2 * m).sum() / denom
            loss = loss_gas + lambda_sw * loss_water
        else:
            loss_gas = torch.mean(R_gas_norm ** 2)
            loss_water = torch.mean(R_water_norm ** 2)
            loss = loss_gas + lambda_sw * loss_water
        
        # Debug NaN: 保存 PDE 中间量供 dump 使用；必要时仅填 debug 不参与 backward
        if self.debug_nan:
            def _to_cpu(v):
                return v.detach().cpu() if isinstance(v, torch.Tensor) else v
            mu_g_denom = mu_g + 1e-20
            mu_w_denom = mu_w + 1e-20
            self._last_pde_debug = {
                'x_pde': _to_cpu(x_pde),
                'p': _to_cpu(p),
                'sw': _to_cpu(sw),
                'rho_g': _to_cpu(rho_g),
                'mu_g': _to_cpu(mu_g),
                'cg_val': _to_cpu(cg_val),
                'krg': _to_cpu(krg),
                'krw': _to_cpu(krw),
                'phi': _to_cpu(phi),
                'h': _to_cpu(h),
                'k_SI': _to_cpu(k_SI),
                'k_eff_mD': _to_cpu(k_eff_mD),
                'scale_g': _to_cpu(scale_g),
                'scale_w': _to_cpu(scale_w),
                'denom_mu_g': _to_cpu(mu_g_denom),
                'denom_mu_w': _to_cpu(mu_w_denom),
                'R_gas': _to_cpu(R_gas),
                'R_water': _to_cpu(R_water),
                'R_gas_norm': _to_cpu(R_gas_norm),
                'R_water_norm': _to_cpu(R_water_norm),
                'T_g': _to_cpu(T_g),
                'T_w': _to_cpu(T_w),
                'd2p_dx2_Pa': _to_cpu(d2p_dx2_Pa),
                'd2p_dy2_Pa': _to_cpu(d2p_dy2_Pa),
            }
        if store_debug_only:
            return torch.tensor(0.0, device=self.device, dtype=loss.dtype)
        if self.debug_nan and getattr(self, '_debug_step', None) is not None:
            if not torch.isfinite(loss).item():
                self._dump_nan_and_raise('PDE', loss, model)
        return loss
    
    def sw_physics_loss(self, model: nn.Module, x_pde: torch.Tensor) -> torch.Tensor:
        """
        Sw 物理弱约束 (Bug2 方案A 安全网): 初始 Swc、水侵单调、上界 1-Sgr
        
        [审查修复 #6] 端点参数统一从 self.relperm 获取, 不再硬编码
        [v3.4] 增加弱 anchor 项: 防止 Stage A 阶段 Sw 漂移到 tanh 上界 0.88
        """
        grads = model.forward_with_grad(x_pde)
        sw = grads['sw']
        dsw_dt = grads['dsw_dt']
        t_norm = x_pde[:, 2:3]
        # 从 TorchRelPerm 统一获取端点, 避免硬编码与 torch_physics 不一致
        Swc = self.relperm.Swc    # 束缚水饱和度 (附表7: 0.26)
        Sgr = self.relperm.Sgr    # 残余气饱和度 (附表7: 0.062)
        # 低含水气藏：上界收紧至 Swc+0.10*t_norm，最大允许 Sw≈0.36（原 0.4 过松导致 Sw 升至 0.6+）
        sw_upper_soft = Swc + 0.10 * t_norm
        loss_drift = torch.mean(torch.relu(sw - sw_upper_soft) ** 2)
        loss_mono = torch.mean(torch.relu(-dsw_dt) ** 2)
        loss_upper = torch.mean(torch.relu(sw - (1.0 - Sgr)) ** 2)
        # v3.4b: 强 anchor — 全时间点 (Sw - Swc)^2, 权重 5.0 确保能对抗 data loss 的间接漂移
        # Sw=0.30 处: 5.0*(0.04)^2=0.008 (几乎不影响), Sw=0.88 处: 5.0*(0.62)^2=1.92 (强拉回)
        loss_anchor = torch.mean((sw - Swc) ** 2)
        return loss_drift + 0.5 * loss_mono + loss_upper + 5.0 * loss_anchor
    
    def total_loss(self, model: nn.Module,
                   x_ic: torch.Tensor,
                   x_bc: torch.Tensor,
                   x_pde: torch.Tensor,
                   x_data: Optional[torch.Tensor] = None,
                   p_obs: Optional[torch.Tensor] = None,
                   weights: Optional[Dict[str, float]] = None
                   ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            model: PINN 模型
            x_ic, x_bc, x_pde: 各类采样点
            x_data, p_obs: 数据锚点（可选）
            weights: 损失权重 {'ic': w1, 'bc': w2, 'pde': w3, 'data': w4}
        
        Returns:
            dict: {'total': L, 'ic': L_IC, 'bc': L_BC, 'pde': L_PDE, 'data': L_data}
        """
        if weights is None:
            weights = {'ic': 1.0, 'bc': 1.0, 'pde': 0.0, 'data': 0.0, 'sw_phys': 0.0}
        
        losses = {}
        losses['ic'] = self.ic_loss(model, x_ic)
        losses['bc'] = self.bc_loss(model, x_bc)
        if weights.get('pde', 0) > 0:
            losses['pde'] = self.pde_loss(model, x_pde)
        else:
            losses['pde'] = torch.tensor(0.0, device=self.device)
        if weights.get('data', 0) > 0 and x_data is not None and p_obs is not None:
            losses['data'] = self.data_loss(model, x_data, p_obs)
        else:
            losses['data'] = torch.tensor(0.0, device=self.device)
        if weights.get('sw_phys', 0) > 0:
            losses['sw_phys'] = self.sw_physics_loss(model, x_pde)
        else:
            losses['sw_phys'] = torch.tensor(0.0, device=self.device)
        
        total = (
            weights.get('ic', 1.0) * losses['ic'] +
            weights.get('bc', 1.0) * losses['bc'] +
            weights.get('pde', 0.0) * losses['pde'] +
            weights.get('data', 0.0) * losses['data'] +
            weights.get('sw_phys', 0.0) * losses['sw_phys']
        )
        losses['total'] = total
        return losses
