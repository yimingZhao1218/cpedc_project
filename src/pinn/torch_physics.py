"""
PyTorch 可微分物性模块
========================
为 PINN 两相 PDE 残差提供全 autograd 兼容的物性计算：

    TorchPVT:      ρ_g(p), Bg(p), μ_g(p), c_g(p)  — 气体 PVT
    TorchRelPerm:  krg(Sw), krw(Sw), dkrg/dSw, dkrw/dSw — 相渗

设计原则:
    1. 所有函数 Tensor→Tensor, create_graph=True 时梯度可回传
    2. 不依赖 scipy/numpy, 用解析公式或多项式拟合
    3. 参数默认从 M3 数据端点校准, 可通过 config 覆盖
    4. 自动 clamp 防止非物理值 (负密度/负渗透率)

参考:
    - Lee-Gonzalez-Eakin (1966): 天然气粘度关联式
    - Hall-Yarborough (1973): Z 因子关联式 (简化版)
    - Corey-Brooks: 两相相渗
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class TorchPVT(nn.Module):
    """
    PyTorch 可微分气体 PVT 物性
    
    基于解析关联式, 全部支持 autograd。
    
    API:
        rho_g(p)   → 气体密度 kg/m³
        bg(p)      → 体积系数 m³/m³ (reservoir/surface)
        mu_g(p)    → 粘度 Pa·s
        cg(p)      → 压缩系数 1/Pa
        z_factor(p)→ 偏差系数 (无量纲)
    
    所有 p 单位: MPa, T 单位: K (内部转换)
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        
        if config is None:
            config = {}
        
        mk = config.get('mk_formation', {})
        priors = config.get('physics', {}).get('priors', {})
        
        # --- 气体参数 ---
        # 气体分子量: 附表5-1井流物组成加权计算
        # M_g = Σ(yi×Mi) = 96.271%×16.04 + 1.854%×44.01 + 0.907%×28.01
        #       + 0.62%×34.08 + 0.288%×30.07 + ... = 16.83 g/mol
        self.M_g = 16.83e-3     # kg/mol (附表5-1: C1=96.27%, CO2=1.85%, N2=0.91%)
        self.R = 8.314           # 通用气体常数 J/(mol·K)
        self.T_res = 140.32 + 273.15  # 储层温度 K
        self.p_sc = 0.101325     # 标准条件压力 MPa
        self.T_sc = 293.15       # 标准条件温度 K (20°C)
        
        # Z 因子: 从附表5-2恒质膨胀数据(T=140.32°C)三次最小二乘拟合
        # Z(p) = a0 + a1*p + a2*p² + a3*p³, p ∈ [12, 76] MPa
        # 拟合 RMSE = 0.0025 (scripts/fit_z_factor_least_squares.py)
        # v3.11-FIX: 替代旧版教科书估算值(1.0, -0.008, 5e-5), 旧值导致 Bg 偏差 50-85%
        self.z_a0 = 1.009820130054331
        self.z_a1 = -0.002216133163443353
        self.z_a2 = 0.0001360507070074421
        self.z_a3 = -5.768441554502428e-07
        
        # Lee-Gonzalez-Eakin (1966) 粘度关联式预计算常数
        # μ_g = 1e-4 × K × exp(X × ρ^Y)  [cp]
        # T: Rankine, ρ: g/cm³, M: g/mol
        T_R = self.T_res * 9.0 / 5.0          # K → Rankine
        M_gmol = self.M_g * 1000.0             # kg/mol → g/mol
        self._lge_K = (9.4 + 0.02 * M_gmol) * T_R**1.5 / (209.0 + 19.0 * M_gmol + T_R)
        self._lge_X = 3.5 + 986.0 / T_R + 0.01 * M_gmol
        self._lge_Y = 2.4 - 0.2 * self._lge_X
        # 参考值(保留用于日志/诊断)
        self.p_ref = 76.0  # MPa (参考压力)
        
        # 岩石压缩系数
        cr_cfg = priors.get('c_r_1_per_MPa', {})
        self.c_r = (cr_cfg.get('value', 0.0) 
                    if isinstance(cr_cfg, dict) else 0.0) * 1e-6  # 1/Pa
        
        # 水相参数 (常数, 微可压)
        # 数据来源: 附表6-水分析 SY9 地层卤水 (2016-09 峰值 TDS=105,158 mg/L)
        # 地面实测相对密度 1.069 → 储层条件(T=140°C, p=76MPa)热膨胀修正 ≈ 1030~1055
        self.rho_w = 1050.0     # kg/m³ (储层原位地层水密度, 取保守估计)
        # v3.21-FIX: 旧值 0.5e-3 偏高~80% (教科书常温值)
        # Kestin-Khalifa关联式 @T=140°C, TDS≈105g/L, p=76MPa → μ_w ≈ 0.28 cp
        self.mu_w = 0.28e-3     # Pa·s (地层水粘度, Kestin关联式)
        self.c_w = 4.5e-10      # 1/Pa (水压缩系数, McCain关联式 @140°C)
        self.Bw = 1.02          # 水体积系数 (高温高压略偏低, 实际~1.04)
    
    def z_factor(self, p: torch.Tensor) -> torch.Tensor:
        """
        偏差系数 Z(p) — 二次多项式近似
        
        Args:
            p: 压力 (MPa)
        Returns:
            Z: 无量纲, > 0
        """
        p_c = torch.clamp(p, 1.0, 120.0)
        Z = self.z_a0 + self.z_a1 * p_c + self.z_a2 * p_c ** 2 + self.z_a3 * p_c ** 3
        return torch.clamp(Z, 0.3, 2.0)
    
    def rho_g(self, p: torch.Tensor) -> torch.Tensor:
        """
        气体密度 ρ_g(p) = p·M / (Z·R·T)
        
        Args:
            p: 压力 (MPa)
        Returns:
            ρ_g: kg/m³
        """
        p_Pa = torch.clamp(p, 0.5, 120.0) * 1e6  # MPa → Pa
        Z = self.z_factor(p)
        rho = p_Pa * self.M_g / (Z * self.R * self.T_res)
        return torch.clamp(rho, 1.0, 1000.0)
    
    def bg(self, p: torch.Tensor) -> torch.Tensor:
        """
        气体体积系数 Bg(p) = Z·T_res·p_sc / (p·T_sc)
        
        Args:
            p: 压力 (MPa)
        Returns:
            Bg: m³/m³ (reservoir / surface)
        """
        p_c = torch.clamp(p, 0.5, 120.0)
        Z = self.z_factor(p)
        Bg = Z * self.T_res * self.p_sc / (p_c * self.T_sc)
        return torch.clamp(Bg, 1e-4, 1.0)
    
    def mu_g(self, p: torch.Tensor) -> torch.Tensor:
        """
        气体粘度 μ_g(p) — 完整 Lee-Gonzalez-Eakin (1966) 关联式
        
        μ_g = 1e-4 × K × exp(X × ρ^Y)  [cp]
        K, X, Y 仅依赖 M_g 和 T_res, 在 __init__ 中预计算
        
        Args:
            p: 压力 (MPa)
        Returns:
            μ_g: Pa·s
        """
        rho = self.rho_g(p)                        # kg/m³
        rho_gcm3 = torch.clamp(rho / 1000.0, 0.005, 1.0)  # → g/cm³
        mu_cp = 1e-4 * self._lge_K * torch.exp(
            self._lge_X * torch.pow(rho_gcm3, self._lge_Y)
        )
        return torch.clamp(mu_cp * 1e-3, 1e-6, 1e-2)  # cp → Pa·s
    
    def cg(self, p: torch.Tensor) -> torch.Tensor:
        """
        气体压缩系数 c_g(p) = 1/p - (1/Z)(dZ/dp)
        
        Args:
            p: 压力 (MPa)
        Returns:
            c_g: 1/MPa
        """
        p_c = torch.clamp(p, 1.0, 120.0)
        Z = self.z_factor(p)
        dZ_dp = self.z_a1 + 2 * self.z_a2 * p_c + 3 * self.z_a3 * p_c ** 2
        cg = 1.0 / p_c - dZ_dp / (Z + 1e-12)
        return torch.clamp(cg, 1e-4, 1.0)
    
    def ct(self, p: torch.Tensor, Sw: torch.Tensor) -> torch.Tensor:
        """
        总压缩系数 c_t = Sg·c_g + Sw·c_w + c_r
        
        Args:
            p: 压力 (MPa)
            Sw: 含水饱和度
        Returns:
            c_t: 1/MPa
        """
        Sg = 1.0 - Sw
        cg_val = self.cg(p)
        cw_val = self.c_w * 1e6  # 1/Pa → 1/MPa
        cr_val = self.c_r * 1e6  # 1/Pa → 1/MPa
        return Sg * cg_val + Sw * cw_val + cr_val
    
    def drho_g_dp(self, p: torch.Tensor) -> torch.Tensor:
        """
        dρ_g/dp — 用 autograd 精确计算
        """
        p_req = p.detach().requires_grad_(True)
        rho = self.rho_g(p_req)
        grad = torch.autograd.grad(
            rho.sum(), p_req, create_graph=True
        )[0]
        return grad


class TorchRelPerm(nn.Module):
    """
    PyTorch 可微分两相相渗 (Corey-Brooks 模型)
    
    krg(Sw) = krg_max · ((1 - Sw - Sgr) / (1 - Swc - Sgr))^ng
    krw(Sw) = krw_max · ((Sw - Swc) / (1 - Swc - Sgr))^nw
    
    端点参数从 M3 RelPermGW 数据获取。
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        
        # 端点参数 (from M3 附表7)
        self.Swc = 0.26       # 束缚水饱和度
        self.Sgr = 0.062      # 残余气饱和度
        self.krg_max = 0.675  # Sw=Swc 时 krg
        self.krw_max = 0.480  # Sw=1-Sgr 时 krw
        
        # Corey 指数: SY13 附表7 21点最小二乘拟合 → SY9 PINN 可学习
        # 初始值来自 scripts/fit_corey_exponents.py (R²>0.98)
        # 训练中由 SY9 产气量数据驱动微调, 实现跨井等效相渗反演
        # 参数化: ng = exp(_ng_log), 保证 ng > 0
        import math
        ng_init, nw_init = 1.0846, 4.4071
        self._ng_log = nn.Parameter(torch.tensor(math.log(ng_init)))
        self._nw_log = nn.Parameter(torch.tensor(math.log(nw_init)))
        
        # 先验值 (用于正则化, 防止偏离 SY13 实验值过远)
        self.ng_prior = ng_init
        self.nw_prior = nw_init
        
        # 归一化分母
        self._denom = 1.0 - self.Swc - self.Sgr  # = 0.678
    
    @property
    def ng(self) -> torch.Tensor:
        """气相 Corey 指数 (exp 参数化, 恒正)"""
        return torch.exp(self._ng_log)
    
    @property
    def nw(self) -> torch.Tensor:
        """水相 Corey 指数 (exp 参数化, 恒正)"""
        return torch.exp(self._nw_log)
    
    def krg(self, Sw: torch.Tensor) -> torch.Tensor:
        """
        气相相对渗透率 krg(Sw)
        """
        Sw_c = torch.clamp(Sw, self.Swc, 1.0 - self.Sgr)
        Se_g = torch.clamp((1.0 - Sw_c - self.Sgr) / self._denom, 0.0, 1.0)
        return self.krg_max * Se_g ** self.ng + 1e-4  # ε-floor
    
    def krw(self, Sw: torch.Tensor) -> torch.Tensor:
        """
        水相相对渗透率 krw(Sw)
        """
        Sw_c = torch.clamp(Sw, self.Swc, 1.0 - self.Sgr)
        Se_w = torch.clamp((Sw_c - self.Swc) / self._denom, 0.0, 1.0)
        return self.krw_max * Se_w ** self.nw
    
    def dkrg_dSw(self, Sw: torch.Tensor) -> torch.Tensor:
        """dkrg/dSw — 解析导数"""
        Sw_c = torch.clamp(Sw, self.Swc + 1e-6, 1.0 - self.Sgr - 1e-6)
        Se_g = torch.clamp((1.0 - Sw_c - self.Sgr) / self._denom, 1e-8, 1.0)
        # d(Se_g)/dSw = -1/denom
        return self.krg_max * self.ng * Se_g ** (self.ng - 1) * (-1.0 / self._denom)
    
    def dkrw_dSw(self, Sw: torch.Tensor) -> torch.Tensor:
        """dkrw/dSw — 解析导数"""
        Sw_c = torch.clamp(Sw, self.Swc + 1e-6, 1.0 - self.Sgr - 1e-6)
        Se_w = torch.clamp((Sw_c - self.Swc) / self._denom, 1e-8, 1.0)
        return self.krw_max * self.nw * Se_w ** (self.nw - 1) * (1.0 / self._denom)
    
    def fractional_flow_water(self, Sw: torch.Tensor,
                               mu_g: torch.Tensor,
                               mu_w: float = 0.5e-3
                               ) -> torch.Tensor:
        """
        水相分流量 f_w = λ_w / (λ_w + λ_g)
        
        用于 Buckley-Leverett 分析和水侵预警
        """
        kr_g = self.krg(Sw)
        kr_w = self.krw(Sw)
        lambda_g = kr_g / (mu_g + 1e-20)
        lambda_w = kr_w / (mu_w + 1e-20)
        return lambda_w / (lambda_w + lambda_g + 1e-20)
