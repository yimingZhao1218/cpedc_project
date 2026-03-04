"""
M4/M5 PINN 网络结构 (增强版)
===============================
核心改进:
    1. Fourier Feature Encoding — 解决 PINN 频谱偏置问题
    2. tanh 参数化压力输出 — 替代 sigmoid 避免梯度消失
    3. 更大网络容量 — 6×256 + 残差连接
    4. k(x,y) 子网络 — 空间渗透率场参数化 (计划书核心要求)
    5. Sw bias 初始化 — sigmoid(bias) ≈ Sw_init

输入: (x, y, t) 归一化坐标
输出: p(x,y,t) 压力, Sw(x,y,t) 含水饱和度
可选: log_k(x,y) 对数渗透率场

参考:
    - Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency Functions"
    - Wang et al. (2021) "Understanding and Mitigating Gradient Flow Pathologies in PINNs"
"""

import sys
import os
import math
import numpy as np
from typing import Tuple, Optional, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("M4 PINN 模块需要 PyTorch，请运行: pip install torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FourierFeatureEncoding(nn.Module):
    """
    Fourier Feature Mapping (随机傅里叶特征)
    
    形式: γ(x) = [x, cos(2π·B·x), sin(2π·B·x)]
    其中 B 为 (n_freq, input_dim) 随机高斯矩阵，不参与训练。
    proj = x @ B^T 得到 (B, n_freq)，再对 proj 做 cos/sin，故输出维度 = input_dim + 2*n_freq。
    有效缓解 MLP 的频谱偏置问题。
    
    Ref: Tancik et al. (2020) NeurIPS
    """
    
    def __init__(self, input_dim: int = 3, n_freq: int = 32, sigma: float = 1.0):
        """
        Args:
            input_dim: 输入维度 (3 for x,y,t)
            n_freq: 频率数
            sigma: 频率矩阵 B 的标准差 (控制频率范围)
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_freq = n_freq
        self.output_dim = input_dim + 2 * n_freq
        
        # 随机高斯频率矩阵 (不参与训练), shape (n_freq, input_dim)
        B = torch.randn(n_freq, input_dim) * sigma
        self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, input_dim + 2*n_freq) 编码后的特征 [x, cos(proj), sin(proj)]
        """
        proj = 2.0 * math.pi * torch.matmul(x, self.B.T)  # (B, n_freq)
        features = [x, torch.cos(proj), torch.sin(proj)]
        return torch.cat(features, dim=-1)


class ResidualBlock(nn.Module):
    """残差块: 每两层一个跳跃连接, 防止梯度消失"""
    
    def __init__(self, width: int, activation: nn.Module, use_layernorm: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = activation
        self.norm1 = nn.LayerNorm(width) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(width) if use_layernorm else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return self.act(out + residual)


class PermeabilityNet(nn.Module):
    """
    k(x,y) 空间渗透率场子网络
    
    输入: (x_norm, y_norm) 二维空间坐标
    输出: log_k(x,y) 对数渗透率 (mD 空间)
    
    正则化:
        - TV (Total Variation) 促进分片平滑
        - Laplacian 正则防噪声
        - 井点先验约束
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        physics_cfg = config.get('physics', {}).get('priors', {})
        k_cfg = physics_cfg.get('k_eff_mD', {})
        self.k_prior_log = math.log(max(
            k_cfg.get('value', 5.0) if isinstance(k_cfg, dict) else 5.0,
            0.01
        ))
        k_bounds = k_cfg.get('bounds', [0.01, 200.0]) if isinstance(k_cfg, dict) else [0.01, 200.0]
        self.log_k_min = math.log(max(k_bounds[0], 1e-6))
        self.log_k_max = math.log(k_bounds[1])
        
        # 小型 MLP: (x, y) → log_k
        # 也加 Fourier features 以捕捉空间变异性
        self.fourier = FourierFeatureEncoding(input_dim=2, n_freq=16, sigma=2.0)
        
        in_dim = self.fourier.output_dim
        layers = []
        hidden = [64, 64, 64, 64]
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 最后一层 bias 设为 k_prior_log, 使初始输出 ≈ 先验
        last_linear = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        last_linear.bias.data.fill_(self.k_prior_log)
    
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: (B, 2) 归一化空间坐标 [x_norm, y_norm]
        Returns:
            log_k: (B, 1) 对数渗透率 (log mD)
        """
        feat = self.fourier(xy)
        log_k = self.net(feat)
        # Soft clamp 到合理范围
        log_k = self.log_k_min + (self.log_k_max - self.log_k_min) * torch.sigmoid(
            (log_k - self.k_prior_log) * 0.5 + 0.0  # 中心化
        )
        return log_k
    
    def get_k_mD(self, xy: torch.Tensor) -> torch.Tensor:
        """返回 k(x,y) in mD"""
        return torch.exp(self.forward(xy))
    
    def get_k_SI(self, xy: torch.Tensor) -> torch.Tensor:
        """返回 k(x,y) in m²"""
        return self.get_k_mD(xy) * 9.869233e-16
    
    def compute_tv_regularization(self, xy: torch.Tensor) -> torch.Tensor:
        """Total Variation 正则: |∇ log_k|₁"""
        xy = xy.requires_grad_(True)
        log_k = self.forward(xy)
        grad_k = torch.autograd.grad(
            log_k, xy,
            grad_outputs=torch.ones_like(log_k),
            create_graph=True, retain_graph=True
        )[0]
        return torch.mean(torch.abs(grad_k))
    
    def compute_laplacian_regularization(self, xy: torch.Tensor) -> torch.Tensor:
        """Laplacian 正则: ||∇² log_k||₂²"""
        xy = xy.requires_grad_(True)
        log_k = self.forward(xy)
        grad_k = torch.autograd.grad(
            log_k, xy,
            grad_outputs=torch.ones_like(log_k),
            create_graph=True, retain_graph=True
        )[0]
        # 二阶导
        d2k_dx2 = torch.autograd.grad(
            grad_k[:, 0:1], xy,
            grad_outputs=torch.ones_like(grad_k[:, 0:1]),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]
        d2k_dy2 = torch.autograd.grad(
            grad_k[:, 1:2], xy,
            grad_outputs=torch.ones_like(grad_k[:, 1:2]),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]
        laplacian = d2k_dx2 + d2k_dy2
        return torch.mean(laplacian ** 2)


class PINNNet(nn.Module):
    """
    PINN 增强网络 (v2)
    
    改进:
        1. Fourier Feature Encoding 输入层 (解决频谱偏置)
        2. 6×256 ResNet 风格主干 (增大容量)
        3. tanh 参数化压力 (替代 sigmoid, 避免梯度消失)
        4. Sw bounded-tanh v2.1 (Sw∈(0.12,0.88), 物理可行域+梯度安全)
        5. 可选 k(x,y) 子网络
    
    输入: [x, y, t] (归一化后)
    输出: [p, Sw] (物理有界)
    """
    
    def __init__(self, config: dict):
        """
        初始化 PINN 网络
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config
        
        # 模型配置
        arch = config.get('model', {}).get('architecture', {})
        hidden_layers = arch.get('hidden_layers', [256, 256, 256, 256, 256, 256])
        activation_name = arch.get('activation', 'tanh')
        use_layernorm = arch.get('use_layernorm', True)
        self.use_fourier = arch.get('use_fourier', True)
        n_freq = arch.get('fourier_n_freq', 32)
        fourier_sigma = arch.get('fourier_sigma', 1.0)
        self.use_residual = arch.get('use_residual', True)
        dropout_rate = arch.get('dropout', 0.0)
        
        # 物理范围
        mk_formation = config.get('mk_formation', {})
        p_avg = mk_formation.get('avg_pressure_MPa', {})
        if isinstance(p_avg, dict):
            p_bounds = p_avg.get('bounds', [30.0, 90.0])
            p_init = p_avg.get('value', 76.0)
        else:
            p_bounds = [30.0, 90.0]
            p_init = 76.0
        
        self.p_min = p_bounds[0]
        self.p_max = p_bounds[1]
        self.p_ref = p_init
        self.p_scale = (p_bounds[1] - p_bounds[0]) / 2.0  # tanh 范围
        self.p_center = (p_bounds[1] + p_bounds[0]) / 2.0
        
        # ★ Sw bounded-tanh v3.14 (SY9 测井数据修正版)
        # ════════════════════════════════════════════════════════
        # v3.4: 范围 (0.20, 0.46), sw_init=0.26 (SY13 Swc, 错误用于 SY9)
        # v3.14 修正: SY9 测井解释 sw_init≈0.15 (附表8 加权平均)
        #   范围扩展到 (0.05, 0.45), 中心=0.25, scale=0.20
        #   Sw=0.15 处: tanh=−0.50, dSw/dx=0.20*(1-0.25)=0.15 (健康梯度)
        #   Sw=0.35 处: tanh=+0.50, dSw/dx=0.15 (水侵缓升仍有梯度)
        #   tanh 饱和时: Sw_max=0.45 (物理合理, 气藏 Sw 不应更高)
        # ════════════════════════════════════════════════════════
        # 端点参数: Corey 模型仍用 SY13 (附表7), 但初始条件用 SY9 (附表8)
        self.Swc = 0.26      # 束缚水饱和度 (Corey 端点, SY13 附表7)
        self.Sgr = 0.062     # 残余气饱和度
        self.sw_init = 0.15  # v3.14: SY9 测井解释 (附表8)

        self.sw_center = 0.25    # v3.14: (0.05+0.45)/2
        self.sw_scale = 0.20     # v3.14: (0.45−0.05)/2
        # 使初始 Sw ≈ 0.15: tanh(bias) = (0.15−0.25)/0.20 = −0.50
        # bias = arctanh(−0.50) ≈ −0.549
        self.sw_init_bias = math.atanh(
            (self.sw_init - self.sw_center) / self.sw_scale
        )
        
        # --- Fourier Feature Encoding ---
        input_dim = 3  # x, y, t
        if self.use_fourier:
            self.fourier = FourierFeatureEncoding(
                input_dim=input_dim, n_freq=n_freq, sigma=fourier_sigma
            )
            encoded_dim = self.fourier.output_dim
        else:
            self.fourier = None
            encoded_dim = input_dim
        
        # 激活函数
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
        }
        Act = activations.get(activation_name, nn.Tanh)
        
        # --- 主干网络 ---
        # 输入映射层
        self.input_proj = nn.Linear(encoded_dim, hidden_layers[0])
        self.input_norm = nn.LayerNorm(hidden_layers[0]) if use_layernorm else nn.Identity()
        self.input_act = Act()
        
        # 残差块 (每两层一个)
        if self.use_residual and len(hidden_layers) >= 2:
            blocks = []
            for i in range(0, len(hidden_layers) - 1, 2):
                w = hidden_layers[i]
                blocks.append(ResidualBlock(w, Act(), use_layernorm))
            self.res_blocks = nn.ModuleList(blocks)
        else:
            # 普通 MLP
            layers = []
            in_features = hidden_layers[0]
            for h in hidden_layers[1:]:
                layers.append(nn.Linear(in_features, h))
                if use_layernorm:
                    layers.append(nn.LayerNorm(h))
                layers.append(Act())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                in_features = h
            self.res_blocks = None
            self.mlp_body = nn.Sequential(*layers) if layers else nn.Identity()
        
        # 输出层 (无激活, 后续手动加物理约束)
        output_width = hidden_layers[-1] if hidden_layers else encoded_dim
        self.output_head = nn.Linear(output_width, 2)  # [p_raw, sw_raw]
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化, 针对 tanh/sigmoid 输出优化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 输出层特殊初始化
        # p_raw: tanh(0)=0 → p=p_center ≈ 60 MPa
        # sw_raw: tanh(bias) → Sw ≈ 0.15 (SY9 测井, v3.14)
        nn.init.zeros_(self.output_head.weight)
        self.output_head.bias.data[0] = 0.0
        self.output_head.bias.data[1] = self.sw_init_bias
    
    def forward(self, xyt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            xyt: shape (B, 3) — [x_norm, y_norm, t_norm]
        
        Returns:
            p: shape (B, 1) — 压力 (MPa), 范围 [p_min, p_max]
            sw: shape (B, 1) — 含水饱和度, 范围 [0.05, 0.50] (v3.14 bounded-tanh + clamp)
        """
        # Fourier 编码
        if self.fourier is not None:
            h = self.fourier(xyt)
        else:
            h = xyt
        
        # 输入映射
        h = self.input_act(self.input_norm(self.input_proj(h)))
        
        # 主干
        if self.res_blocks is not None:
            for block in self.res_blocks:
                h = block(h)
        else:
            h = self.mlp_body(h)
        
        # 输出
        raw = self.output_head(h)  # (B, 2)
        
        p_raw = raw[:, 0:1]
        sw_raw = raw[:, 1:2]
        
        # --- 物理约束输出 ---
        p = self.p_center + self.p_scale * torch.tanh(p_raw)
        sw = self.sw_center + self.sw_scale * torch.tanh(sw_raw)
        # 数值稳定：防止极端值导致 PVT/相渗或梯度 NaN（clamp 在合理物理范围内）
        # v3.4: Sw 上界从 0.95 收紧到 0.50 (M4 气藏基线, Sw 不应超过边水前缘阈值)
        # 这从架构层面杜绝 Sw 漂移到 tanh 上界 0.88 的问题
        p = torch.clamp(p, 1.0, 150.0)
        sw = torch.clamp(sw, 0.05, 0.50)  # v3.14: 下界从Swc(0.26)改为0.05, 允许SY9 sw_init=0.15
        
        return p, sw
    
    def forward_with_grad(self, xyt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播并计算物理所需的梯度（用于 PDE 残差）
        
        Args:
            xyt: shape (B, 3) — [x_norm, y_norm, t_norm], requires_grad=True
        
        Returns:
            dict with:
                'p', 'sw': 输出
                'dp_dx', 'dp_dy', 'dp_dt': 压力梯度
                'dsw_dx', 'dsw_dy', 'dsw_dt': 饱和度梯度
        """
        xyt = xyt.requires_grad_(True)
        p, sw = self.forward(xyt)
        
        # 对 p 求梯度
        grad_p = torch.autograd.grad(
            p, xyt,
            grad_outputs=torch.ones_like(p),
            create_graph=True, retain_graph=True
        )[0]
        dp_dx = grad_p[:, 0:1]
        dp_dy = grad_p[:, 1:2]
        dp_dt = grad_p[:, 2:3]
        
        # 对 sw 求梯度
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
    
    def count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
