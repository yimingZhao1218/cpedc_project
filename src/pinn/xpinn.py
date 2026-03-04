"""
XPINN / APINN 域分解模块
==========================
为缝洞型强非均质气藏提供训练稳定性增强。

方案 A: XPINN (硬域分解)
    - 将空间域划分为子域，各子域独立网络
    - 界面施加连续/守恒匹配条件
    - 参考: XPINN (Jagtap et al., 2020)

方案 B: APINN (软分区 gating)
    - 门控网络实现可训练软域分解
    - 不需要显式划分子域边界
    - 参考: APINN (Li et al., 2023)

两种方案通过配置开关切换，不破坏主训练流程。

使用方法:
    xpinn = create_domain_decomposition(config, base_model)
    p, sw = xpinn(xyt)  # 接口与 PINNNet 完全兼容
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
    raise ImportError("xpinn 需要 PyTorch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger
from pinn.model import PINNNet


class SubdomainNet(nn.Module):
    """
    子域网络 (用于 XPINN)
    与 PINNNet 结构相同但独立权重
    """
    
    def __init__(self, config: dict, subdomain_id: int = 0):
        super().__init__()
        self.subdomain_id = subdomain_id
        
        arch = config.get('model', {}).get('architecture', {})
        hidden_layers = arch.get('hidden_layers', [128, 128, 128, 128])
        # 子域网络可以更小
        m6_cfg = config.get('m6_config', {}).get('xpinn', {})
        scale = m6_cfg.get('subnet_scale', 0.75)
        hidden_layers = [max(16, int(h * scale)) for h in hidden_layers]
        
        use_layernorm = arch.get('use_layernorm', True)
        
        mk = config.get('mk_formation', {})
        p_avg = mk.get('avg_pressure_MPa', {})
        p_bounds = p_avg.get('bounds', [30.0, 90.0]) if isinstance(p_avg, dict) else [30.0, 90.0]
        self.p_min = p_bounds[0]
        self.p_max = p_bounds[1]
        
        layers = []
        in_dim = 3
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, xyt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(xyt)
        p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(raw[:, 0:1])
        sw = torch.sigmoid(raw[:, 1:2])
        return p, sw


class XPINNModel(nn.Module):
    """
    XPINN: 硬域分解 PINN
    
    将空间域分为 n_subdomains 个子域，每个子域一个独立网络。
    在子域界面施加连续性匹配条件。
    
    域划分策略: 基于 x 坐标均匀划分 (可扩展为 Voronoi/自适应)
    """
    
    def __init__(self, config: dict, n_subdomains: int = 2):
        super().__init__()
        self.logger = setup_logger('XPINN')
        self.n_subdomains = n_subdomains
        self.config = config
        
        # 子域网络
        self.subnets = nn.ModuleList([
            SubdomainNet(config, i) for i in range(n_subdomains)
        ])
        
        # 域划分边界 (归一化坐标空间 [-1, 1])
        self.boundaries = torch.linspace(-1, 1, n_subdomains + 1)
        
        # 界面匹配权重
        m6_cfg = config.get('m6_config', {}).get('xpinn', {})
        self.interface_weight = m6_cfg.get('interface_weight', 10.0)
        
        # 物理范围 (同 PINNNet)
        mk = config.get('mk_formation', {})
        p_avg = mk.get('avg_pressure_MPa', {})
        p_bounds = p_avg.get('bounds', [30.0, 90.0]) if isinstance(p_avg, dict) else [30.0, 90.0]
        self.p_min = p_bounds[0]
        self.p_max = p_bounds[1]
        
        self.logger.info(
            f"XPINN 初始化: {n_subdomains} 子域, "
            f"边界={self.boundaries.tolist()}, "
            f"interface_weight={self.interface_weight}"
        )
    
    def _get_subdomain_idx(self, x_norm: torch.Tensor) -> torch.Tensor:
        """确定每个点所属的子域索引"""
        idx = torch.zeros(x_norm.shape[0], dtype=torch.long, device=x_norm.device)
        for i in range(self.n_subdomains):
            mask = (x_norm >= self.boundaries[i]) & (x_norm < self.boundaries[i + 1])
            idx[mask.squeeze()] = i
        return idx
    
    def forward(self, xyt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: 根据 x 坐标分配到对应子域网络
        
        Args:
            xyt: (B, 3) [x_norm, y_norm, t_norm]
        Returns:
            p: (B, 1), sw: (B, 1)
        """
        x_coord = xyt[:, 0:1]
        subdomain_idx = self._get_subdomain_idx(x_coord)
        
        p_out = torch.zeros(xyt.shape[0], 1, device=xyt.device)
        sw_out = torch.zeros(xyt.shape[0], 1, device=xyt.device)
        
        for i in range(self.n_subdomains):
            mask = subdomain_idx == i
            if mask.any():
                p_i, sw_i = self.subnets[i](xyt[mask])
                p_out[mask] = p_i
                sw_out[mask] = sw_i
        
        return p_out, sw_out
    
    def forward_with_grad(self, xyt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """带梯度的前向传播 (兼容 PINNNet 接口)"""
        xyt = xyt.requires_grad_(True)
        p, sw = self.forward(xyt)
        
        grad_p = torch.autograd.grad(
            p, xyt, grad_outputs=torch.ones_like(p),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_sw = torch.autograd.grad(
            sw, xyt, grad_outputs=torch.ones_like(sw),
            create_graph=True, retain_graph=True
        )[0]
        
        return {
            'p': p, 'sw': sw,
            'dp_dx': grad_p[:, 0:1], 'dp_dy': grad_p[:, 1:2], 'dp_dt': grad_p[:, 2:3],
            'dsw_dx': grad_sw[:, 0:1], 'dsw_dy': grad_sw[:, 1:2], 'dsw_dt': grad_sw[:, 2:3],
            'xyt': xyt,
        }
    
    def interface_loss(self, n_points: int = 256) -> torch.Tensor:
        """
        界面连续性匹配损失
        在子域边界上要求相邻子域输出一致
        """
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        
        for i in range(self.n_subdomains - 1):
            x_bnd = self.boundaries[i + 1].item()
            
            # 在界面上采样
            y_rand = torch.rand(n_points, 1, device=device) * 2 - 1
            t_rand = torch.rand(n_points, 1, device=device)
            x_const = torch.full((n_points, 1), x_bnd, device=device)
            
            xyt_bnd = torch.cat([x_const, y_rand, t_rand], dim=1)
            
            # 两侧子域的输出
            p_left, sw_left = self.subnets[i](xyt_bnd)
            p_right, sw_right = self.subnets[i + 1](xyt_bnd)
            
            # 连续性
            loss = loss + torch.mean((p_left - p_right) ** 2)
            loss = loss + torch.mean((sw_left - sw_right) ** 2)
        
        return self.interface_weight * loss
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GatingNetwork(nn.Module):
    """门控网络: 输入 (x, y, t) 输出 n_experts 个 softmax 权重"""
    
    def __init__(self, n_experts: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_experts),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """返回 (B, n_experts) softmax 权重"""
        logits = self.net(xyt)
        return F.softmax(logits, dim=-1)


class APINNModel(nn.Module):
    """
    APINN: 软域分解 PINN (Auxiliary/Adaptive)
    
    用门控网络实现可训练的软分区:
        output = Σ_i g_i(x,y,t) · f_i(x,y,t)
    
    其中 g_i 是门控权重, f_i 是专家子网络
    """
    
    def __init__(self, config: dict, n_experts: int = 3):
        super().__init__()
        self.logger = setup_logger('APINN')
        self.n_experts = n_experts
        
        # 专家网络
        self.experts = nn.ModuleList([
            SubdomainNet(config, i) for i in range(n_experts)
        ])
        
        # 门控网络
        m6_cfg = config.get('m6_config', {}).get('apinn', {})
        gate_hidden = m6_cfg.get('gate_hidden', 32)
        self.gate = GatingNetwork(n_experts, hidden=gate_hidden)
        
        self.logger.info(f"APINN 初始化: {n_experts} 专家, gate_hidden={gate_hidden}")
    
    def forward(self, xyt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        软混合前向传播
        
        Args:
            xyt: (B, 3)
        Returns:
            p: (B, 1), sw: (B, 1)
        """
        weights = self.gate(xyt)  # (B, n_experts)
        
        p_out = torch.zeros(xyt.shape[0], 1, device=xyt.device)
        sw_out = torch.zeros(xyt.shape[0], 1, device=xyt.device)
        
        for i in range(self.n_experts):
            p_i, sw_i = self.experts[i](xyt)
            w_i = weights[:, i:i+1]  # (B, 1)
            p_out = p_out + w_i * p_i
            sw_out = sw_out + w_i * sw_i
        
        return p_out, sw_out
    
    def forward_with_grad(self, xyt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """带梯度的前向传播"""
        xyt = xyt.requires_grad_(True)
        p, sw = self.forward(xyt)
        
        grad_p = torch.autograd.grad(
            p, xyt, grad_outputs=torch.ones_like(p),
            create_graph=True, retain_graph=True
        )[0]
        
        grad_sw = torch.autograd.grad(
            sw, xyt, grad_outputs=torch.ones_like(sw),
            create_graph=True, retain_graph=True
        )[0]
        
        return {
            'p': p, 'sw': sw,
            'dp_dx': grad_p[:, 0:1], 'dp_dy': grad_p[:, 1:2], 'dp_dt': grad_p[:, 2:3],
            'dsw_dx': grad_sw[:, 0:1], 'dsw_dy': grad_sw[:, 1:2], 'dsw_dt': grad_sw[:, 2:3],
            'xyt': xyt,
        }
    
    def get_gate_distribution(self, xyt: torch.Tensor) -> torch.Tensor:
        """获取门控分布 (用于可视化)"""
        with torch.no_grad():
            return self.gate(xyt)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_domain_decomposition(config: dict) -> Optional[nn.Module]:
    """
    工厂函数: 根据配置创建域分解模型
    
    Args:
        config: 全局配置
        
    Returns:
        XPINN/APINN 模型, 或 None (不使用)
    """
    m6_cfg = config.get('m6_config', {})
    dd_mode = m6_cfg.get('domain_decomposition', 'none')
    
    if dd_mode == 'xpinn':
        n_sub = m6_cfg.get('xpinn', {}).get('n_subdomains', 2)
        return XPINNModel(config, n_subdomains=n_sub)
    
    elif dd_mode == 'apinn':
        n_exp = m6_cfg.get('apinn', {}).get('n_experts', 3)
        return APINNModel(config, n_experts=n_exp)
    
    else:
        return None
