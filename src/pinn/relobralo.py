"""
ReLoBRaLo: Relative Loss Balancing with Random Lookback
========================================================
自适应多目标损失权重平衡，专为 PINN 多损失项设计。

核心思想:
    - 跟踪每个损失项的历史指数移动平均
    - 根据当前损失与历史平均的比值（相对变化率）动态调整权重
    - 使用随机 lookback 引入探索性，避免陷入局部
    - 通过 temperature 参数控制权重分布的 "尖锐度"

参考: ReLoBRaLo (arXiv:2110.09813)

使用方法:
    relobralo = ReLoBRaLo(loss_names=['ic', 'bc', 'pde', 'qg', 'smooth_pwf'])
    for step in range(max_steps):
        losses = compute_losses(...)  # dict
        weights = relobralo.get_weights(losses, step)
        total = sum(w * losses[k] for k, w in weights.items())
"""

import math
import numpy as np
from typing import Dict, List, Optional

try:
    import torch
except ImportError:
    raise ImportError("relobralo 需要 PyTorch")


class ReLoBRaLo:
    """
    ReLoBRaLo 自适应损失权重平衡器
    
    Args:
        loss_names: 参与平衡的损失项名称列表
        temperature: softmax 温度参数 (τ)，越小权重越集中
        alpha: 指数移动平均衰减系数 (EMA)
        rho: 随机 lookback 概率 (random lookback probability)
        warmup_steps: 预热步数（预热期间使用均匀权重）
    """
    
    def __init__(self,
                 loss_names: List[str],
                 temperature: float = 1.0,
                 alpha: float = 0.999,
                 rho: float = 0.999,
                 warmup_steps: int = 100):
        self.loss_names = loss_names
        self.n_losses = len(loss_names)
        self.temperature = temperature
        self.alpha = alpha
        self.rho = rho
        self.warmup_steps = warmup_steps
        
        # 历史记录
        self.loss_history: Dict[str, List[float]] = {name: [] for name in loss_names}
        self.ema: Dict[str, float] = {name: 1.0 for name in loss_names}
        self.init_losses: Dict[str, float] = {}
        self.current_weights: Dict[str, float] = {name: 1.0 for name in loss_names}
        
        self._initialized = False
        self._step = 0
    
    def _softmax(self, values: List[float]) -> List[float]:
        """带温度的 softmax"""
        max_v = max(values)
        exp_v = [math.exp((v - max_v) / max(self.temperature, 1e-8)) for v in values]
        sum_exp = sum(exp_v)
        n = len(values)
        return [n * e / (sum_exp + 1e-12) for e in exp_v]
    
    def get_weights(self,
                    losses: Dict[str, float],
                    step: int
                    ) -> Dict[str, float]:
        """
        根据当前损失值计算自适应权重
        
        Args:
            losses: {loss_name: loss_value} 当前各项损失
            step: 当前训练步数
            
        Returns:
            {loss_name: weight} 自适应权重
        """
        self._step = step
        
        # --- 记录历史 ---
        for name in self.loss_names:
            val = losses.get(name, 0.0)
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.loss_history[name].append(val)
        
        # --- 初始化（第一步）---
        if not self._initialized:
            for name in self.loss_names:
                self.init_losses[name] = max(losses.get(name, 1.0), 1e-12)
                if isinstance(self.init_losses[name], torch.Tensor):
                    self.init_losses[name] = self.init_losses[name].item()
                self.ema[name] = self.init_losses[name]
            self._initialized = True
            return {name: 1.0 for name in self.loss_names}
        
        # --- 预热期间使用均匀权重 ---
        if step < self.warmup_steps:
            return {name: 1.0 for name in self.loss_names}
        
        # --- 更新 EMA ---
        for name in self.loss_names:
            val = losses.get(name, 0.0)
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.ema[name] = self.alpha * self.ema[name] + (1 - self.alpha) * max(val, 1e-12)
        
        # --- 随机 lookback ---
        # 以概率 rho 使用 EMA，以概率 (1-rho) 随机选取历史某步的值
        lookback_vals = {}
        for name in self.loss_names:
            if np.random.random() < self.rho or len(self.loss_history[name]) < 2:
                lookback_vals[name] = self.ema[name]
            else:
                # 随机选取历史某一步
                idx = np.random.randint(0, len(self.loss_history[name]))
                lookback_vals[name] = max(self.loss_history[name][idx], 1e-12)
        
        # --- 计算相对变化率 (v2: 对数归一化处理量级差异) ---
        # 先对损失做对数归一化: normalized = log(1 + L / L_init)
        # 再计算 ratio = normalized(L) / normalized(lookback)
        ratios = []
        for name in self.loss_names:
            val = losses.get(name, 0.0)
            if isinstance(val, torch.Tensor):
                val = val.item()
            
            # 对数归一化: 处理损失量级差异 (qg~10 vs IC~0.001)
            init_val = max(self.init_losses.get(name, 1.0), 1e-12)
            val_norm = math.log(1.0 + max(val, 0) / init_val)
            lookback_norm = math.log(1.0 + max(lookback_vals[name], 0) / init_val)
            
            ratio = max(val_norm, 1e-12) / max(lookback_norm, 1e-12)
            ratios.append(math.log(ratio + 1e-12))
        
        # --- Softmax 得到权重 ---
        weights_list = self._softmax(ratios)
        
        self.current_weights = {
            name: w for name, w in zip(self.loss_names, weights_list)
        }
        
        return self.current_weights.copy()
    
    def get_weight_summary(self) -> str:
        """获取权重摘要字符串"""
        parts = [f"{k}={v:.3f}" for k, v in self.current_weights.items()]
        return "ReLoBRaLo weights: " + ", ".join(parts)


class ManualLossBalancer:
    """
    手动/课程学习损失权重管理器（ReLoBRaLo 的简化替代）
    
    支持线性插值权重调度，与 M4 的 training_stages 兼容
    """
    
    def __init__(self, base_weights: Dict[str, float]):
        self.base_weights = base_weights.copy()
        self.current_weights = base_weights.copy()
    
    def get_weights(self,
                    losses: Dict[str, float] = None,
                    step: int = 0,
                    overrides: Dict[str, float] = None
                    ) -> Dict[str, float]:
        if overrides:
            self.current_weights = overrides.copy()
        return self.current_weights.copy()
    
    def get_weight_summary(self) -> str:
        parts = [f"{k}={v:.3f}" for k, v in self.current_weights.items()]
        return "Manual weights: " + ", ".join(parts)
