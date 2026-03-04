"""
RAR: Residual-based Adaptive Refinement (残差驱动自适应采样)
============================================================
每隔固定步数评估 PDE 残差，从候选点中挑选残差最大的点加入 collocation 集。

核心思路:
    1. 维护一个"候选池" (从原始配点网格采样)
    2. 每 rar_interval 步，在候选池上计算 PDE 残差
    3. 选取残差最大的 top-K 点加入 active collocation 集
    4. 重复直到达到最大点数

特别适合:
    - 井周高残差区域 (压力梯度陡)
    - 饱和度前缘 (Sw 变化剧烈)

参考: DeepXDE RAR (arXiv:2003.02751)

使用方法:
    rar = RARSampler(config, base_sampler)
    for step in range(max_steps):
        x_pde = rar.sample(step, model, loss_fn)  # 返回增强的配点集
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, Tuple

try:
    import torch
except ImportError:
    raise ImportError("rar_sampler 需要 PyTorch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger


class RARSampler:
    """
    残差自适应加点采样器
    
    Args:
        config: 全局配置
        base_sampler: 基础 PINNSampler 实例
    """
    
    def __init__(self, config: dict, base_sampler):
        self.config = config
        self.base_sampler = base_sampler
        self.logger = setup_logger('RARSampler')
        
        m5_cfg = config.get('m5_config', {})
        rar_cfg = m5_cfg.get('rar', {})
        
        # RAR 参数
        # 候选池从 5000 缩小到 2000: 减少每次 refine 的计算开销 (~60%),
        # 因为 compute_residuals 需要二阶导, 成本与候选数线性相关.
        # 2000 个候选已足够覆盖域内关键区域 (井周 + 前缘).
        # 频率从 500 步缩小到 300 步: 更早开始加点, 在 Stage B 即可
        # 引导配点向高残差区域集中, 加速 PDE 收敛.
        self.enable = rar_cfg.get('enable', True)
        self.interval = rar_cfg.get('interval', 300)    # 每 N 步执行一次
        self.n_candidates = rar_cfg.get('n_candidates', 2000)  # 候选池大小
        self.n_add = rar_cfg.get('n_add', 200)  # 每次加入的点数
        self.max_total = rar_cfg.get('max_total', 8000)  # 最大总点数
        
        # 已加入的高残差点 (归一化坐标)
        self.rar_points: Optional[np.ndarray] = None  # (M, 3) [x_n, y_n, t_n]
        self.rar_h_grad_gx: Optional[np.ndarray] = None
        self.rar_h_grad_gy: Optional[np.ndarray] = None
        
        self._n_rar_added = 0
        
        self.logger.info(
            f"RARSampler: enable={self.enable}, interval={self.interval}, "
            f"n_candidates={self.n_candidates}, n_add={self.n_add}, "
            f"max_total={self.max_total}"
        )
    
    def should_refine(self, step: int) -> bool:
        """判断当前步是否应该执行 RAR 加点"""
        if not self.enable:
            return False
        if step < self.interval:
            return False
        if step % self.interval != 0:
            return False
        if self._n_rar_added >= self.max_total:
            return False
        return True
    
    @torch.no_grad()
    def compute_residuals(self,
                          model: torch.nn.Module,
                          x_candidates: torch.Tensor,
                          h_grad: Optional[Dict[str, torch.Tensor]] = None,
                          device: str = 'cpu'
                          ) -> torch.Tensor:
        """
        在候选点上计算 PDE 残差 (用 forward_with_grad 代替 no_grad)
        
        注意: 需要梯度计算所以这里不能完全 no_grad
        """
        # 暂时开启梯度
        x_candidates = x_candidates.requires_grad_(True)
        
        with torch.enable_grad():
            grads = model.forward_with_grad(x_candidates)
            
            dp_dx = grads['dp_dx']
            dp_dy = grads['dp_dy']
            dp_dt = grads['dp_dt']
            xyt = grads['xyt']
            
            # 二阶导
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
        
        # 简化残差 (与 losses.py 的 PDE 形式一致)
        residual = dp_dt - d2p_dx2 - d2p_dy2
        
        if h_grad is not None:
            gx = h_grad.get('gx')
            gy = h_grad.get('gy')
            if gx is not None and gy is not None:
                residual = residual - gx * dp_dx - gy * dp_dy
        
        return residual.abs().squeeze(-1)  # (N,)
    
    def refine(self,
               step: int,
               model: torch.nn.Module,
               device: str = 'cpu'
               ) -> int:
        """
        执行一次 RAR 加点
        
        Args:
            step: 当前训练步数
            model: PINN 模型
            device: 计算设备
            
        Returns:
            本次新增的点数
        """
        model.eval()
        
        # 生成候选点
        x_cand_np = self.base_sampler.sample_domain(
            self.n_candidates, seed=step + 10000
        )
        gx_np, gy_np = self.base_sampler.get_last_h_grad()
        
        x_cand = torch.from_numpy(x_cand_np).float().to(device)
        
        h_grad = None
        if gx_np is not None:
            h_grad = {
                'gx': torch.from_numpy(gx_np).float().to(device).unsqueeze(-1),
                'gy': torch.from_numpy(gy_np).float().to(device).unsqueeze(-1),
            }
        
        # 计算残差
        residuals = self.compute_residuals(model, x_cand, h_grad, device)
        
        # 选取 top-K
        n_add = min(self.n_add, self.max_total - self._n_rar_added)
        if n_add <= 0:
            return 0
        
        _, top_indices = torch.topk(residuals, min(n_add, len(residuals)))
        top_indices = top_indices.cpu().numpy()
        
        new_points = x_cand_np[top_indices]
        new_gx = gx_np[top_indices] if gx_np is not None else None
        new_gy = gy_np[top_indices] if gy_np is not None else None
        
        # 加入 RAR 点集
        if self.rar_points is None:
            self.rar_points = new_points
            self.rar_h_grad_gx = new_gx
            self.rar_h_grad_gy = new_gy
        else:
            self.rar_points = np.concatenate([self.rar_points, new_points], axis=0)
            if new_gx is not None and self.rar_h_grad_gx is not None:
                self.rar_h_grad_gx = np.concatenate([self.rar_h_grad_gx, new_gx], axis=0)
                self.rar_h_grad_gy = np.concatenate([self.rar_h_grad_gy, new_gy], axis=0)
        
        self._n_rar_added += len(top_indices)
        
        # 日志
        res_stats = residuals.cpu().numpy()
        self.logger.info(
            f"[RAR step {step}] 加入 {len(top_indices)} 个高残差点, "
            f"总 RAR 点数: {self._n_rar_added}, "
            f"残差统计: mean={res_stats.mean():.4e}, "
            f"max={res_stats.max():.4e}, median={np.median(res_stats):.4e}"
        )
        
        model.train()
        return len(top_indices)
    
    def get_augmented_domain_points(self,
                                    N_base: int,
                                    seed: int = 0
                                    ) -> Tuple[np.ndarray,
                                               Optional[np.ndarray],
                                               Optional[np.ndarray]]:
        """
        获取增强后的域内配点（基础采样 + RAR 加点）
        
        Args:
            N_base: 基础采样点数
            seed: 随机种子
            
        Returns:
            (xyt_combined, gx_combined, gy_combined)
        """
        # 基础采样
        x_base = self.base_sampler.sample_domain(N_base, seed=seed)
        gx_base, gy_base = self.base_sampler.get_last_h_grad()
        
        if self.rar_points is None or len(self.rar_points) == 0:
            return x_base, gx_base, gy_base
        
        # 合并
        x_combined = np.concatenate([x_base, self.rar_points], axis=0)
        
        gx_combined = None
        gy_combined = None
        if gx_base is not None and self.rar_h_grad_gx is not None:
            gx_combined = np.concatenate([gx_base, self.rar_h_grad_gx], axis=0)
            gy_combined = np.concatenate([gy_base, self.rar_h_grad_gy], axis=0)
        
        return x_combined, gx_combined, gy_combined
    
    def get_stats(self) -> Dict[str, int]:
        """返回 RAR 统计信息"""
        return {
            'n_rar_points': self._n_rar_added,
            'total_points': (len(self.rar_points) if self.rar_points is not None else 0),
        }
