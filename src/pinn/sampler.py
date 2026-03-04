"""
M4 PINN 采样器
PINN 成败 50% 在采样策略

三类采样点:
    1. 域内配点 (collocation): 用于 PDE 残差
    2. 边界点 (boundary): 用于外边界定压/无流
    3. 初始点 (initial): 用于 t=0 初始压力/初始含水

策略:
    - 域内: 均匀 + 井周加密
    - 时间: 前期密集（学习初始→见水前沿）
    - 输出: 归一化坐标张量
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger, load_config


class PINNSampler:
    """PINN 采样器：生成域内/边界/初始采样点"""
    
    def __init__(self, config: Optional[dict] = None, config_path: str = 'config.yaml'):
        """
        初始化采样器
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        if config is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            config = load_config(str(project_root / config_path))
            for key, value in config['paths'].items():
                config['paths'][key] = str(project_root / value)
        
        self.config = config
        self.logger = setup_logger('PINNSampler')
        
        # 加载地质数据
        self._load_geo_data()
        # 加载生产数据时间范围
        self._load_time_range()
        # 计算归一化参数
        self._compute_normalization()
        # 构建厚度场插值器 + 预计算配点处 log-thickness 梯度
        self._build_thickness_field()
        
        self.logger.info(
            f"PINNSampler 初始化完成: "
            f"x=[{self.x_min:.0f}, {self.x_max:.0f}], "
            f"y=[{self.y_min:.0f}, {self.y_max:.0f}], "
            f"t=[0, {self.t_max:.0f}] days, "
            f"域内点={len(self.collocation_xy)}, 边界点={len(self.boundary_xy)}"
        )
    
    def _load_geo_data(self):
        """加载 M2 生成的地质域数据"""
        geo_path = self.config['paths']['geo_data']
        
        # 配点网格
        grid_file = os.path.join(geo_path, 'grids', 'collocation_grid.csv')
        grid_df = pd.read_csv(grid_file)
        self.collocation_xy = grid_df[['x', 'y']].values
        self.is_near_well = grid_df['is_near_well'].values if 'is_near_well' in grid_df else None
        
        # 边界点 — 优先使用M2生成的512等弧长采样点(更均匀), 回退到原始边界多边形
        boundary_pts_file = os.path.join(geo_path, 'grids', 'boundary_points.csv')
        if os.path.exists(boundary_pts_file):
            boundary_df = pd.read_csv(boundary_pts_file)
            self.logger.info(f"  使用M2等弧长边界采样点: {len(boundary_df)} 个")
        else:
            boundary_file = os.path.join(geo_path, 'boundary', 'model_boundary.csv')
            boundary_df = pd.read_csv(boundary_file)
            self.logger.info(f"  回退到原始边界多边形: {len(boundary_df)} 个顶点")
        self.boundary_xy = boundary_df[['x', 'y']].values
        
        # 厚度场
        thickness_file = os.path.join(geo_path, 'surfaces', 'mk_thickness.csv')
        thickness_df = pd.read_csv(thickness_file)
        self.thickness_xy = thickness_df[['x', 'y']].values
        self.thickness_h = thickness_df['z'].values  # h(x,y)
        
        # 井位坐标
        clean_path = self.config['paths']['clean_data']
        mk_file = os.path.join(clean_path, 'mk_interval_points.csv')
        mk_df = pd.read_csv(mk_file)
        self.well_xy = mk_df[['x_mid', 'y_mid']].values
        self.well_ids = mk_df['well_id'].values
        
        # 域范围
        self.x_min = self.collocation_xy[:, 0].min()
        self.x_max = self.collocation_xy[:, 0].max()
        self.y_min = self.collocation_xy[:, 1].min()
        self.y_max = self.collocation_xy[:, 1].max()
    
    def _load_time_range(self):
        """从 SY9 生产数据获取时间范围"""
        clean_path = self.config['paths']['clean_data']
        prod_file = os.path.join(clean_path, 'production_SY9.csv')
        prod_df = pd.read_csv(prod_file)
        
        self.t_max = float(prod_df['t_day'].max())
        self.production_data = prod_df
        self.logger.info(f"  生产数据时间范围: 0 ~ {self.t_max:.0f} 天, 共 {len(prod_df)} 条记录")
    
    def _compute_normalization(self):
        """计算归一化参数（将坐标映射到 [-1, 1]，时间映射到 [0, 1]）"""
        self.norm_params = {
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            't_max': self.t_max,
            'x_range': self.x_max - self.x_min,
            'y_range': self.y_max - self.y_min,
        }
    
    def normalize_xy(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """坐标归一化到 [-1, 1]"""
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        return x_norm, y_norm
    
    def normalize_t(self, t: np.ndarray) -> np.ndarray:
        """时间归一化到 [0, 1]"""
        return t / self.t_max
    
    def denormalize_xy(self, x_norm: np.ndarray, y_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """坐标反归一化"""
        x = (x_norm + 1.0) / 2.0 * (self.x_max - self.x_min) + self.x_min
        y = (y_norm + 1.0) / 2.0 * (self.y_max - self.y_min) + self.y_min
        return x, y
    
    def denormalize_t(self, t_norm: np.ndarray) -> np.ndarray:
        """时间反归一化"""
        return t_norm * self.t_max
    
    # ================================================================== #
    #                  厚度场 h(x,y) — 2.5D 核心
    # ================================================================== #
    def _build_thickness_field(self):
        """
        构建厚度场插值器，并在所有配点处预计算：
        - h(x,y):  厚度 (m)
        - gx, gy:  log-thickness 归一化梯度 = (1/h)(∂h/∂x_n), (1/h)(∂h/∂y_n)
        
        2.5D PDE 中 ∇·(h ∇p) = h ∇²p + (∇h)·(∇p)
        除以 h 后得到修正项: gx·(∂p/∂x_n), gy·(∂p/∂y_n)
        
        P0: 厚度统计用 nan-aware，梯度统计只在 finite 区域，禁止输出 NaN。
        """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
        
        # --- 仅用 finite 点构建插值器，避免 NaN 传播 ---
        valid_h = np.isfinite(self.thickness_h)
        xy_valid = self.thickness_xy[valid_h]
        h_valid = self.thickness_h[valid_h]
        if len(xy_valid) == 0:
            raise ValueError("厚度场无有效数据（全部 NaN）")
        self._h_linear = LinearNDInterpolator(xy_valid, h_valid)
        self._h_nearest = NearestNDInterpolator(xy_valid, h_valid)
        
        # --- 在配点处查询 h ---
        cx, cy = self.collocation_xy[:, 0], self.collocation_xy[:, 1]
        h_colloc = self._h_linear(cx, cy)
        nan_mask = np.isnan(h_colloc)
        if np.any(nan_mask):
            h_colloc[nan_mask] = self._h_nearest(cx[nan_mask], cy[nan_mask])
        
        # 强制 h > 0（审计检查）
        h_colloc = np.maximum(h_colloc, 1e-3)
        self.collocation_h = h_colloc
        
        # --- P0: 梯度在 regular grid 上计算，用填充后的 h，仅在 finite 区域统计 ---
        # 建立规则网格
        nx, ny = 80, 60
        xg = np.linspace(self.x_min, self.x_max, nx)
        yg = np.linspace(self.y_min, self.y_max, ny)
        Xg, Yg = np.meshgrid(xg, yg)
        dx_m = (self.x_max - self.x_min) / max(nx - 1, 1)
        dy_m = (self.y_max - self.y_min) / max(ny - 1, 1)
        
        h_grid = self._h_linear(Xg.ravel(), Yg.ravel()).reshape(Xg.shape)
        mask = np.isfinite(h_grid)
        h_fill = h_grid.copy()
        if np.any(~mask):
            h_fill[~mask] = self._h_nearest(Xg[~mask], Yg[~mask])
        still_nan = np.isnan(h_fill)
        if np.any(still_nan):
            fill_val = np.nanmean(h_valid) if np.any(np.isfinite(h_valid)) else 90.0
            h_fill[still_nan] = fill_val
        
        gx_phys, gy_phys = np.gradient(h_fill, dx_m, dy_m)
        half_dx = (self.x_max - self.x_min) / 2.0
        half_dy = (self.y_max - self.y_min) / 2.0
        gx_grid = (gx_phys * half_dx) / np.maximum(h_fill, 1e-3)
        gy_grid = (gy_phys * half_dy) / np.maximum(h_fill, 1e-3)
        
        gx_max = float(np.nanmax(np.abs(gx_grid[mask]))) if np.any(mask) else 0.0
        gy_max = float(np.nanmax(np.abs(gy_grid[mask]))) if np.any(mask) else 0.0
        
        # 将梯度插值到配点
        gx_colloc = griddata(
            (Xg.ravel(), Yg.ravel()), gx_grid.ravel(),
            (cx, cy), method='linear', fill_value=0.0
        )
        gy_colloc = griddata(
            (Xg.ravel(), Yg.ravel()), gy_grid.ravel(),
            (cx, cy), method='linear', fill_value=0.0
        )
        gx_colloc = np.nan_to_num(gx_colloc, nan=0.0, posinf=0.0, neginf=0.0)
        gy_colloc = np.nan_to_num(gy_colloc, nan=0.0, posinf=0.0, neginf=0.0)
        self.collocation_gx = gx_colloc.astype(np.float32)
        self.collocation_gy = gy_colloc.astype(np.float32)
        
        # --- P0: 厚度统计 nan-aware，禁止输出 NaN ---
        self.h_min = float(np.nanmin(self.thickness_h))
        self.h_max = float(np.nanmax(self.thickness_h))
        self.h_mean = float(np.nanmean(self.thickness_h))
        nan_ratio = float(np.isnan(self.thickness_h).mean())
        
        if not np.isfinite(self.h_min):
            self.h_min = float(np.nanmin(h_valid))
        if not np.isfinite(self.h_max):
            self.h_max = float(np.nanmax(h_valid))
        if not np.isfinite(self.h_mean):
            self.h_mean = float(np.nanmean(h_valid))
        
        n_oor = int(np.sum(nan_mask))
        r_oor = n_oor / len(cx) * 100
        
        # P1: 域外点标记，供 PDE 过滤（可选）
        self.collocation_is_oor = nan_mask  # True = 厚度插值失败，用了最近邻回退
        
        self.logger.info(
            f"  厚度场: h ∈ [{self.h_min:.1f}, {self.h_max:.1f}] m, "
            f"mean={self.h_mean:.1f} m, nan_ratio={nan_ratio:.4f}, 网格={len(self.thickness_h)} 点"
        )
        self.logger.info(
            f"  配点处: h ∈ [{h_colloc.min():.1f}, {h_colloc.max():.1f}] m, "
            f"|gx|_max={gx_max:.4f}, |gy|_max={gy_max:.4f}"
        )
        if n_oor > 0:
            self.logger.warning(
                f"  厚度场域外率: {n_oor}/{len(cx)} = {r_oor:.2f}% "
                f"(已用最近邻回退)"
            )
    
    def get_last_h_grad(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        返回最近一次 sample_domain 所采样点的 log-thickness 梯度 (gx, gy)。
        在 _sample_batch 后调用，供 PDE 残差使用。
        """
        return (getattr(self, '_last_gx', None),
                getattr(self, '_last_gy', None))
    
    def get_last_pde_mask(self) -> Optional[np.ndarray]:
        """
        返回最近一次 sample_domain 所采样点的 PDE 有效掩码。
        1.0 = 域内（厚度插值成功），0.0 = 域外（最近邻回退），PDE loss 可据此过滤。
        """
        is_oor = getattr(self, '_last_is_oor', None)
        if is_oor is None:
            return None
        return (1.0 - is_oor.astype(np.float32))
    
    # ================================================================== #
    #                        采样方法
    # ================================================================== #
    def sample_domain(self, N: int, seed: int = None,
                      training_progress: float = 0.0,
                      t_min_raw: Optional[float] = None,
                      t_max_raw: Optional[float] = None) -> np.ndarray:
        """
        域内配点采样（用于 PDE 残差）
        策略: 井周加密混合采样 + 时间分布随训练阶段切换
        
        改进:
            - 前期 (progress < 0.5): Beta(1.5, 3.0) 偏向早期
            - 后期 (progress >= 0.5): 混合分布 50%均匀 + 50% Beta(1.5, 2.0)
              确保后期压降也被学到
        
        Args:
            N: 采样点数
            seed: 随机种子
            training_progress: 训练进度 [0, 1]
            
        Returns:
            shape (N, 3) 的归一化坐标 [x_norm, y_norm, t_norm]
        """
        rng = np.random.RandomState(seed)
        
        # 井周加密比例（从 config 读取，默认 0.3）
        near_well_ratio = (self.config.get('m4_config', {})
                           .get('sampling', {})
                           .get('near_well_ratio', 0.30))
        
        n_pts = len(self.collocation_xy)
        
        if (self.is_near_well is not None
                and near_well_ratio > 0
                and np.any(self.is_near_well)):
            n_near = int(N * near_well_ratio)
            n_uniform = N - n_near
            
            idx_uniform = rng.randint(0, n_pts, size=n_uniform)
            near_indices = np.where(self.is_near_well.astype(bool))[0]
            idx_near = rng.choice(near_indices, size=n_near, replace=True)
            indices = np.concatenate([idx_uniform, idx_near])
        else:
            indices = rng.randint(0, n_pts, size=N)
        
        xy = self.collocation_xy[indices]
        
        # 缓存该批次对应的 log-thickness 梯度及 PDE 有效掩码（P1 域外过滤）
        if hasattr(self, 'collocation_gx'):
            self._last_gx = self.collocation_gx[indices]
            self._last_gy = self.collocation_gy[indices]
        if hasattr(self, 'collocation_is_oor'):
            self._last_is_oor = self.collocation_is_oor[indices]
        else:
            self._last_is_oor = np.zeros(len(indices), dtype=bool)
        
        # --- 时间采样: 随训练阶段切换分布 ---
        samp_cfg = self.config.get('m4_config', {}).get('sampling', {})
        t_lo = 0.0 if t_min_raw is None else float(t_min_raw)
        t_hi = self.t_max if t_max_raw is None else float(t_max_raw)
        t_lo = float(np.clip(t_lo, 0.0, self.t_max))
        t_hi = float(np.clip(t_hi, 0.0, self.t_max))
        if t_hi <= t_lo:
            eps = max(self.t_max * 1e-6, 1e-6)
            t_hi = min(self.t_max, t_lo + eps)
            t_lo = max(0.0, min(t_lo, t_hi - 1e-6))
        t_span = max(t_hi - t_lo, 1e-6)
        
        if training_progress < 0.5:
            # 前期: Beta(1.5, 3.0) 偏向早期
            beta_a = samp_cfg.get('time_beta_a', 1.5)
            beta_b = samp_cfg.get('time_beta_b', 3.0)
            u = rng.beta(beta_a, beta_b, size=N)
        else:
            # 后期: 混合分布 — 50% 均匀 + 50% Beta(1.5, 2.0)
            n_uniform_t = N // 2
            n_beta_t = N - n_uniform_t
            u_uniform = rng.uniform(0.0, 1.0, size=n_uniform_t)
            u_beta = rng.beta(1.5, 2.0, size=n_beta_t)
            u = np.concatenate([u_uniform, u_beta])
            rng.shuffle(u)
        
        # 归一化
        x_norm, y_norm = self.normalize_xy(xy[:, 0], xy[:, 1])
        t_raw = t_lo + u * t_span
        t_norm = self.normalize_t(t_raw)
        
        return np.stack([x_norm, y_norm, t_norm], axis=-1).astype(np.float32)
    
    def sample_boundary(self, N: int, seed: int = None) -> np.ndarray:
        """
        边界采样（用于 BC 损失）
        策略: 从边界点随机抽取 + 随机时间
        
        Args:
            N: 采样点数
            
        Returns:
            shape (N, 3) 的归一化坐标 [x_norm, y_norm, t_norm]
        """
        rng = np.random.RandomState(seed)
        
        n_bnd = len(self.boundary_xy)
        indices = rng.randint(0, n_bnd, size=N)
        xy = self.boundary_xy[indices]
        
        # 时间均匀分布
        t_raw = rng.uniform(0, self.t_max, size=N)
        
        x_norm, y_norm = self.normalize_xy(xy[:, 0], xy[:, 1])
        t_norm = self.normalize_t(t_raw)
        
        return np.stack([x_norm, y_norm, t_norm], axis=-1).astype(np.float32)
    
    def sample_initial(self, N: int, seed: int = None) -> np.ndarray:
        """
        初始条件采样（t=0）
        
        Args:
            N: 采样点数
            
        Returns:
            shape (N, 3) 的归一化坐标 [x_norm, y_norm, 0]
        """
        rng = np.random.RandomState(seed)
        
        n_pts = len(self.collocation_xy)
        indices = rng.randint(0, n_pts, size=N)
        xy = self.collocation_xy[indices]
        
        x_norm, y_norm = self.normalize_xy(xy[:, 0], xy[:, 1])
        t_norm = np.zeros(N, dtype=np.float32)
        
        return np.stack([x_norm, y_norm, t_norm], axis=-1).astype(np.float32)
    
    def sample_well_data(self, well_id: str = 'SY9') -> Dict:
        """
        井点数据采样（用于数据锚点损失）
        
        Args:
            well_id: 井号
            
        Returns:
            {'xyt': (N_data, 3), 'p_obs': (N_data,), 'qg_obs': (N_data,), 'qw_obs': (N_data,),
             't_days': (N_data,), 'p_obs_full': (N_full,), 'valid_mask': (N_full,), 'missing_runs': list,
             't_days_full': (N_full,), 'xyt_full': (N_full, 3), 'nan_ratio': float, ...}
        """
        # 明确指定观测列：井口压力
        obs_pressure_col = 'tubing_p_avg'
        well_mask = self.well_ids == well_id
        if not np.any(well_mask):
            self.logger.warning(f"未找到井 {well_id} 的坐标数据")
            return {}
        
        well_x = self.well_xy[well_mask][0, 0]
        well_y = self.well_xy[well_mask][0, 1]
        
        prod = self.production_data
        if obs_pressure_col not in prod.columns:
            obs_pressure_col = 'casing_p_avg'
        
        t_days_full = prod['t_day'].values.astype(np.float32)
        p_obs_full = np.asarray(prod[obs_pressure_col].values, dtype=np.float64)  # 含 NaN，长度 N_full
        qg_raw = prod['qg_m3d'].values.astype(np.float32) \
            if 'qg_m3d' in prod.columns else np.zeros_like(t_days_full, dtype=np.float32)
        qw_full = prod['qw_td'].values.astype(np.float32) \
            if 'qw_td' in prod.columns else np.zeros_like(t_days_full, dtype=np.float32)

        qg_is_finite = np.isfinite(qg_raw) & ~np.isnan(qg_raw)
        # 识别关井：生产时间=0 且 qg 缺失 → 视为有效零产
        prod_hours_col = '生产时间_(H)' if '生产时间_(H)' in prod.columns else None
        if prod_hours_col is not None:
            prod_hours_full = pd.to_numeric(prod[prod_hours_col], errors='coerce').values.astype(np.float32)
            shutin_from_hours = (~qg_is_finite) & np.isfinite(prod_hours_full) & (prod_hours_full <= 0.0)
            shutin_schedule_full = np.isfinite(prod_hours_full) & (prod_hours_full <= 0.0)
        else:
            prod_hours_full = np.full_like(t_days_full, 24.0, dtype=np.float32)
            shutin_from_hours = np.zeros_like(qg_is_finite, dtype=bool)
            shutin_schedule_full = np.isfinite(qg_raw) & (qg_raw <= 1.0)
        # v16: 归一化生产时间 [0,24]h → [0,1]，用于 pwf_net 工况输入
        prod_hours_full_safe = np.nan_to_num(prod_hours_full, nan=0.0)
        prod_hours_norm_full = np.clip(prod_hours_full_safe / 24.0, 0.0, 1.0).astype(np.float32)
        # v17: 套压作为工况输入，归一化到 [0,1]
        if 'casing_p_avg' in prod.columns:
            casing_raw = pd.to_numeric(prod['casing_p_avg'], errors='coerce').values.astype(np.float32)
            casing_safe = np.nan_to_num(casing_raw, nan=0.0)
            casing_norm_full = np.clip(casing_safe / 60.0, 0.0, 1.0).astype(np.float32)
        else:
            casing_norm_full = np.full_like(t_days_full, 0.5, dtype=np.float32)
        qg_valid_mask = qg_is_finite | shutin_from_hours
        qg_filled_full = np.where(np.isfinite(qg_raw), qg_raw, 0.0).astype(np.float32)

        p_valid_mask = np.isfinite(p_obs_full) & ~np.isnan(p_obs_full)
        combined_mask = p_valid_mask | qg_valid_mask

        # 向后兼容: valid_mask 仍表示压力有效掩码（用于压力缺失段统计）
        valid_mask = p_valid_mask
        
        total_points = len(p_obs_full)
        valid_points = int(np.sum(valid_mask))
        nan_ratio = 1.0 - (valid_points / total_points) if total_points else 0.0
        
        # 缺失区间：(start_idx, end_idx, start_date, end_date, length_days)
        missing_runs = self._compute_missing_runs(valid_mask, prod, t_days_full)
        
        self.logger.info(
            f"  SY9 压力观测: total_points={total_points}, valid_points={valid_points}, "
            f"nan_ratio={nan_ratio:.4f}"
        )
        if missing_runs:
            longest = max(missing_runs, key=lambda r: r[4])
            self.logger.info(
                f"  最长缺失段: {longest[2]} ~ {longest[3]}, length_days={longest[4]}"
            )
        
        t_valid = t_days_full[combined_mask]
        p_valid = np.asarray(p_obs_full[combined_mask], dtype=np.float32)
        # v12: qg_obs 为 qg_filled_full 在 combined_mask 上的切片，不按 valid 再置零
        qg_valid = qg_filled_full[combined_mask].astype(np.float32)
        qg_valid_local_mask = qg_valid_mask[combined_mask]
        shutin_mask_local = shutin_schedule_full[combined_mask]
        prod_hours_norm_local = prod_hours_norm_full[combined_mask]
        casing_norm_local = casing_norm_full[combined_mask]
        qw_valid = qw_full[combined_mask]
        p_valid_local_mask = p_valid_mask[combined_mask]
        p_valid = np.where(p_valid_local_mask, p_valid, np.nan)
        qw_valid = np.nan_to_num(qw_valid, nan=0.0)
        
        N = len(t_valid)
        well_x_arr = np.full(N, well_x)
        well_y_arr = np.full(N, well_y)
        x_norm, y_norm = self.normalize_xy(well_x_arr, well_y_arr)
        t_norm = self.normalize_t(t_valid)
        xyt = np.stack([x_norm, y_norm, t_norm], axis=-1).astype(np.float32)
        
        # 全时间网格（用于绘图连续预测与缺失区间标注）
        well_x_full = np.full(total_points, well_x)
        well_y_full = np.full(total_points, well_y)
        x_norm_full, y_norm_full = self.normalize_xy(well_x_full, well_y_full)
        t_norm_full = self.normalize_t(t_days_full)
        xyt_full = np.stack([x_norm_full, y_norm_full, t_norm_full], axis=-1).astype(np.float32)
        
        self.logger.info(f"  井 {well_id}: {N} 条有效数据, "
                         f"p=[{p_valid.min():.1f}, {p_valid.max():.1f}] MPa")
        
        # 返回局部 valid mask，供 trainer/loss/source 过滤缺失补零
        return {
            'xyt': xyt,
            'p_obs': p_valid,
            'qg_obs': qg_valid,
            'qw_obs': qw_valid,
            'qg_valid_mask': qg_valid_local_mask.astype(np.float32),
            'shutin_mask': shutin_mask_local.astype(np.float32),
            'prod_hours_norm': prod_hours_norm_local.astype(np.float32),
            'casing_norm': casing_norm_local.astype(np.float32),
            'p_valid_mask_local': p_valid_local_mask.astype(np.float32),
            't_days': t_valid,
            'p_obs_full': p_obs_full,
            'valid_mask': valid_mask,
            'missing_runs': missing_runs,
            't_days_full': t_days_full,
            'xyt_full': xyt_full,
            'nan_ratio': nan_ratio,
            'total_points': total_points,
            'valid_points': valid_points,
        }
    
    def _compute_missing_runs(self, valid_mask: np.ndarray, prod: pd.DataFrame,
                              t_days: np.ndarray) -> list:
        """从 valid_mask 得到连续缺失区间列表 (start_idx, end_idx, start_date, end_date, length_days)."""
        missing_runs = []
        in_run = False
        start_idx = None
        dates = prod['date'] if 'date' in prod.columns else None
        n = len(valid_mask)
        for i in range(n):
            if not valid_mask[i]:
                if not in_run:
                    in_run = True
                    start_idx = i
            else:
                if in_run:
                    end_idx = i - 1
                    start_date = str(dates.iloc[start_idx]) if dates is not None else str(start_idx)
                    end_date = str(dates.iloc[end_idx]) if dates is not None else str(end_idx)
                    length_days = int(round(t_days[end_idx] - t_days[start_idx])) + 1 if end_idx >= start_idx else 1
                    missing_runs.append((start_idx, end_idx, start_date, end_date, length_days))
                    in_run = False
        if in_run:
            end_idx = n - 1
            start_date = str(dates.iloc[start_idx]) if dates is not None else str(start_idx)
            end_date = str(dates.iloc[end_idx]) if dates is not None else str(end_idx)
            length_days = int(round(t_days[end_idx] - t_days[start_idx])) + 1 if end_idx >= start_idx else 1
            missing_runs.append((start_idx, end_idx, start_date, end_date, length_days))
        return missing_runs
    
    def get_normalization_params(self) -> Dict[str, float]:
        """返回归一化参数（供模型/损失使用）"""
        return self.norm_params.copy()
    
    def to_tensor(self, arr: np.ndarray) -> 'torch.Tensor':
        """将 numpy 数组转换为 PyTorch 张量"""
        if torch is None:
            raise ImportError("PyTorch 未安装，请运行: pip install torch")
        return torch.from_numpy(arr).float()

