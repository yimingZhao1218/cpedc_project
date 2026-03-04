"""
M3 PVT 多温度插值模块
使用 PCHIP（分段三次 Hermite）保持单调性，避免超调
策略：沿 p 方向 PCHIP + 沿 T 方向线性插值 → 稳定的 f(p, T)

提供 API：
    GasPVT.z(p, T)       → 偏差系数 Z
    GasPVT.bg(p, T)      → 体积系数 Bg (m³/m³)
    GasPVT.cg(p, T)      → 压缩系数 cg (1/MPa)
    GasPVT.rho(p, T)     → 密度 rho (kg/m³)
    GasPVT.alphaT(p, T)  → 热膨胀系数 αT (1/℃)
    GasPVT.query_all(p, T) → dict
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
from scipy.interpolate import PchipInterpolator
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger, load_config
from physics.units import clamp_with_warning

ArrayLike = Union[float, np.ndarray]


class GasPVT:
    """
    气体 PVT 多温度插值器
    
    使用 PCHIP 沿压力方向插值（保持单调），沿温度方向线性插值
    数据来源：附表5 PVT 数据表
    """
    
    def __init__(self, config: Optional[dict] = None, config_path: str = 'config.yaml'):
        """
        初始化 PVT 插值器
        
        Args:
            config: 配置字典（若为 None 则从文件加载）
            config_path: 配置文件路径
        """
        if config is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            config = load_config(str(project_root / config_path))
            for key, value in config['paths'].items():
                config['paths'][key] = str(project_root / value)
        
        self.config = config
        self.logger = setup_logger('GasPVT')
        
        # 温度列表 (℃)
        self.temperatures: List[float] = []
        # 各物性的 PCHIP 插值器 {T: PchipInterpolator}
        self._interp_z: Dict[float, PchipInterpolator] = {}
        self._interp_bg: Dict[float, PchipInterpolator] = {}
        self._interp_cg: Dict[float, PchipInterpolator] = {}
        self._interp_rho: Dict[float, PchipInterpolator] = {}
        self._interp_alphaT: Dict[float, PchipInterpolator] = {}
        # 热膨胀系数的温度列表（不同于其他属性）
        self.temperatures_alphaT: List[float] = []
        
        # 压力范围
        self.p_min = np.inf
        self.p_max = -np.inf
        
        self._load_all_data()
        self.logger.info(
            f"GasPVT 初始化完成: T={self.temperatures} ℃, "
            f"p=[{self.p_min:.1f}, {self.p_max:.1f}] MPa"
        )
    
    # ================================================================== #
    #                          数据加载
    # ================================================================== #
    def _load_all_data(self):
        """加载全部 PVT 数据并构建插值器"""
        pvt_sources = self.config['data']['sources']['pvt']
        
        # 从 YAML 读取单位因子（可审计，不写死）
        m3_cfg = self.config.get('m3_config', {}).get('pvt', {})
        uf = m3_cfg.get('unit_factors', {})
        uf_bg = uf.get('bg', 1e-3)
        uf_cg = uf.get('cg', 0.01)
        uf_rho = uf.get('rho', 1000.0)
        uf_aT = uf.get('alphaT', 0.01)
        
        # Z 因子（偏差系数）—— 从恒质膨胀数据表
        p_z, temps_z, data_z = self._parse_standard_pvt(
            pvt_sources['cce_csv'], unit_factor=1.0
        )
        self._interp_z = self._build_pchip(p_z, temps_z, data_z, 'Z')
        
        # Bg 体积系数: 原始单位 10^{-3} m³/m³
        p_bg, temps_bg, data_bg = self._parse_standard_pvt(
            pvt_sources['bg_csv'], unit_factor=uf_bg
        )
        self._interp_bg = self._build_pchip(p_bg, temps_bg, data_bg, 'Bg')
        
        # cg 压缩系数: 原始单位 10^{-2}/MPa
        p_cg, temps_cg, data_cg = self._parse_standard_pvt(
            pvt_sources['c_g_csv'], unit_factor=uf_cg
        )
        self._interp_cg = self._build_pchip(p_cg, temps_cg, data_cg, 'cg')
        
        # rho 密度: 原始单位 g/cm³
        p_rho, temps_rho, data_rho = self._parse_standard_pvt(
            pvt_sources['density_csv'], unit_factor=uf_rho
        )
        self._interp_rho = self._build_pchip(p_rho, temps_rho, data_rho, 'rho')
        
        # αT 热膨胀系数: 原始单位 10^{-2}/℃
        p_aT, temps_aT, data_aT = self._parse_alphaT_pvt(
            pvt_sources['alphaT_csv'], unit_factor=uf_aT
        )
        self._interp_alphaT = self._build_pchip(p_aT, temps_aT, data_aT, 'alphaT')
        self.temperatures_alphaT = sorted(temps_aT)
        
        # 统一温度列表（取 Z/Bg/cg/rho 共有的温度）
        self.temperatures = sorted(temps_z)
        
        # 更新压力范围
        for p_arr in [p_z, p_bg, p_cg, p_rho, p_aT]:
            self.p_min = min(self.p_min, p_arr.min())
            self.p_max = max(self.p_max, p_arr.max())
    
    def _parse_standard_pvt(self, filepath: str,
                            unit_factor: float = 1.0
                            ) -> Tuple[np.ndarray, List[float], Dict[float, np.ndarray]]:
        """
        解析标准 PVT CSV 文件（Z / Bg / cg / rho 格式）
        
        格式:
            Row 0: 标题行（压力单位 + 属性名）
            Row 1: 温度标签（,16.5℃,46.5℃,...）
            Row 2+: 数据行（压力, 值@T1, 值@T2, ...）
            最后一行可能带 * 前缀（地层压力模拟值）
            末行为注释
        
        Returns:
            pressures: 压力数组 (MPa)
            temperatures: 温度列表 (℃)
            data: {T: values_array}
        """
        self.logger.info(f"解析 PVT 数据: {filepath}")
        
        # 读取原始文件
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 提取温度（从第2行，匹配 数字+℃ 模式）
        temperatures = []
        temp_line = lines[1]
        temp_matches = re.findall(r'([\d.]+)℃', temp_line)
        temperatures = [float(t) for t in temp_matches]
        n_temps = len(temperatures)
        
        if n_temps == 0:
            raise ValueError(f"无法从文件 {filepath} 解析温度")
        
        self.logger.info(f"  温度: {temperatures} ℃, 共 {n_temps} 个")
        
        # 提取数据行
        pressures = []
        data_by_temp = {T: [] for T in temperatures}
        
        for line in lines[2:]:
            line = line.strip()
            if not line or line.startswith('（') or line.startswith('('):
                continue
            
            parts = line.split(',')
            # 提取压力值（可能有 * 前缀）
            p_str = parts[0].strip().lstrip('*')
            try:
                p = float(p_str)
            except ValueError:
                continue
            
            # 提取各温度对应的值
            values = []
            for i in range(1, n_temps + 1):
                if i < len(parts):
                    try:
                        values.append(float(parts[i].strip()))
                    except ValueError:
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            
            if len(values) == n_temps and not any(np.isnan(values)):
                pressures.append(p)
                for j, T in enumerate(temperatures):
                    data_by_temp[T].append(values[j] * unit_factor)
        
        pressures = np.array(pressures, dtype=np.float64)
        data = {T: np.array(vals, dtype=np.float64) for T, vals in data_by_temp.items()}
        
        self.logger.info(f"  压力范围: [{pressures.min():.1f}, {pressures.max():.1f}] MPa, "
                         f"共 {len(pressures)} 个点")
        
        return pressures, temperatures, data
    
    def _parse_alphaT_pvt(self, filepath: str,
                          unit_factor: float = 0.01
                          ) -> Tuple[np.ndarray, List[float], Dict[float, np.ndarray]]:
        """
        解析热膨胀系数 CSV（温度为区间）
        
        αT 按温区等效曲线分段匹配（不做温区间插值，避免温区语义被错误平滑）：
        - 查询时按 T 落入的温区选择对应曲线
        - 超出范围按最近区间 clamp + warning
        
        为兼容 _interp_2d / _build_pchip（仍以"代表温度"为 key），
        这里用区间中点作为 key，但 alpha_T() 查询走 _interp_alphaT_zone()。
        """
        self.logger.info(f"解析热膨胀系数数据: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 从 row 1 和 row 2 解析温度区间
        row1_parts = lines[1].split(',')
        row2_parts = lines[2].split(',')
        
        temp_intervals = []  # [(T_start, T_end)]
        for i in range(1, len(row1_parts)):
            start_str = row1_parts[i].strip().rstrip('-').replace('℃', '')
            if i < len(row2_parts):
                end_str = row2_parts[i].strip().replace('℃', '')
            else:
                continue
            try:
                T_start = float(start_str)
                T_end = float(end_str)
                temp_intervals.append((T_start, T_end))
            except ValueError:
                continue
        
        # 使用前4个区间（跳过最后一个全范围平均）
        if len(temp_intervals) > 4:
            temp_intervals = temp_intervals[:4]
        
        # 保存温区区间（供 zone-based 查询使用）
        self._alphaT_intervals = temp_intervals
        
        # 中点温度（作为 PCHIP 插值器的 key）
        temperatures = [(t[0] + t[1]) / 2.0 for t in temp_intervals]
        n_temps = len(temperatures)
        
        self.logger.info(f"  温度区间: {temp_intervals}")
        self.logger.info(f"  代表温度(中点): {temperatures} ℃")
        
        # 提取数据行
        pressures = []
        data_by_temp = {T: [] for T in temperatures}
        
        for line in lines[3:]:
            line = line.strip()
            if not line or line.startswith('（') or line.startswith('('):
                continue
            
            parts = line.split(',')
            p_str = parts[0].strip().lstrip('*')
            try:
                p = float(p_str)
            except ValueError:
                continue
            
            values = []
            for i in range(1, n_temps + 1):
                if i < len(parts):
                    try:
                        values.append(float(parts[i].strip()))
                    except ValueError:
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            
            if len(values) == n_temps and not any(np.isnan(values)):
                pressures.append(p)
                for j, T in enumerate(temperatures):
                    data_by_temp[T].append(values[j] * unit_factor)
        
        pressures = np.array(pressures, dtype=np.float64)
        data = {T: np.array(vals, dtype=np.float64) for T, vals in data_by_temp.items()}
        
        self.logger.info(f"  压力范围: [{pressures.min():.1f}, {pressures.max():.1f}] MPa, "
                         f"共 {len(pressures)} 个点")
        
        return pressures, temperatures, data
    
    # ================================================================== #
    #                        插值器构建
    # ================================================================== #
    def _build_pchip(self, pressures: np.ndarray, temperatures: List[float],
                     data: Dict[float, np.ndarray],
                     name: str) -> Dict[float, PchipInterpolator]:
        """
        为每个温度建立 PCHIP 插值器（沿压力方向）
        PCHIP 保持数据单调性并避免过冲，适合 PVT 曲线
        """
        interpolators = {}
        for T in temperatures:
            values = data[T]
            # 确保压力和值同步排序
            sort_idx = np.argsort(pressures)
            p_sorted = pressures[sort_idx]
            v_sorted = values[sort_idx]
            
            try:
                interpolators[T] = PchipInterpolator(p_sorted, v_sorted)
                self.logger.debug(f"  {name} @ T={T}℃: PCHIP 插值器创建成功, "
                                  f"p=[{p_sorted[0]:.1f}, {p_sorted[-1]:.1f}]")
            except Exception as e:
                self.logger.error(f"  {name} @ T={T}℃: PCHIP 创建失败: {e}")
                raise
        
        return interpolators
    
    # ================================================================== #
    #                         2D 查询
    # ================================================================== #
    def _interp_2d(self, p: ArrayLike, T: ArrayLike,
                   interpolators: Dict[float, PchipInterpolator],
                   temps: List[float],
                   name: str) -> np.ndarray:
        """
        2D 插值查询：沿 p 方向 PCHIP，沿 T 方向线性
        
        Args:
            p: 压力 (MPa)，标量或数组
            T: 温度 (℃)，标量或数组
            interpolators: {T: PchipInterpolator}
            temps: 排序后的温度列表
            name: 属性名（用于日志）
        
        Returns:
            插值结果数组
        """
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))
        T = np.atleast_1d(np.asarray(T, dtype=np.float64))
        
        # 广播
        if p.shape != T.shape:
            p, T = np.broadcast_arrays(p, T)
        
        # --- 禁止静默外推：压力 clamp + warning（物性取表端点值，非先验）---
        p = clamp_with_warning(p, self.p_min, self.p_max,
                               name=f'{name} pressure(MPa)', logger=self.logger,
                               extra_note='，物性取端点值')
        
        result = np.empty_like(p, dtype=np.float64)
        T_arr = np.array(sorted(temps), dtype=np.float64)
        T_min, T_max = T_arr[0], T_arr[-1]
        
        # --- 禁止静默外推：温度 clamp + warning ---
        T = clamp_with_warning(T, T_min, T_max,
                               name=f'{name} temperature(℃)', logger=self.logger)
        
        for i in range(len(p.flat)):
            pi = p.flat[i]
            Ti = T.flat[i]
            
            # 温度已经在上面 clamp 过了
            Ti_c = Ti
            
            if Ti_c <= T_arr[0]:
                # 直接用最低温度的插值器
                result.flat[i] = float(interpolators[T_arr[0]](pi))
            elif Ti_c >= T_arr[-1]:
                result.flat[i] = float(interpolators[T_arr[-1]](pi))
            else:
                # 找到包围温度
                idx = np.searchsorted(T_arr, Ti_c) - 1
                idx = max(0, min(idx, len(T_arr) - 2))
                T_lo = T_arr[idx]
                T_hi = T_arr[idx + 1]
                
                v_lo = float(interpolators[T_lo](pi))
                v_hi = float(interpolators[T_hi](pi))
                
                # 线性插值
                w = (Ti_c - T_lo) / (T_hi - T_lo)
                result.flat[i] = v_lo + w * (v_hi - v_lo)
        
        return result
    
    # ================================================================== #
    #                          公开 API
    # ================================================================== #
    # 物理量下限 ε（防止 PCHIP 微小异常导致负值，PINN 残差里除零/负值灾难）
    _EPS = 1e-12
    
    def z(self, p: ArrayLike, T: ArrayLike) -> np.ndarray:
        """
        偏差系数 Z(p, T)
        
        Args:
            p: 压力 (MPa)
            T: 温度 (℃)
        Returns:
            Z 因子（无量纲, > 0）
        """
        return np.maximum(self._interp_2d(p, T, self._interp_z, self.temperatures, 'Z'),
                          self._EPS)
    
    def bg(self, p: ArrayLike, T: ArrayLike) -> np.ndarray:
        """
        体积系数 Bg(p, T) — 单位 m³/m³ (> 0)
        
        Args:
            p: 压力 (MPa)
            T: 温度 (℃)
        """
        return np.maximum(self._interp_2d(p, T, self._interp_bg, self.temperatures, 'Bg'),
                          self._EPS)
    
    def cg(self, p: ArrayLike, T: ArrayLike) -> np.ndarray:
        """
        压缩系数 cg(p, T) — 单位 1/MPa (>= 0)
        
        Args:
            p: 压力 (MPa)
            T: 温度 (℃)
        """
        return np.maximum(self._interp_2d(p, T, self._interp_cg, self.temperatures, 'cg'),
                          0.0)
    
    def rho(self, p: ArrayLike, T: ArrayLike) -> np.ndarray:
        """
        密度 rho(p, T) — 单位 kg/m³ (> 0)
        
        Args:
            p: 压力 (MPa)
            T: 温度 (℃)
        """
        return np.maximum(self._interp_2d(p, T, self._interp_rho, self.temperatures, 'rho'),
                          self._EPS)
    
    def alpha_T(self, p: ArrayLike, T: ArrayLike) -> np.ndarray:
        """
        热膨胀系数 αT(p, T) — 单位 1/℃
        
        αT 按温区等效曲线分段匹配（不在温区间插值，避免温区语义被错误平滑）：
        查询 T 落入哪个温区 → 直接用该区间的 PCHIP(p) 曲线。
        超出范围按最近区间 clamp + warning。
        
        Args:
            p: 压力 (MPa)
            T: 温度 (℃)
        """
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))
        T = np.atleast_1d(np.asarray(T, dtype=np.float64))
        if p.shape != T.shape:
            p, T = np.broadcast_arrays(p, T)
        
        # 压力 clamp（物性取端点值）
        p = clamp_with_warning(p, self.p_min, self.p_max,
                               name='alphaT pressure(MPa)', logger=self.logger,
                               extra_note='，物性取端点值')
        
        intervals = self._alphaT_intervals  # [(T_lo, T_hi), ...]
        mid_temps = self.temperatures_alphaT  # 对应中点温度（插值器 key）
        T_global_min = intervals[0][0]
        T_global_max = intervals[-1][1]
        
        result = np.empty_like(p, dtype=np.float64)
        
        n_below = 0
        n_above = 0
        for i in range(len(p.flat)):
            pi = p.flat[i]
            Ti = T.flat[i]
            
            # 分段选择温区
            zone_idx = -1
            for k, (lo, hi) in enumerate(intervals):
                if lo <= Ti <= hi:
                    zone_idx = k
                    break
            
            if zone_idx < 0:
                # 超出范围：clamp 到最近区间
                if Ti < T_global_min:
                    zone_idx = 0
                    n_below += 1
                else:
                    zone_idx = len(intervals) - 1
                    n_above += 1
            
            # 用该温区的 PCHIP 插值器查询
            T_key = mid_temps[zone_idx]
            result.flat[i] = float(self._interp_alphaT[T_key](pi))
        
        if n_below > 0 or n_above > 0:
            self.logger.warning(
                f"alphaT 温区越界: {n_below} 个值 < {T_global_min}℃, "
                f"{n_above} 个值 > {T_global_max}℃，已 clamp 到最近温区"
            )
        
        return result
    
    def query_all(self, p: ArrayLike, T: ArrayLike) -> dict:
        """
        一次性查询全部物性
        
        Args:
            p: 压力 (MPa)
            T: 温度 (℃)
        
        Returns:
            {'Z': ..., 'Bg': ..., 'cg': ..., 'rho': ..., 'alphaT': ...}
        """
        return {
            'Z': self.z(p, T),
            'Bg': self.bg(p, T),
            'cg': self.cg(p, T),
            'rho': self.rho(p, T),
            'alphaT': self.alpha_T(p, T),
        }
    
    def get_pressure_range(self) -> Tuple[float, float]:
        """返回可用压力范围 (p_min, p_max) MPa"""
        return (self.p_min, self.p_max)
    
    def get_temperature_range(self) -> Tuple[float, float]:
        """返回可用温度范围 (T_min, T_max) ℃"""
        return (min(self.temperatures), max(self.temperatures))
    
    # ================================================================== #
    #          [审查修复 #15] M3→M4 数据管道: 多项式系数导出
    # ================================================================== #
    def export_polynomial_coeffs(self, prop_name: str, degree: int = 3,
                                  T: float = 140.32,
                                  n_dense: int = 200
                                  ) -> Tuple[np.ndarray, float]:
        """
        从 PCHIP 评估密集采样点, 拟合多项式系数, 供 TorchPVT 使用。
        
        建立 M3→M4 数据管道的核心方法:
            M3 PCHIP (精确) → polyfit → 多项式系数 → TorchPVT (可微分)
        
        Args:
            prop_name: 物性名 ('z', 'bg', 'cg', 'rho')
            degree: 多项式阶数 (默认3, 与 TorchPVT 当前 Z 因子一致)
            T: 温度 (℃), 默认 140.32 (储层温度)
            n_dense: 密集采样点数 (默认200)
        
        Returns:
            coeffs: 多项式系数, 从高次到低次 [a_n, a_{n-1}, ..., a_1, a_0]
                    即 f(p) = a_n*p^n + ... + a_1*p + a_0
            rmse: 拟合 RMSE
        
        用法示例:
            z_coeffs, z_rmse = gas_pvt.export_polynomial_coeffs('z', degree=3)
            # z_coeffs = [a3, a2, a1, a0]
            # Z(p) = a3*p³ + a2*p² + a1*p + a0
        """
        # 属性名 → 插值器映射
        interp_map = {
            'z': (self._interp_z, self.temperatures),
            'bg': (self._interp_bg, self.temperatures),
            'cg': (self._interp_cg, self.temperatures),
            'rho': (self._interp_rho, self.temperatures),
        }
        
        if prop_name not in interp_map:
            raise ValueError(
                f"未知属性 '{prop_name}', 支持: {list(interp_map.keys())}")
        
        interpolators, temps = interp_map[prop_name]
        
        # 密集采样
        p_dense = np.linspace(self.p_min, self.p_max, n_dense)
        values = self._interp_2d(p_dense, np.full_like(p_dense, T),
                                  interpolators, temps, prop_name)
        
        # 最小二乘多项式拟合
        coeffs = np.polyfit(p_dense, values, degree)
        
        # 计算 RMSE
        fitted = np.polyval(coeffs, p_dense)
        rmse = np.sqrt(np.mean((values - fitted) ** 2))
        
        self.logger.info(
            f"  export_polynomial_coeffs('{prop_name}', deg={degree}, "
            f"T={T}°C): RMSE={rmse:.6e}, "
            f"coeffs={[f'{c:.6e}' for c in coeffs]}"
        )
        
        return coeffs, rmse
    
    def export_all_polynomial_coeffs(self, degree: int = 3,
                                      T: float = 140.32,
                                      save_path: Optional[str] = None
                                      ) -> Dict[str, Dict]:
        """
        一次性导出全部物性多项式系数, 建立完整 M3→M4 数据管道。
        
        Args:
            degree: 多项式阶数
            T: 温度 (℃)
            save_path: 可选, JSON 保存路径
        
        Returns:
            {
                'z':   {'coeffs': [...], 'rmse': float, 'degree': int},
                'bg':  {'coeffs': [...], 'rmse': float, 'degree': int},
                'cg':  {'coeffs': [...], 'rmse': float, 'degree': int},
                'rho': {'coeffs': [...], 'rmse': float, 'degree': int},
                'meta': {
                    'T_C': float, 'p_range_MPa': [float, float],
                    'source': 'M3 GasPVT PCHIP → polyfit'
                }
            }
        """
        self.logger.info(
            f"导出全部多项式系数: deg={degree}, T={T}°C, "
            f"p=[{self.p_min:.1f}, {self.p_max:.1f}] MPa"
        )
        
        result = {}
        for prop in ['z', 'bg', 'cg', 'rho']:
            coeffs, rmse = self.export_polynomial_coeffs(prop, degree, T)
            result[prop] = {
                'coeffs': coeffs.tolist(),
                'rmse': float(rmse),
                'degree': degree,
            }
        
        result['meta'] = {
            'T_C': T,
            'p_range_MPa': [float(self.p_min), float(self.p_max)],
            'source': 'M3 GasPVT PCHIP → polyfit',
            'note': '系数从高次到低次: [a_n, ..., a_1, a_0]',
        }
        
        if save_path is not None:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self.logger.info(f"  多项式系数已保存: {save_path}")
        
        return result
