"""
M3 单位转换与数据预处理工具
统一全项目物理量单位：p(MPa), T(℃/K), rho(kg/m³), Bg(m³/m³)
"""

import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


# ------------------------------------------------------------------ #
#  压力转换
# ------------------------------------------------------------------ #
def to_MPa(p: ArrayLike, from_unit: str = 'MPa') -> ArrayLike:
    """将压力转换为 MPa"""
    conversions = {
        'MPa': 1.0,
        'Pa': 1e-6,
        'kPa': 1e-3,
        'bar': 0.1,
        'atm': 0.101325,
        'psi': 0.00689476,
    }
    if from_unit not in conversions:
        raise ValueError(f"不支持的压力单位: {from_unit}，可选: {list(conversions.keys())}")
    return np.asarray(p, dtype=np.float64) * conversions[from_unit]


def to_Pa(p: ArrayLike, from_unit: str = 'MPa') -> ArrayLike:
    """将压力转换为 Pa"""
    return to_MPa(p, from_unit) * 1e6


# ------------------------------------------------------------------ #
#  温度转换
# ------------------------------------------------------------------ #
def to_K(T_C: ArrayLike) -> ArrayLike:
    """摄氏度 → 开尔文"""
    return np.asarray(T_C, dtype=np.float64) + 273.15


def to_C(T_K: ArrayLike) -> ArrayLike:
    """开尔文 → 摄氏度"""
    return np.asarray(T_K, dtype=np.float64) - 273.15


# ------------------------------------------------------------------ #
#  密度转换
# ------------------------------------------------------------------ #
def to_kg_m3(rho: ArrayLike, from_unit: str = 'g/cm3') -> ArrayLike:
    """将密度转换为 kg/m³"""
    conversions = {
        'kg/m3': 1.0,
        'g/cm3': 1000.0,
        'g/cc': 1000.0,
    }
    if from_unit not in conversions:
        raise ValueError(f"不支持的密度单位: {from_unit}")
    return np.asarray(rho, dtype=np.float64) * conversions[from_unit]


# ------------------------------------------------------------------ #
#  数据预处理
# ------------------------------------------------------------------ #
def ensure_sorted_unique(arr: np.ndarray) -> np.ndarray:
    """确保数组排序且无重复"""
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.unique(arr)
    return np.sort(arr)


def clamp_with_warning(value: ArrayLike, vmin: float, vmax: float,
                       name: str = '', logger=None, extra_note: str = '') -> np.ndarray:
    """
    将值限制在 [vmin, vmax] 范围内，超范围时记录警告。
    返回的是 clamp 后的值，后续用该值查表即得到端点/边界处的物性（非先验）。
    extra_note: 追加说明，如 "，物性取端点值"。
    """
    value = np.asarray(value, dtype=np.float64)
    below = value < vmin
    above = value > vmax
    if np.any(below) or np.any(above):
        n_below = int(np.sum(below))
        n_above = int(np.sum(above))
        msg = (f"{name}: {n_below} 个值 < {vmin}, {n_above} 个值 > {vmax}，已 clamp 到表范围{extra_note}")
        if logger:
            logger.warning(msg)
    return np.clip(value, vmin, vmax)
