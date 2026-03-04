"""
M3 气水相渗模块
使用 PCHIP 插值 + 端点/单调/非负强制约束

提供 API：
    RelPermGW.krw(sw)       → 水相相对渗透率
    RelPermGW.krg(sw)       → 气相相对渗透率
    RelPermGW.dkrw_dsw(sw)  → dkrw/dSw（解析导数）
    RelPermGW.dkrg_dsw(sw)  → dkrg/dSw（解析导数）
    RelPermGW.endpoints()   → (Swr, Sgr, krw_max, krg_max)
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger, load_config

ArrayLike = Union[float, np.ndarray]


class RelPermGW:
    """
    气水两相相渗插值器
    
    使用 PCHIP 保持单调性：
    - krw(Sw) 单调递增
    - krg(Sw) 单调递减
    - 0 <= kr <= 1
    - Sw 限制在 [Swr, 1-Sgr] 范围内
    
    数据来源：附表7-相对渗透率数据表
    """
    
    def __init__(self, config: Optional[dict] = None, config_path: str = 'config.yaml'):
        """
        初始化相渗插值器
        
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
        self.logger = setup_logger('RelPermGW')
        
        # 相渗数据
        self.sw_data: np.ndarray = np.array([])  # 含水饱和度 (分数)
        self.krg_data: np.ndarray = np.array([])  # 气相相渗 (分数)
        self.krw_data: np.ndarray = np.array([])  # 水相相渗 (分数)
        
        # 端点
        self.Swr: float = 0.0    # 束缚水饱和度
        self.Sgr: float = 0.0    # 残余气饱和度
        self.krw_max: float = 0.0  # 最大水相相渗
        self.krg_max: float = 0.0  # 最大气相相渗
        
        # PCHIP 插值器
        self._interp_krw: Optional[PchipInterpolator] = None
        self._interp_krg: Optional[PchipInterpolator] = None
        
        self._load_data()
        self.logger.info(
            f"RelPermGW 初始化完成: Swr={self.Swr:.4f}, Sgr={self.Sgr:.4f}, "
            f"Sw=[{self.sw_data.min():.4f}, {self.sw_data.max():.4f}]"
        )
    
    def _load_data(self):
        """加载并处理相渗数据"""
        filepath = self.config['data']['sources']['relperm_csv']
        self.logger.info(f"加载相渗数据: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找数据起始行（含 "序号" 的行是表头）
        header_idx = -1
        for i, line in enumerate(lines):
            if '序号' in line and '含水饱和度' in line:
                header_idx = i
                break
        
        if header_idx < 0:
            raise ValueError(f"无法找到相渗数据表头: {filepath}")
        
        # 解析数据行
        sw_list, krg_list, krw_list = [], [], []
        for line in lines[header_idx + 1:]:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            try:
                # 序号, Sw(%), Krg, Krw
                sw_pct = float(parts[1].strip())
                krg_val = float(parts[2].strip())
                krw_val = float(parts[3].strip())
                
                sw_list.append(sw_pct / 100.0)  # % → 分数
                krg_list.append(krg_val)  # 已是分数（0~1）
                krw_list.append(krw_val)  # 已是分数（0~1）
            except ValueError:
                continue
        
        self.sw_data = np.array(sw_list, dtype=np.float64)
        self.krg_data = np.array(krg_list, dtype=np.float64)
        self.krw_data = np.array(krw_list, dtype=np.float64)
        
        # 排序（按 Sw 升序）
        sort_idx = np.argsort(self.sw_data)
        self.sw_data = self.sw_data[sort_idx]
        self.krg_data = self.krg_data[sort_idx]
        self.krw_data = self.krw_data[sort_idx]
        
        # 去重
        _, unique_idx = np.unique(self.sw_data, return_index=True)
        self.sw_data = self.sw_data[unique_idx]
        self.krg_data = self.krg_data[unique_idx]
        self.krw_data = self.krw_data[unique_idx]
        
        # 强制非负和上限
        self.krg_data = np.clip(self.krg_data, 0.0, 1.0)
        self.krw_data = np.clip(self.krw_data, 0.0, 1.0)
        
        # 端点
        self.Swr = self.sw_data[0]   # 束缚水饱和度（最小 Sw）
        self.Sgr = 1.0 - self.sw_data[-1]  # 残余气饱和度
        self.krg_max = self.krg_data[0]   # Sw=Swr 时的 krg
        self.krw_max = self.krw_data[-1]  # Sw=1-Sgr 时的 krw
        
        self.logger.info(
            f"  数据点数: {len(self.sw_data)}, "
            f"Swr={self.Swr:.4f}, Sgr={self.Sgr:.4f}, "
            f"krg_max={self.krg_max:.4f}, krw_max={self.krw_max:.4f}"
        )
        
        # 构建 PCHIP 插值器
        self._interp_krw = PchipInterpolator(self.sw_data, self.krw_data)
        self._interp_krg = PchipInterpolator(self.sw_data, self.krg_data)
        
        self.logger.info("  PCHIP 插值器创建成功")
    
    def _clamp_sw(self, sw: ArrayLike) -> np.ndarray:
        """将 Sw 限制在有效范围 [Swr, 1-Sgr] 内，越界时 warning（禁止静默外推）"""
        sw = np.atleast_1d(np.asarray(sw, dtype=np.float64))
        sw_min = self.sw_data[0]
        sw_max = self.sw_data[-1]
        below = sw < sw_min
        above = sw > sw_max
        if np.any(below) or np.any(above):
            n_below = int(np.sum(below))
            n_above = int(np.sum(above))
            self.logger.warning(
                f"Sw 越界: {n_below} 个值 < {sw_min:.4f}, "
                f"{n_above} 个值 > {sw_max:.4f}，已 clamp 到 [{sw_min:.4f}, {sw_max:.4f}]"
            )
        return np.clip(sw, sw_min, sw_max)
    
    def krw(self, sw: ArrayLike) -> np.ndarray:
        """
        水相相对渗透率 krw(Sw)
        
        Args:
            sw: 含水饱和度（分数，0~1）
        Returns:
            krw 值（0~1）
        """
        sw_c = self._clamp_sw(sw)
        result = self._interp_krw(sw_c)
        return np.clip(result, 0.0, 1.0)
    
    def krg(self, sw: ArrayLike) -> np.ndarray:
        """
        气相相对渗透率 krg(Sw)
        
        Args:
            sw: 含水饱和度（分数，0~1）
        Returns:
            krg 值（0~1）
        """
        sw_c = self._clamp_sw(sw)
        result = self._interp_krg(sw_c)
        return np.clip(result, 0.0, 1.0)
    
    def dkrw_dsw(self, sw: ArrayLike) -> np.ndarray:
        """krw 对 Sw 的导数（PCHIP 解析导数）"""
        sw_c = self._clamp_sw(sw)
        return self._interp_krw(sw_c, 1)  # 1 阶导数
    
    def dkrg_dsw(self, sw: ArrayLike) -> np.ndarray:
        """krg 对 Sw 的导数（PCHIP 解析导数）"""
        sw_c = self._clamp_sw(sw)
        return self._interp_krg(sw_c, 1)  # 1 阶导数
    
    def endpoints(self) -> Tuple[float, float, float, float]:
        """
        返回相渗端点参数
        
        Returns:
            (Swr, Sgr, krw_max, krg_max)
        """
        return (self.Swr, self.Sgr, self.krw_max, self.krg_max)
    
    def get_sw_range(self) -> Tuple[float, float]:
        """返回有效 Sw 范围"""
        return (self.sw_data[0], self.sw_data[-1])
