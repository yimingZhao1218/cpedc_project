"""
CPEDC M3 物性模块
提供 PVT 多温度插值 + 气水相渗函数
"""

from .pvt import GasPVT
from .relperm import RelPermGW

__all__ = ['GasPVT', 'RelPermGW']
