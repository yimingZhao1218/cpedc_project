"""
连通性分析模块 — 向后兼容重导出
================================
实际代码已迁移至 src/m6/connectivity.py

保留此文件仅防止旧 import 路径 `from pinn.connectivity import ...` 报错。
"""
from m6.connectivity import ConnectivityAnalyzer  # noqa: F401
