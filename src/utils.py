"""
CPEDC 项目工具模块
提供通用函数：日志、配置加载、数据验证等
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


class UTF8StreamHandler(logging.StreamHandler):
    """
    强制使用 UTF-8 编码的 StreamHandler，解决 Windows 控制台中文乱码问题
    """
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        # 如果 stream 有 buffer 属性，则用 UTF-8 包装
        if hasattr(stream, 'buffer'):
            import io
            stream = io.TextIOWrapper(
                stream.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True,
                write_through=True
            )
        super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # 确保消息以 UTF-8 编码输出
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


# ================================================================== #
#       一等奖配色方案 (专业期刊风格, 全项目统一引用)
# ================================================================== #
COLORS = {
    'primary':  '#2C3E50',   # 深蓝灰 — 观测值 / 主线
    'accent':   '#E74C3C',   # 红     — 预测值 / 高亮
    'train':    '#3498DB',   # 蓝     — 训练集
    'val':      '#F39C12',   # 橙     — 验证集
    'test':     '#27AE60',   # 绿     — 测试集
    'channel':  '#00FF7F',   # 亮绿   — 主控通道
    'well':     '#E74C3C',   # 红     — 井位标记
    'grid':     '#ECF0F1',   # 浅灰   — 网格
    'info_box': '#BDC3C7',   # 银灰   — 文本框边框
    'ic':       '#9B59B6',   # 紫     — IC 损失
    'bc':       '#1ABC9C',   # 青     — BC 损失
    'pde':      '#E67E22',   # 深橙   — PDE 损失
    'k_eff':    '#27AE60',   # 绿     — k_eff 参数
    'f_frac':   '#8E44AD',   # 深紫   — f_frac 参数
    'sw_lo':    '#3498DB',   # 蓝     — Sw 下界
    'sw_hi':    '#E74C3C',   # 红     — Sw 上界
}

# 色彩映射表 (colormap)
CMAP_K    = 'turbo'       # 渗透率场 — turbo 比 jet 更专业
CMAP_SW   = 'RdYlBu_r'   # 含水饱和度 — 红(高) → 蓝(低)
CMAP_P    = 'viridis'     # 压力场
CMAP_HEAT = 'YlOrRd'      # 残差/连通性热图

# 通用绘图参数
PLOT_RC = {
    'figure.dpi':      150,
    'savefig.dpi':     250,
    'axes.linewidth':  0.8,
    'axes.grid':       True,
    'grid.alpha':      0.25,
    'grid.linewidth':  0.5,
    'lines.linewidth': 1.4,
    'font.size':       11,
    'axes.titlesize':  13,
    'axes.labelsize':  12,
    'legend.fontsize': 10,
    'legend.framealpha': 0.85,
    'legend.edgecolor':  '#BDC3C7',
}


def apply_plot_style():
    """一键应用一等奖级别 matplotlib 全局样式"""
    import matplotlib.pyplot as plt
    for k, v in PLOT_RC.items():
        plt.rcParams[k] = v


def _force_console_utf8():
    """
    强制控制台使用 UTF-8，避免 Windows 下中文日志乱码。
    - Windows 下先设置控制台代码页为 65001 (UTF-8)
    - 再设置 Python stdout/stderr 编码为 utf-8
    """
    # 1. Windows: 先设控制台代码页，否则终端会按 GBK 解释 UTF-8 字节导致乱码
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    # 2. 设置 Python 标准流为 UTF-8
    for name, stream in [('stdout', sys.stdout), ('stderr', sys.stderr)]:
        enc = getattr(stream, 'encoding', None) or ''
        if enc.lower() != 'utf-8':
            try:
                if hasattr(stream, 'reconfigure'):
                    stream.reconfigure(encoding='utf-8', errors='replace')
                elif getattr(stream, 'buffer', None) is not None:
                    import io
                    new_stream = io.TextIOWrapper(
                        stream.buffer, encoding='utf-8', errors='replace', line_buffering=True
                    )
                    if name == 'stdout':
                        sys.stdout = new_stream
                    else:
                        sys.stderr = new_stream
            except Exception:
                pass


def setup_chinese_support():
    """
    配置 matplotlib 中文支持，避免乱码和负号缺失。
    
    优先级: Microsoft YaHei > SimHei > Noto Sans SC > 系统默认
    - Microsoft YaHei 覆盖 Unicode 最全（含负号 U+2212）
    - SimHei 缺 U+2212，需配合 axes.unicode_minus=False 用 ASCII 减号代替
    - 始终设 axes.unicode_minus=False 以防万一
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # 按优先级尝试可用字体
    preferred = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'SimSun']
    available = set(f.name for f in fm.fontManager.ttflist)
    chosen = [f for f in preferred if f in available]
    
    if chosen:
        # 选中的字体放前面，保留 DejaVu Sans 作后备（英文/符号）
        plt.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
    else:
        # 没有任何 CJK 字体，用默认
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 用 ASCII 减号代替 Unicode 负号（避免 SimHei 缺 U+2212 的方框）
    plt.rcParams['axes.unicode_minus'] = False
    
    # 数学公式/对数坐标指数 用 DejaVu Sans 渲染（CJK 字体缺数学符号）
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    
    # 刷新字体缓存，确保上面的设置立即生效
    fm._load_fontmanager(try_read_cache=False)
    
    # 控制台中文不乱码：强制 stdout/stderr 使用 UTF-8，Windows 下并设置控制台代码页
    _force_console_utf8()


def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径（默认在项目根目录）
        
    Returns:
        配置字典
    """
    # 如果是相对路径，从项目根目录查找
    if not os.path.isabs(config_path):
        # 获取当前文件所在目录（src/）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 项目根目录是 src 的上一级
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理路径变量替换
    config = _resolve_config_variables(config)
    
    return config


def _get_nested(config: Dict, key_path: str):
    """从 config 中按 'a.b.c' 路径取值，取不到返回 None。"""
    val = config
    for part in key_path.split('.'):
        if not isinstance(val, dict):
            return None
        val = val.get(part)
        if val is None:
            return None
    return val


def _resolve_config_variables(config: Dict) -> Dict:
    """
    解析配置中的变量（如 ${meta.experiment_name}），用 config 中已有字段替换。
    
    Args:
        config: 原始配置字典
        
    Returns:
        解析后的配置字典
    """
    import re

    def get_var_value(key_path: str):
        """优先从 config 取，否则用默认值"""
        v = _get_nested(config, key_path)
        if v is not None and not (isinstance(v, dict) and len(v) == 0):
            return str(v)
        defaults = {
            'meta.experiment_name': 'default',
            'paths.raw_data': config.get('paths', {}).get('raw_data', 'data/raw'),
        }
        return defaults.get(key_path, '')

    def replace_vars(obj):
        """递归替换配置中的变量"""
        if isinstance(obj, dict):
            return {k: replace_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_vars(item) for item in obj]
        elif isinstance(obj, str):
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, obj)
            for match in matches:
                value = get_var_value(match)
                obj = obj.replace(f'${{{match}}}', value)
            return obj
        else:
            return obj

    return replace_vars(config)


def setup_logger(name: str, log_dir: str = 'logs', level=logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志名称
        log_dir: 日志目录
        level: 日志级别
        
    Returns:
        配置好的logger
    """
    # 先强制设置控制台 UTF-8 编码，避免日志乱码
    _force_console_utf8()
    
    # 如果是相对路径，从项目根目录计算
    if not os.path.isabs(log_dir):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        log_dir = os.path.join(project_root, log_dir)
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 文件处理器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(
        f'{log_dir}/{name}_{timestamp}.log',
        encoding='utf-8'
    )
    fh.setLevel(level)
    
    # 控制台处理器 - 使用自定义的 UTF8StreamHandler
    ch = UTF8StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def ensure_dir(directory: str):
    """
    确保目录存在
    
    Args:
        directory: 目录路径（相对于项目根目录）
    """
    # 如果是相对路径，从项目根目录计算
    if not os.path.isabs(directory):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        directory = os.path.join(project_root, directory)
    
    Path(directory).mkdir(parents=True, exist_ok=True)


def detect_outliers_iqr(data: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """
    使用IQR方法检测离群值
    
    Args:
        data: 输入数据
        factor: IQR倍数因子
        
    Returns:
        布尔数组，True表示离群值
    """
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (data < lower) | (data > upper)


def calculate_statistics(data: pd.Series, name: str = "") -> Dict:
    """
    计算数据统计信息
    
    Args:
        data: pandas Series
        name: 数据名称
        
    Returns:
        统计信息字典
    """
    stats = {
        '名称': name,
        '总数': len(data),
        '缺失值': data.isna().sum(),
        '缺失率': f"{data.isna().sum() / len(data) * 100:.2f}%",
        '最小值': data.min() if not data.isna().all() else np.nan,
        '最大值': data.max() if not data.isna().all() else np.nan,
        '平均值': data.mean() if not data.isna().all() else np.nan,
        '中位数': data.median() if not data.isna().all() else np.nan,
        '标准差': data.std() if not data.isna().all() else np.nan,
    }
    return stats


def haversine_distance(lon1: float, lat1: float, 
                       lon2: float, lat2: float) -> float:
    """
    计算两个经纬度点之间的距离（米）
    
    Args:
        lon1, lat1: 第一个点的经纬度
        lon2, lat2: 第二个点的经纬度
        
    Returns:
        距离（米）
    """
    R = 6371000  # 地球半径（米）
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def write_markdown_report(content: List[str], filepath: str):
    """
    写入Markdown格式报告
    
    Args:
        content: 内容行列表
        filepath: 输出文件路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))


def format_number(num: float, decimals: int = 2) -> str:
    """格式化数字显示"""
    if pd.isna(num):
        return 'N/A'
    return f"{num:.{decimals}f}"


class DataValidator:
    """数据验证器类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.errors = []
        self.warnings = []
    
    def validate_range(self, data: pd.Series, min_val: float, 
                       max_val: float, name: str = ""):
        """验证数据范围"""
        invalid = (data < min_val) | (data > max_val)
        if invalid.any():
            count = invalid.sum()
            msg = f"{name} 有 {count} 个值超出范围 [{min_val}, {max_val}]"
            self.warnings.append(msg)
            self.logger.warning(msg)
        return ~invalid
    
    def validate_not_null(self, data: pd.Series, name: str = ""):
        """验证非空"""
        null_count = data.isna().sum()
        if null_count > 0:
            msg = f"{name} 有 {null_count} 个缺失值"
            self.warnings.append(msg)
            self.logger.warning(msg)
        return data.notna()
    
    def validate_unique(self, data: pd.Series, name: str = ""):
        """验证唯一性"""
        dup_count = data.duplicated().sum()
        if dup_count > 0:
            msg = f"{name} 有 {dup_count} 个重复值"
            self.errors.append(msg)
            self.logger.error(msg)
        return ~data.duplicated()
    
    def get_report(self) -> str:
        """获取验证报告"""
        lines = ["# 数据验证报告\n"]
        
        if self.errors:
            lines.append("## 错误\n")
            for err in self.errors:
                lines.append(f"- ❌ {err}")
            lines.append("")
        
        if self.warnings:
            lines.append("## 警告\n")
            for warn in self.warnings:
                lines.append(f"- ⚠️ {warn}")
            lines.append("")
        
        if not self.errors and not self.warnings:
            lines.append("✅ 所有验证通过！\n")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    # 测试
    setup_chinese_support()
    logger = setup_logger('test')
    logger.info("工具模块测试成功")
    print("✅ 工具模块加载正常")
