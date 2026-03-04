"""
CPEDC 可视化配置模块
====================
统一配色方案 + 图表样式

所有绘图模块（connectivity, water_invasion, m5_trainer 等）
统一引用此文件，确保视觉风格一致性。

使用方法：
    from pinn.viz_config import COLORS, CMAP_K, apply_professional_style
    
    apply_professional_style()  # 应用到 matplotlib
    plt.plot(x, y, color=COLORS['accent'])
"""

import matplotlib.pyplot as plt
from typing import Dict

# ===== CPEDC 一等奖统一配色方案 =====
COLORS = {
    # 核心颜色
    'primary': '#2C3E50',      # 深蓝灰 - 观测值/标题/文本
    'accent': '#E74C3C',       # 红色 - 预测值/高亮/最优
    
    # 训练集划分
    'train': '#3498DB',        # 蓝色 - 训练集
    'val': '#F39C12',          # 橙色 - 验证集
    'test': '#27AE60',         # 绿色 - 测试集
    
    # 可视化元素
    'channel': '#00FF7F',      # 亮绿 - 主控流动通道
    'well': '#E74C3C',         # 红色 - 井位标记（三角形）
    'grid': '#BDC3C7',         # 浅灰 - 网格线
    'background': '#FFFFFF',   # 白色 - 背景
    
    # 风险等级（M7 水侵预警）
    'safe': '#27AE60',         # 绿色 - 安全
    'warning': '#F39C12',      # 橙色 - 预警
    'danger': '#E67E22',       # 深橙 - 危险
    'critical': '#E74C3C',     # 红色 - 水淹/紧急
    
    # 消融实验配色（与 run_ablation_suite.py 一致）
    'pure_ml': '#95A5A6',      # 灰色 - 基线
    'pinn_const_k': '#3498DB', # 蓝色
    'pinn_knet': '#2980B9',    # 深蓝
    'pinn_full': '#E74C3C',    # 红色 - 最优
    'pinn_no_fourier': '#F39C12',  # 橙色
    'pinn_no_rar': '#9B59B6',  # 紫色
    
    # 制度优化策略配色（与 water_invasion.py 一致）
    'strategy_steady': '#E74C3C',      # 红色 - 稳产方案
    'strategy_decay': '#F39C12',       # 橙色 - 阶梯降产
    'strategy_control': '#27AE60',     # 绿色 - 控压方案
}

# ===== Colormap 配置 =====
CMAP_K = 'turbo'               # 渗透率场 k(x,y) (彩虹色，动态范围大)
CMAP_SW = 'RdYlBu_r'           # 含水饱和度 Sw(x,y,t) (红黄蓝反转)
CMAP_P = 'viridis'             # 压力场 p(x,y,t) (蓝绿黄)
CMAP_HEAT = 'YlOrRd'           # 连通性热图 C_ij (黄橙红)
CMAP_RESIDUAL = 'RdBu_r'       # PDE 残差 (红蓝反转)

# ===== Matplotlib 全局样式 =====
MPL_STYLE_CONFIG = {
    # 字体设置
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'SimHei'],  # SimHei 支持中文
    'axes.unicode_minus': False,  # 负号显示
    
    # 标题与标签
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.titlepad': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'normal',
    
    # 图例
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#E0E0E0',
    
    # 刻度
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    
    # 网格
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.color': '#BDC3C7',
    
    # 图形质量
    'figure.dpi': 150,           # 屏幕显示 DPI
    'savefig.dpi': 200,          # 保存 DPI (评委报告用)
    'savefig.bbox': 'tight',     # 裁剪空白
    'savefig.pad_inches': 0.1,
    
    # 线条与标记
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    
    # 颜色循环
    'axes.prop_cycle': plt.cycler(color=[
        COLORS['accent'], COLORS['train'], COLORS['test'],
        COLORS['val'], COLORS['warning'], COLORS['pinn_no_rar']
    ]),
}


def apply_professional_style():
    """
    应用专业配色方案到所有 matplotlib 图件
    
    使用方法:
        在任何绘图脚本开头调用：
        from pinn.viz_config import apply_professional_style
        apply_professional_style()
    """
    plt.rcParams.update(MPL_STYLE_CONFIG)


def get_strategy_color(strategy_name: str) -> str:
    """
    根据策略名称返回颜色
    
    Args:
        strategy_name: '稳产方案', '阶梯降产', '控压方案' 等
    
    Returns:
        hex color string
    """
    mapping = {
        '稳产方案': COLORS['strategy_steady'],
        '稳产': COLORS['strategy_steady'],
        'steady': COLORS['strategy_steady'],
        
        '阶梯降产': COLORS['strategy_decay'],
        '降产': COLORS['strategy_decay'],
        'decay': COLORS['strategy_decay'],
        
        '控压方案': COLORS['strategy_control'],
        '控压': COLORS['strategy_control'],
        'control': COLORS['strategy_control'],
    }
    return mapping.get(strategy_name, COLORS['primary'])


def get_risk_color(risk_level: str) -> str:
    """
    根据风险等级返回颜色
    
    Args:
        risk_level: '安全', '预警', '危险', '水淹' 等
    
    Returns:
        hex color string
    """
    mapping = {
        '安全': COLORS['safe'],
        'safe': COLORS['safe'],
        
        '预警': COLORS['warning'],
        'warning': COLORS['warning'],
        
        '危险': COLORS['danger'],
        'danger': COLORS['danger'],
        
        '水淹': COLORS['critical'],
        '紧急': COLORS['critical'],
        'critical': COLORS['critical'],
    }
    return mapping.get(risk_level, COLORS['primary'])


def get_ablation_color(exp_name: str) -> str:
    """
    根据消融实验名称返回颜色
    
    Args:
        exp_name: 'pure_ml', 'pinn_const_k', 'pinn_full' 等
    
    Returns:
        hex color string
    """
    mapping = {
        'pure_ml': COLORS['pure_ml'],
        'pinn_const_k': COLORS['pinn_const_k'],
        'pinn_knet': COLORS['pinn_knet'],
        'pinn_full': COLORS['pinn_full'],
        'pinn_no_fourier': COLORS['pinn_no_fourier'],
        'pinn_no_rar': COLORS['pinn_no_rar'],
    }
    return mapping.get(exp_name, COLORS['primary'])


# ===== 常用图表模板配置 =====
FIGURE_SIZES = {
    'single': (10, 7),         # 单图
    'double': (16, 7),         # 左右对比
    'grid_2x2': (14, 12),      # 2×2 面板
    'grid_3x3': (20, 16),      # 3×3 面板（制度优化）
    'wide': (16, 5),           # 宽屏时序图
    'timeline': (14, 6),       # 训练曲线
}

# 井位标记样式
WELL_MARKER_STYLE = {
    'marker': '^',             # 三角形
    'markersize': 14,
    'markerfacecolor': COLORS['well'],
    'markeredgecolor': 'white',
    'markeredgewidth': 2,
    'zorder': 10,              # 确保显示在最上层
}

# 通道路径样式
CHANNEL_LINE_STYLE = {
    'color': COLORS['channel'],
    'linewidth': 3.0,
    'alpha': 0.8,
    'zorder': 5,
}

# 网格样式
GRID_STYLE = {
    'alpha': 0.3,
    'linewidth': 0.5,
    'color': COLORS['grid'],
}


if __name__ == '__main__':
    # 测试：打印所有配色
    print("=" * 60)
    print("CPEDC PINN 可视化配色方案")
    print("=" * 60)
    
    for category in ['核心颜色', '训练集', '可视化元素', '风险等级', '消融实验', '制度策略']:
        print(f"\n{category}:")
        if category == '核心颜色':
            for k in ['primary', 'accent']:
                print(f"  {k:15s}: {COLORS[k]}")
        elif category == '训练集':
            for k in ['train', 'val', 'test']:
                print(f"  {k:15s}: {COLORS[k]}")
        elif category == '可视化元素':
            for k in ['channel', 'well', 'grid']:
                print(f"  {k:15s}: {COLORS[k]}")
        elif category == '风险等级':
            for k in ['safe', 'warning', 'danger', 'critical']:
                print(f"  {k:15s}: {COLORS[k]}")
        elif category == '消融实验':
            for k in ['pure_ml', 'pinn_const_k', 'pinn_knet', 'pinn_full', 'pinn_no_fourier', 'pinn_no_rar']:
                print(f"  {k:15s}: {COLORS[k]}")
        elif category == '制度策略':
            for k in ['strategy_steady', 'strategy_decay', 'strategy_control']:
                print(f"  {k:15s}: {COLORS[k]}")
    
    print("\n" + "=" * 60)
    print("Colormap 配置:")
    print(f"  渗透率场:     {CMAP_K}")
    print(f"  含水饱和度:   {CMAP_SW}")
    print(f"  压力场:       {CMAP_P}")
    print(f"  连通性热图:   {CMAP_HEAT}")
    print("=" * 60)
