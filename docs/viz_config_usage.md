# CPEDC 统一配色方案使用指南

## 📌 配置文件位置

`src/pinn/viz_config.py`

## 🎨 使用方法

### 1. 导入配置

```python
from pinn.viz_config import (
    COLORS, CMAP_K, CMAP_SW, CMAP_P, CMAP_HEAT,
    apply_professional_style,
    get_strategy_color, get_risk_color, get_ablation_color,
    WELL_MARKER_STYLE, CHANNEL_LINE_STYLE
)
```

### 2. 应用全局样式

```python
# 在绘图脚本开头调用
apply_professional_style()

# 后续所有 matplotlib 图件自动应用统一样式
fig, ax = plt.subplots()
ax.plot(x, y, color=COLORS['accent'])
```

### 3. 使用统一配色

#### 场景 1：渗透率场绘制

```python
# 在 connectivity.py 中
im = ax.pcolormesh(xx, yy, k_map, cmap=CMAP_K, 
                   norm=mcolors.LogNorm(vmin=0.1, vmax=k_max))

# 井位标记
ax.plot(wx, wy, **WELL_MARKER_STYLE)

# 主控通道
ax.plot(path_x, path_y, **CHANNEL_LINE_STYLE)
```

#### 场景 2：消融实验对比

```python
# 在 run_ablation_suite.py 中
from pinn.viz_config import get_ablation_color

for exp_name in ['pure_ml', 'pinn_const_k', 'pinn_full']:
    color = get_ablation_color(exp_name)
    ax.plot(t, qg, color=color, label=exp_name)
```

#### 场景 3：制度优化策略

```python
# 在 water_invasion.py 中
from pinn.viz_config import get_strategy_color

for strategy in ['稳产方案', '阶梯降产', '控压方案']:
    color = get_strategy_color(strategy)
    ax.plot(t, Gp, color=color, label=strategy)
```

#### 场景 4：风险等级着色

```python
# 在风险评估中
from pinn.viz_config import get_risk_color

risk_level = '预警'
color = get_risk_color(risk_level)
ax.fill_between(t, 0, R_w, color=color, alpha=0.3)
```

### 4. Colormap 使用

```python
# 渗透率场
ax.pcolormesh(xx, yy, k_map, cmap=CMAP_K)

# 含水饱和度
ax.pcolormesh(xx, yy, sw_map, cmap=CMAP_SW, vmin=0.2, vmax=0.6)

# 压力场
ax.pcolormesh(xx, yy, p_map, cmap=CMAP_P)

# 连通性热图
ax.imshow(C_matrix, cmap=CMAP_HEAT)
```

## 📊 统一配色方案

### 核心颜色
- **Primary** (#2C3E50): 深蓝灰 - 观测值/文本
- **Accent** (#E74C3C): 红色 - 预测值/高亮

### 训练集划分
- **Train** (#3498DB): 蓝色
- **Val** (#F39C12): 橙色
- **Test** (#27AE60): 绿色

### 风险等级
- **Safe** (#27AE60): 绿色
- **Warning** (#F39C12): 橙色
- **Danger** (#E67E22): 深橙
- **Critical** (#E74C3C): 红色

### 消融实验（6 组）
- **pure_ml** (#95A5A6): 灰色 - 基线
- **pinn_const_k** (#3498DB): 蓝色
- **pinn_knet** (#2980B9): 深蓝
- **pinn_full** (#E74C3C): 红色 - 最优
- **pinn_no_fourier** (#F39C12): 橙色
- **pinn_no_rar** (#9B59B6): 紫色

### 制度优化策略（3 种）
- **稳产方案** (#E74C3C): 红色 - 高风险
- **阶梯降产** (#F39C12): 橙色 - 平衡
- **控压方案** (#27AE60): 绿色 - 保守

## 🔧 迁移现有代码

### Step 1: 在文件头部导入

```python
from pinn.viz_config import COLORS, CMAP_K, apply_professional_style
apply_professional_style()
```

### Step 2: 替换硬编码颜色

**替换前**:
```python
colors = ['blue', 'orange', 'green', 'red']
ax.plot(x, y, color='red')
```

**替换后**:
```python
ax.plot(x, y, color=COLORS['accent'])
```

### Step 3: 使用辅助函数

```python
# 自动映射
color = get_strategy_color('稳产方案')  # 返回 '#E74C3C'
```

## 📈 效果

- ✅ 所有图件视觉风格统一
- ✅ 配色符合专业审美（红蓝对比、冷暖分明）
- ✅ 支持风险等级直观识别（绿→黄→橙→红）
- ✅ 消融实验可区分（6 种颜色不重复）
- ✅ 支持色盲友好（避免红绿混淆）

---

**更新日期**: 2026-02-13  
**适用版本**: v3.1+
