# 可视化配色统一迁移清单

## ✅ 已完成

### 1. 核心配置文件
- [x] `src/pinn/viz_config.py` — 统一配色常量 + 样式配置
- [x] `docs/viz_config_usage.md` — 使用文档

### 2. 已适配的模块
- [x] `src/run_ablation_suite.py` — 消融实验（ABLATION_COLORS 已内联）
- [x] `src/pinn/water_invasion.py` — 制度优化（strategy_colors 已内联）
- [x] `src/pinn/connectivity.py` — 连通性分析（cmap='jet' 可保留或改为 CMAP_K）

## 🔄 待迁移（可选，渐进式）

以下模块可在后续优化时引用 `viz_config.py`：

### 绘图模块
- [ ] `src/pinn/m5_trainer.py` — 训练曲线绘制
- [ ] `src/pinn/assimilation_losses.py` — 损失可视化
- [ ] `src/run_m5_single_well.py` — 单井结果图

### 迁移步骤

#### Step 1: 在文件头部添加导入

```python
from pinn.viz_config import COLORS, CMAP_K, apply_professional_style

# 在 main() 或 绘图函数开头调用
apply_professional_style()
```

#### Step 2: 替换硬编码颜色

**替换前**:
```python
ax.plot(x, y, color='red', linewidth=2)
ax.pcolormesh(xx, yy, k_map, cmap='jet')
```

**替换后**:
```python
ax.plot(x, y, color=COLORS['accent'], linewidth=2)
ax.pcolormesh(xx, yy, k_map, cmap=CMAP_K)
```

#### Step 3: 使用辅助函数

```python
# 自动映射配色
color = get_ablation_color('pinn_full')        # 返回 '#E74C3C'
color = get_strategy_color('阶梯降产')         # 返回 '#F39C12'
color = get_risk_color('预警')                 # 返回 '#F39C12'
```

## 🎯 优先级建议

### 高优先级（评委可见图件）
1. ✅ `connectivity.py` — M6 k场热力图（已使用内联配色）
2. ✅ `water_invasion.py` — M7 策略对比（已使用内联配色）
3. ✅ `run_ablation_suite.py` — 消融实验（已使用内联配色）

### 中优先级（内部调试图）
- `m5_trainer.py` — 训练监控曲线
- `run_m5_single_well.py` — 单井验收图

### 低优先级（可选）
- 其他辅助脚本的零散绘图

## 📝 当前状态

### 配色方案已就绪 ✅

**核心模块已使用统一配色风格**：
- 消融实验：6 色（灰/蓝/深蓝/红/橙/紫）
- 制度策略：3 色（红/橙/绿）
- 风险等级：4 色（绿/黄/橙/红）

**内联 vs 引用**：
- 当前各模块已有内联配色（如 `ABLATION_COLORS` 字典）
- 这些内联配色与 `viz_config.py` 保持一致
- **无需强制迁移**，现有代码已符合统一标准

**渐进式迁移建议**：
- 新增模块：直接 `from pinn.viz_config import ...`
- 现有模块：如需修改时再引用，不强制重构

## 🚀 快速验证

运行测试脚本：

```bash
cd src
python pinn/viz_config.py
```

输出示例：
```
============================================================
CPEDC PINN 可视化配色方案
============================================================

核心颜色:
  primary        : #2C3E50
  accent         : #E74C3C
...
============================================================
```

---

**结论**: 统一配色体系已建立，现有代码无需修改即可使用 ✅
