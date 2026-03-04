# 🚀 CPEDC PINN Web 应用快速启动指南

## 前置条件

1. ✅ Python 3.8+
2. ✅ 虚拟环境已激活
3. ✅ 已运行过至少一次训练（生成 `outputs/` 数据）

## 安装步骤

### 1. 安装 Streamlit 依赖

```bash
cd c:\Users\16281\Desktop\cpedc_project
pip install streamlit>=1.28.0 plotly>=5.18.0 Pillow>=9.0.0
```

或直接安装完整依赖：

```bash
pip install -r requirements.txt
```

### 2. 启动应用

#### Windows 双击启动（推荐）

```
双击运行: app\run_app.bat
```

#### 命令行启动

```bash
cd app
streamlit run streamlit_app.py
```

### 3. 访问应用

浏览器自动打开：**http://localhost:8501**

如果没有自动打开，手动访问上述地址。

## 📂 数据准备

### 必需的输出文件

应用会从 `outputs/mk_pinn_dt_v2/` 目录读取以下文件：

#### 报告文件（`reports/`）
- `M5_training_history.json` — 训练历史
- `M5_inversion_params.json` — 反演参数
- `M6_connectivity_matrix.csv` — 连通性矩阵
- `M6_connectivity_report.md` — 工程解释
- `M7_water_invasion_report.md` — 水侵报告

#### 图片文件（`figs/`）
- `M1_validation.png` — 数据验证
- `M2_geological_domain.png` — 地质域
- `M3_pvt_curves.png` — PVT 曲线
- `M3_relperm_curves.png` — 相渗曲线
- `M5_training_history.png` — 训练曲线
- `M5_qg_comparison_SY9.png` — 产量拟合
- `M5_pwf_inversion_SY9.png` — 井底流压
- `M5_pde_residual_map.png` — PDE 残差
- `M6_k_field_channels.png` — 渗透率场 + 通道
- `M6_connectivity_matrix.png` — 连通性热图
- `M6_sw_evolution.png` — Sw 演化
- `M7_water_invasion_dashboard.png` — 水侵仪表盘
- `M7_strategy_comparison.png` — 策略对比

### 生成这些输出

如果 `outputs/` 目录为空，需要先运行训练：

```bash
# 单井训练（快速测试，~15 分钟）
python src/run_m5_single_well.py --max_steps 10000

# 完整训练（推荐，~1 小时）
python src/run_m5_single_well.py --max_steps 30000

# 连通性分析（需要先训练）
python -c "from pinn.connectivity import ConnectivityAnalyzer; ..."

# 水侵分析（需要先训练）
python -c "from pinn.water_invasion import WaterInvasionAnalyzer; ..."
```

**注意**: 如果部分输出文件缺失，应用仍可正常运行，会显示 "图件将在 xxx 完成后生成" 提示。

## 🎯 页面导航

应用包含 7 个页面：

1. **🏠 首页** — 项目概览 + 指标卡片
2. **📊 数据概览** — 井位分布 + 生产数据
3. **🗺️ 地质域** — 构造面 + 厚度
4. **🔬 物性查询** — PVT 计算器（交互式）
5. **📈 训练监控** — 损失曲线 + 反演参数
6. **🔥 渗透率反演** — k(x,y) 场 + 连通性矩阵（评委重点）
7. **🌊 水侵预警** — 风险仪表盘 + 风险排序
8. **⚙️ 制度优化** — 3 种策略对比 + 决策推荐

## 🔧 配置说明

### 自定义端口

```bash
streamlit run streamlit_app.py --server.port 8502
```

### 禁用浏览器自动打开

```bash
streamlit run streamlit_app.py --server.headless true
```

### 局域网访问

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

## 🐛 常见问题

### 问题 1: ModuleNotFoundError

**原因**: 未激活虚拟环境或缺少依赖

**解决**:
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### 问题 2: 图片不显示

**原因**: 训练输出文件未生成

**解决**: 先运行训练脚本生成 `outputs/` 数据

### 问题 3: 中文乱码

**原因**: matplotlib 字体配置

**解决**: 
- Windows: 已在 `viz_config.py` 中配置 SimHei
- 如仍有问题，安装中文字体

### 问题 4: 端口被占用

**解决**:
```bash
# 查看端口占用
netstat -ano | findstr :8501

# 使用其他端口
streamlit run streamlit_app.py --server.port 8502
```

## 📊 演示建议

### 评委展示流程（推荐）

1. **首页** (30 秒) — 展示 4 个核心指标
2. **渗透率反演** (2 分钟) — k(x,y) 场 + 连通性矩阵（重点）
3. **训练监控** (1 分钟) — 损失曲线 + 反演参数
4. **水侵预警** (1.5 分钟) — 风险仪表盘 + 排序表
5. **制度优化** (1.5 分钟) — 3×3 策略对比 + 决策推荐
6. **物性查询** (1 分钟) — 交互式 PVT 计算器（附加亮点）

**总时长**: 7 分钟

### 技术答辩要点

在各页面停留时强调：

- **页面 5（渗透率反演）**: 
  - "基于 PINN 反演的空间渗透率场"
  - "图论 Dijkstra 算法提取主控通道"
  - "井间连通性量化为工程决策提供依据"

- **页面 7（制度优化）**:
  - "PINN 做 forward simulation，1 分钟完成 3 种策略推演"
  - "传统数值模拟需 2-4 小时，效率提升 100 倍"
  - "基于真实 Peaceman 压差-产量关系，物理可靠"

- **页面 8（水侵预警）**:
  - "提前 3-6 个月预测见水时间"
  - "风险排序指导排水采气措施部署优先级"

## 🎨 技术亮点

- ✅ **前端技术**: Streamlit（Python 原生 Web 框架）
- ✅ **交互图表**: Plotly（缩放/悬停/导出）
- ✅ **响应式布局**: 宽屏适配
- ✅ **模块化架构**: 7 个独立页面 + 可复用组件
- ✅ **数据驱动**: 自动读取训练结果，无需手动更新
- ✅ **专业配色**: 统一视觉风格（红蓝对比、风险分级）

## 📦 部署（可选）

### Streamlit Cloud 免费部署

1. 推送代码到 GitHub
2. 访问 https://streamlit.io/cloud
3. 连接仓库，选择 `app/streamlit_app.py`
4. 自动部署，获得公网访问链接

### 本地服务器

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

局域网其他设备可通过 `http://<本机IP>:8501` 访问。

---

**准备时间**: 5 分钟  
**启动时间**: 10 秒  
**展示效果**: ⭐⭐⭐⭐⭐

祝答辩顺利！🎉
