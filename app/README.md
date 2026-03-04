# CPEDC 马口组 PINN 可视化平台

基于 Streamlit 的交互式 Web 应用，用于展示 PINN 训练结果、渗透率反演、水侵预警等。

## 🚀 快速启动

### 1. 安装依赖

```bash
cd c:\Users\16281\Desktop\cpedc_project
pip install -r requirements.txt
```

### 2. 运行应用

```bash
cd app
streamlit run streamlit_app.py
```

应用将在浏览器中自动打开（默认 http://localhost:8501）

## 📂 目录结构

```
app/
├── streamlit_app.py           # 主入口（首页）
├── pages/                     # 多页面应用
│   ├── 01_📊_数据概览.py
│   ├── 02_🗺️_地质域.py
│   ├── 03_🔬_物性查询.py
│   ├── 04_📈_训练监控.py
│   ├── 05_🔥_渗透率反演.py
│   ├── 06_🌊_水侵预警.py
│   └── 07_⚙️_制度优化.py
└── components/                # 可复用组件
    ├── config_loader.py       # 数据加载器
    └── plotly_charts.py       # 图表函数
```

## 📊 功能模块

### M1 数据概览
- 井位分布地图
- 生产数据统计
- 基本指标展示

### M2 地质域
- 构造面展示
- 厚度分布图

### M3 物性查询
- PVT 交互式计算器
- 相渗曲线可视化

### M4/M5 训练监控
- 训练损失曲线
- 反演参数演化
- 最终结果展示

### M6 渗透率反演
- k(x,y) 场热力图
- 连通性矩阵
- Sw 空间演化

### M7 水侵预警
- 风险仪表盘
- 见水时间预测
- R_w 指数演化

### M7 制度优化
- 3 种策略对比
- 累计产气量预测
- 工程建议

## 🔧 配置说明

应用自动读取 `outputs/` 目录下的训练结果：

- `outputs/mk_pinn_dt_v2/reports/` - JSON 报告
- `outputs/mk_pinn_dt_v2/figs/` - 图表文件

请确保先运行训练脚本生成输出。

## 📝 开发建议

1. **添加新页面**: 在 `pages/` 目录创建新文件，文件名格式 `XX_图标_名称.py`
2. **自定义图表**: 在 `components/plotly_charts.py` 添加新函数
3. **数据加载**: 使用 `components.config_loader.get_loader()` 统一加载

## 🌐 部署

生产环境部署可使用：

```bash
# Streamlit Cloud (推荐)
# 或本地服务器
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 📖 技术栈

- **Streamlit** 1.28+ - Web 框架
- **Plotly** 5.18+ - 交互式图表
- **Pandas** - 数据处理
- **NumPy** - 数值计算
- **Pillow** - 图像加载

---

**CPEDC 2026** | Powered by Physics-Informed Neural Networks
