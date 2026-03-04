# CPEDC 创新组 — 深层碳酸盐岩气藏 PINN 数字孪生系统

## 最后更新时间（UTC+8）/19：50

2026-02-21（第三次 — 仓库结构整理）

## 最近更新摘要

### 2026-02-21（仓库结构整理）

- **测试文件归位**：`src/m5/test_dp_freeze.py` → `tests/test_dp_freeze.py`
- **工具脚本归位**：`run_utf8.ps1` → `scripts/run_utf8.ps1`
- **PDF 提取结果归位**：`outputs/pdf_text_extract/` → `data/pdf_extract/`
- **文件名规范**：`docs/chatgpt view.md` → `docs/chatgpt_view.md`（消除空格）
- **清理缓存**：删除 `src/__pycache__/`（旧扁平结构残留）
- **README 更新**：项目结构树与实际文件同步

### 2026-02-21（根因修复）

- **M5 根因1**：`k_eff_mD` 从 `5.0` 改回 `0.5`（`k_frac_init=4 mD`）；SY9 局部 36.8 mD 是裂缝峰值，全域先验取其余 5 口井均值。
- **M5 根因2**：所有 Stage 显式设置 `pde` 权重（A=0.001 ~ D=0.003），阻止 M4 权重通过合并链泄漏。
- **M5 阶段权重 v4.2**：PDE 权重 0.0005~0.003；qg 权重 50→800（A→D）。
- **M6 消融公平性**：所有 PINN 变体统一 `loss.physics.base_weight=1.0`；2×2 正交消融。

## 模块架构

| 模块 | 功能 | 关键交付物 | 状态 |
|------|------|-----------|------|
| **M1** | 数据层与坐标统一 | 井轨迹、MK 段代表点、SY9 生产数据 | ✅ 完成 |
| **M2** | 弱空间地质域构建 | 顶底面 Kriging、厚度场、PINN 配点网格 | ✅ 完成 |
| **M3** | 物性模块（PVT + 相渗） | GasPVT / RelPermGW 可复用 API | ✅ 完成 |
| **M4** | PINN 增强基线 (v2) | Fourier Features + 6×256 ResNet + tanh 参数化 | ✅ v2 |
| **M5** | 井—藏耦合同化 + k(x,y) 反演 | Bg 修复、k-net 空间反演、WHP→p_wf、ReLoBRaLo v2 | ✅ v2 |
| **M6** | 消融(5组) + UQ + 连通性 | C_ij 矩阵、主控通道、k(x,y) 热力图、5 组消融 | ✅ 一等奖 |
| **M7** | 水侵预警 + 制度优化 | R_w 风险指数、见水时间、稳产/阶梯/控压方案对比 | ✅ 一等奖 |

---

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

# 2. 原始数据放在 data/raw/ 目录（附表 1~10 等 CSV）

# 3. 运行完整流程（M1~M6）
python src/main.py --stage all

# 分步执行（可从任意阶段开始，节省时间）
python src/main.py --stage m1     # M1: 数据清洗与坐标统一
python src/main.py --stage m2     # M2: 地质域构建（需先完成 M1）
python src/main.py --stage m3     # M3: PVT + 相渗物性模块
python src/main.py --stage m4     # M4: PINN 基线训练（需先完成 M2，推荐 GPU）
python src/main.py --stage m5     # M5: 井—藏耦合同化（需 M2 配点网格；可从 M5 续跑，不重跑 M1~M4）
python src/main.py --stage m6     # M6: 消融/UQ/连通性（简化版，见下 M6 专用脚本）

# M5 专用：单井同化 + k(x,y) 反演（推荐）
python src/m5/run_m5_single_well.py --well SY9 --steps 50000 --device cuda

# M6 专用：消融、UQ、连通性（推荐）
python src/m6/run_ablation_suite.py --steps 20000 --device cuda
python src/m6/run_uq_ensemble.py --n 5 --device cuda

# 仅验收（不重新执行）
python src/main.py --validate-only
python src/main.py --validate-only --stage m3

# 执行流程但跳过验收
python src/main.py --stage all --skip-validation
```

### M4 专用脚本（完整训练 + 验收，推荐）
```bash
python src/m4/run_m4_baseline.py                    # 默认步数、自动选设备、训练后验收
python src/m4/run_m4_baseline.py --steps 50000      # 指定步数
python src/m4/run_m4_baseline.py --device cuda      # 指定 GPU（或 cpu）
python src/m4/run_m4_baseline.py --skip-validation  # 只训练不验收

# 仅执行 M4 验收（不重新训练）
python src/m4/validate_m4.py                         # 自动选设备；CUDA 不可用时回退 CPU
python src/m4/validate_m4.py --device cpu            # 强制 CPU
```

> **GPU 训练**：M4/M5/M6 自动检测 CUDA。请确保安装了 CUDA 版 PyTorch（如 `pip install torch --index-url https://download.pytorch.org/whl/cu128`）。  
> **M4 验收与设备**：验收默认优先 CUDA；若当前 PyTorch 与显卡架构不兼容（如部分新卡 sm_120），会自动回退到 CPU 完成验收。训练时可通过 `--device cpu` 与验收保持一致。  
> **可复现性**：若启用 `reproducibility.deterministic`，建议设置环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（Windows PowerShell: `$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"`）。  
> **Windows**：M5 的 `torch.compile` 在 Windows 上会自动跳过（Triton 仅支持 Linux），使用 eager 模式，训练正常。  
> 已验证设备：NVIDIA RTX 5060 Laptop (8 GB)，支持 Mixed Precision (AMP) + cudnn.benchmark。

---

## 项目结构

```
cpedc_project/
├── config.yaml                        # 主配置（全项目统一参数，含 m5_config / m6_config）
├── requirements.txt                   # Python 依赖
├── README.md
│
├── src/                               # ===== 源码 =====
│   ├── main.py                        # 主程序入口（M1~M7）
│   ├── utils.py                       # 配置加载、日志、工具函数
│   │
│   ├── m1/                            # M1：数据层与坐标统一
│   │   ├── __init__.py
│   │   ├── m1_data_processor.py       # 数据与坐标统一
│   │   └── validate_m1.py             # M1 验收
│   ├── m2/                            # M2：弱空间地质域构建
│   │   ├── __init__.py
│   │   ├── m2_geo_builder.py          # 地质域与网格
│   │   └── validate_m2.py             # M2 验收
│   ├── m3/                            # M3：物性模块（PVT + 相渗）
│   │   ├── __init__.py
│   │   └── validate_m3.py             # M3 验收
│   ├── m4/                            # M4：PINN 增强基线
│   │   ├── __init__.py
│   │   ├── run_m4_baseline.py         # 训练 + 出图 + 验收
│   │   └── validate_m4.py             # M4 验收（CUDA 不可用时回退 CPU）
│   ├── m5/                            # M5：单井同化与反演
│   │   ├── __init__.py
│   │   ├── run_m5_single_well.py      # 单井同化入口
│   │   └── diagnose_k_frac_gradient.py  # k_frac 梯度诊断
│   ├── m6/                            # M6：消融、UQ、连通性
│   │   ├── __init__.py
│   │   ├── run_ablation_suite.py      # 消融实验（5 组）
│   │   ├── run_uq_ensemble.py         # 不确定性量化（多种子 ensemble）
│   │   └── validate_m6.py             # M6 验收
│   │
│   ├── physics/                       # M3 物性模块（CPU 端）
│   │   ├── __init__.py
│   │   ├── units.py                   # 单位转换（MPa/K/kg·m⁻³）
│   │   ├── pvt.py                     # GasPVT：Z / Bg / cg / ρ / αT 多温度 2D 插值
│   │   ├── relperm.py                 # RelPermGW：气水相渗 PCHIP 插值
│   │   ├── pvt_validate.py            # PVT 验证脚本
│   │   └── relperm_validate.py        # 相渗验证脚本
│   │
│   └── pinn/                          # M4 基线 + M5/M6/M7 扩展（GPU 端）
│       ├── __init__.py
│       ├── model.py                   # PINNNet（MLP + 物理有界输出）
│       ├── losses.py                  # M4 损失（IC/BC/PDE/data/sw_phys）
│       ├── trainer.py                 # M4 分阶段训练器（Warmup+Cosine）
│       ├── sampler.py                 # 域内/边界/初始/井数据采样（2.5D）
│       ├── torch_physics.py           # 可微分物性：TorchPVT / TorchRelPerm
│       ├── viz_config.py              # 统一可视化配置
│       ├── m5_model.py                # M5PINNNet（场网络 + 井模型）
│       ├── m5_trainer.py              # M5 训练器（同化 + 反演 + 报告）
│       ├── well_model.py              # Peaceman WI、p_wf、高斯源项
│       ├── assimilation_losses.py     # M5/M6 完整两相守恒 PDE + L_qg/L_whp
│       ├── relobralo.py               # ReLoBRaLo 自适应损失权重
│       ├── rar_sampler.py             # 残差驱动自适应采样（RAR）
│       ├── compute_priors.py          # 数据驱动先验（附表3+4+9）
│       ├── connectivity.py            # M6 连通性矩阵 C_ij、主控通道
│       ├── xpinn.py                   # M6 XPINN/APINN 域分解
│       ├── uq_runner.py               # M6 多种子 UQ ensemble
│       └── water_invasion.py          # M7 水侵预警、制度优化
│
├── data/                              # ===== 数据 =====
│   ├── raw/                           # 原始数据（附表 1~10：井位/井斜/测井/分层/PVT/相渗/生产等）
│   ├── staged/                        # 中间数据（wells_staged.csv）
│   ├── clean/                         # M1 产出：清洗后数据
│   │   ├── wellpath_stations.csv      #   井眼 3D 轨迹点
│   │   ├── mk_interval_points.csv     #   MK 段代表点
│   │   ├── production_SY9.csv         #   SY9 日生产数据
│   │   └── normalization_params.json  #   归一化参数
│   └── pdf_extract/                   # 赛题 PDF 文本提取结果
│
├── geo/                               # ===== 地质数据（M2 生成） =====
│   ├── surfaces/                      # MK 顶底面、厚度（mk_top_surface.csv 等）
│   ├── grids/                         # 配点网格、边界点（collocation_grid.csv 等）
│   └── boundary/                      # 模型边界（model_boundary.csv）
│
├── outputs/<experiment_name>/         # ===== 实验输出（按 meta.experiment_name 分目录） =====
│   ├── reports/                       # M1~M7 验收/质量/反演/UQ/水侵报告
│   ├── figs/                          # M1~M7 图件（训练曲线/对比图/热力图等）
│   ├── ckpt/                          # PINN 检查点（pinn_best/final.pt, m5_pinn_*.pt）
│   ├── M6_ablation/                   # 消融实验子目录
│   └── resolved_config_m5.json        # M5 训练时的完整合并配置快照
│
├── app/                               # ===== Streamlit 可视化仪表盘 =====
│   ├── streamlit_app.py               # 入口
│   ├── pages/                         # 多页面（数据概览/地质域/物性/训练/反演/水侵/优化）
│   ├── components/                    # 复用组件（config_loader / plotly_charts）
│   └── .streamlit/config.toml         # Streamlit 配置
│
├── scripts/                           # ===== 工具与可复现脚本 =====
│   ├── fit_z_factor_least_squares.py  # Z 因子最小二乘拟合（附表5-2）
│   ├── check_sy9_missing.py           # SY9 缺失数据检查
│   ├── generate_html_preview.py       # 输出 HTML 预览
│   ├── json_to_tree.py                # JSON 配置树形打印
│   ├── quick_view_images.py           # 快速查看图片
│   └── run_utf8.ps1                   # Windows UTF-8 启动脚本
│
├── tests/                             # ===== 测试 =====
│   └── test_dp_freeze.py             # dp_wellbore 冻结 TDD 验证
│
├── docs/                              # ===== 项目文档 =====
│   ├── PROJECT_OBJECTIVES.md          # 项目实施计划书
│   ├── M1-M7_模块核心作用与实现.md    # 模块设计详述
│   ├── M5_v35_fix_summary.md          # M5 v3.5 修复总结
│   ├── M5_latest_result_analysis.md   # M5 最新结果分析
│   ├── M5_本次输出结果分析.md         # M5 输出分析
│   ├── RTX50_PyTorch_升级说明.md      # RTX 50 系列兼容说明
│   ├── chatgpt_view.md                # ChatGPT 设计讨论
│   ├── viz_config_usage.md            # 可视化配置使用说明
│   └── viz_migration_checklist.md     # 可视化迁移检查清单
│
└── logs/                              # 运行日志（自动生成，可清理）
```

---

## M1 数据层与坐标统一

- 最小曲率法计算井眼 3D 轨迹
- MK 组顶底界定位（分层数据 + 井斜插值）
- SY9 日生产数据清洗（缺失插补、时间连续性）
- 坐标基准：MSL 向上为正，$z = z_{\text{datum}} - \text{TVD}$

### 验收标准

- ✅ 井位散点图正常
- ✅ MK 厚度 > 0，范围合理
- ✅ 井眼轨迹 MD 单调递增
- ✅ 生产数据时间连续

---

## M2 弱空间地质域构建

- 凸包 + 缓冲区构建模型边界
- Kriging 插值 MK 顶底面
- 厚度场 $h(x,y) = z_{\text{top}}(x,y) - z_{\text{bot}}(x,y)$
- PINN 配点网格（均匀 + 井周加密）

### 验收标准

- ✅ 边界包含所有井位
- ✅ 顶底厚等值图生成
- ✅ 交叉验证 MAE < 20 m，RMSE < 30 m
- ✅ 配点网格井周加密清晰

---

## M3 物性模块（PVT + 气水相渗）

### 输入 / 输出 API

| 输入 | 输出 | 单位 | 方法 |
|------|------|------|------|
| p (MPa), T (℃) | Z 偏差系数 | — | `GasPVT.z(p, T)` |
| p (MPa), T (℃) | Bg 体积系数 | m³/m³ | `GasPVT.bg(p, T)` |
| p (MPa), T (℃) | cg 压缩系数 | 1/MPa | `GasPVT.cg(p, T)` |
| p (MPa), T (℃) | ρ 密度 | kg/m³ | `GasPVT.rho(p, T)` |
| p (MPa), T (℃) | αT 热膨胀系数 | 1/℃ | `GasPVT.alpha_T(p, T)` |
| p (MPa), T (℃) | 全部物性 | dict | `GasPVT.query_all(p, T)` |
| Sw (分数) | krw 水相相渗 | — | `RelPermGW.krw(sw)` |
| Sw (分数) | krg 气相相渗 | — | `RelPermGW.krg(sw)` |
| Sw (分数) | dkrw/dSw 导数 | — | `RelPermGW.dkrw_dsw(sw)` |
| Sw (分数) | dkrg/dSw 导数 | — | `RelPermGW.dkrg_dsw(sw)` |
| — | 端点参数 | tuple | `RelPermGW.endpoints()` |

### 插值方法

- **压力方向**：PCHIP（分段三次 Hermite）— 保持数据单调性，避免过冲
- **温度方向**：线性插值（5 个温度：16.5 / 46.5 / 78.0 / 109.0 / 140.32 ℃）
- **压力范围**：12.0 ~ 75.7 MPa（含地层压力模拟点）
- **相渗**：PCHIP 插值 + 端点 / 单调 / 非负强制约束；支持解析导数

### M3→M4 数据管道（审查修复）

M3 GasPVT 可导出多项式系数，供 M4 TorchPVT 使用，建立完整数据链：

```python
# 在 M3 初始化后导出系数
from physics.pvt import GasPVT
gas_pvt = GasPVT(config)
coeffs = gas_pvt.export_all_polynomial_coeffs(degree=3, T=140.32, save_path='pvt_coeffs.json')

# M4 TorchPVT 接收外部系数（可选）
from pinn.torch_physics import TorchPVT
torch_pvt = TorchPVT(config, pvt_coeffs=coeffs)  # 使用 M3 拟合系数
```

数据链：**附表5 实测 → M3 PCHIP → polyfit → pvt_coeffs → TorchPVT → autograd → PDE**

### 端点参数

| 参数 | 值 |
|------|------|
| 束缚水饱和度 Swr | 0.2600 |
| 残余气饱和度 Sgr | 0.0620 |
| 最大水相相渗 krw_max | 0.4800 |
| 最大气相相渗 krg_max | 0.6750 |

---

## M4 PINN 基线（最小闭环）

M4 实现**完整训练流程**：分阶段 PINN 训练 → 保存 checkpoint → 训练曲线与 SY9 压力对比图 → M4 验收（压力 MAPE、Sw 范围、报告）。  
**入口**：`python src/m4/run_m4_baseline.py`（与 `python src/main.py --stage m4` 等价）；**前置**：需先完成 M2 生成配点网格 `geo/grids/collocation_grid.csv`。

### 网络结构 (v2 增强)

| 项目 | 配置 |
|------|------|
| 输入 | (x, y, t) → Fourier Feature Encoding (32 频率) |
| 主干 | ResNet [256×6] + LayerNorm + Tanh + 残差连接 |
| 输出 | p = p_center + p_scale·tanh(·)，**p ∈ [15, 90] MPa**；Sw = σ(· + bias_init) ∈ (0, 1)，Sw 初值 ≈ 0.26 |
| 参数量 | ~416k（M4 基线无 k_net；k(x,y) 为 M5 扩展） |
| 物理保证 | tanh 强制 p 有界；Sigmoid 强制 Sw ∈ (0, 1)；井筒压差 dp_wellbore 可配置/可学习 |

### 损失函数

$$\mathcal{L} = \lambda_{\text{IC}}\mathcal{L}_{\text{IC}} + \lambda_{\text{BC}}\mathcal{L}_{\text{BC}} + \lambda_{\text{PDE}}\mathcal{L}_{\text{PDE}} + \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \lambda_{\text{sw\_phys}}\mathcal{L}_{\text{sw\_phys}}$$

| 损失项 | 说明 |
|--------|------|
| $\mathcal{L}_{\text{IC}}$ | t = 0 时 p ≈ p_init (76 MPa)，Sw ≈ Swc (0.26) |
| $\mathcal{L}_{\text{BC}}$ | 外边界定压 p\|∂Ω = p_boundary |
| $\mathcal{L}_{\text{PDE}}$ | **两相守恒 PDE 残差**（气相 + 水相）：TorchPVT / TorchRelPerm 可微分物性，归一化坐标缩放，残差截断防爆炸；含厚度修正与 ∇kr 交叉项 |
| $\mathcal{L}_{\text{data}}$ | SY9 井口压力锚点（油管压力，~1025 条）；验收时用地层压力 − dp_wellbore 与观测对比 |
| $\mathcal{L}_{\text{sw\_phys}}$ | Sw 物理弱约束：Swc 初值、dSw/dt ≥ 0、Sw ≤ 1−Sgr，作为水相 PDE 的安全网 |

### 分阶段训练策略

由 `config.yaml` 中 `m4_config.training_stages` 驱动；学习率为 **Warmup + Cosine**（`train.scheduler.warmup_steps` / `min_lr`）。

| 阶段 | 步数占比 | 活跃损失 | 说明 |
|------|---------|---------|------|
| **Stage A** | 0–15% | 仅 IC + BC | 纯预训练，pde/data/sw_phys 权重均为 0 |
| **Stage B** | 15–40% | + PDE、data、sw_phys（λ 渐进） | PDE 从 0.001→0.1，data 0→0.5，sw_phys 0.5→1.0 |
| **Stage C** | 40–70% | 全部，权重继续渐变 | PDE 0.1→0.5，data 1→3，sw_phys 1.5→2.0 |
| **Stage D** | 70–100% | 全部微调 | 固定权重：ic/bc/pde/data/sw_phys = 1/1/1/5/2 |

### 验收与设备

- **验收标准**：SY9 压力 MAPE < 30%（井口压力）；Sw 输出在 [0, 1]；详见 `outputs/.../reports/M4_validation_report.md`。
- **设备**：训练与验收均可通过 `--device cuda` 或 `--device cpu` 指定；验收时若 CUDA 与当前显卡架构不兼容会**自动回退 CPU**，保证能跑完并出报告。

### GPU 加速

- 设备自动检测；Mixed Precision（AMP）+ GradScaler；cudnn.benchmark；batch_size 由 config 控制。已验证：NVIDIA RTX 5060 Laptop 8 GB，CUDA 12.8。

---

## M5 井—藏耦合同化与反演（一等奖版）

- **井模型 (v2)**：修复 Bg 双重除法致命 Bug；Peaceman 公式 $q_{g,surface} = WI \cdot k_{rg}/\mu_g \cdot (p_{cell} - p_{wf}) / B_g$；压力依赖粘度 $\mu_g(p)$；相渗统一由 **TorchRelPerm** 计算（Corey 指数 ng=1.08、nw=4.41，附表7 拟合），well_model 不再独立硬编码。
- **可微分物性**：`torch_physics.py` 提供 TorchPVT（$\rho_g(p)$, $B_g(p)$, $\mu_g(p)$, $c_g(p)$）与 TorchRelPerm（$k_{rg}(S_w)$, $k_{rw}(S_w)$），全 autograd 兼容，供两相 PDE 与井模型使用。
- **完整两相守恒 PDE**（M5 同化损失）：
  - **气相**：$\phi h[-\rho_g \partial S_w/\partial t + (1-S_w)\rho_g c_g \partial p/\partial t] = \nabla\cdot(k k_{rg} h/\mu_g \nabla p) + q_g$
  - **水相**：$\phi h \rho_w \partial S_w/\partial t = \nabla\cdot(k k_{rw} h/\mu_w \nabla p) + q_w$
  - 含 $\nabla k$、$\nabla k_r$、厚度修正与井源项。
- **k(x,y) 空间反演**：PermeabilityNet 子网络；TV/Laplacian/井点先验正则化。
- **WHP→p_wf 转换**：可学习井筒压差 $\Delta p_{wb}$，不直接用油管压力约束 p_cell。
- **反演参数**：$k_{\text{eff}}$、$f_{\text{frac}}$、$\Delta p_{wb}$、$k(x,y)$ 场。

### 运行方式

- 主流程：`python src/main.py --stage m5` 或 `--stage all`
- 推荐：`python src/m5/run_m5_single_well.py --well SY9 --steps 50000 --device cuda`

### 输出

- `figs/M5_qg_comparison_*.png`、`M5_pwf_inversion_*.png`、`M5_training_history.png`、`M5_pde_residual_map.png`
- `reports/M5_validation_report.md`、`M5_inversion_params.json`、`resolved_config_m5.json`
- 训练结束后自动调用 **M6 连通性分析** 与 **M7 水侵预警**，生成对应图表与报告。

---

## M6 消融、UQ 与连通性分析（一等奖版）

- **强非均质**：可选 XPINN/APINN，`config.yaml` 中 `m6_config.domain_decomposition`。
- **消融实验**：5 组（`pure_ml`、`pinn_const_k`、`pinn_knet`、`pinn_full`、`pinn_no_fourier`），override 机制已修复，每组建议 ≥20000 步。
- **UQ**：多随机种子 ensemble，**多维度扰动**（k 先验、p_boundary、f_frac），输出 P10/P50/P90 与参数后验。
- **连通性分析**（`connectivity.py`）：
  - 基于反演 k(x,y) 场构建网格图，边权 = 渗流阻力 $1/(k\cdot h)$；
  - Dijkstra 最短路径 → 井间连通性矩阵 $C_{ij}$、主控流动通道；
  - 输出：**k(x,y) 热力图 + 井位 + 通道叠加**、**$C_{ij}$ 热力图**、$S_w$ 时空演化、`M6_connectivity_matrix.csv`。

### 运行方式

- 主流程：`python src/main.py --stage m6`
- 消融套件：`python src/m6/run_ablation_suite.py --steps 20000 --device cuda`
- UQ：`python src/m6/run_uq_ensemble.py --n 5 --device cuda`
- **M5 训练完成后** `generate_report()` 会自动调用连通性分析，无需单独运行。

### 输出

- `figs/M6_ablation_*.png`、`M6_uq_*.png`、**`M6_k_field_channels.png`**、**`M6_connectivity_matrix.png`**、**`M6_sw_evolution.png`**
- `reports/M6_ablation_report.md`、`M6_uq_report.md`、**`M6_connectivity_matrix.csv`**

---

## M7 水侵预警与制度优化（一等奖版）

- **水侵风险指数**：$R_w(t) = (S_w(x_w,t) - S_{wc})/(1 - S_{wc} - S_{gr})$，由 PINN 推理 $S_w$ 时间序列得到。
- **见水时间预测**：各井 $S_w$ 超过设定阈值（如 0.4）的时刻。
- **制度优化**（`water_invasion.py`，用 PINN 作 forward simulator）：
  - **稳产方案**：维持当前产量；
  - **阶梯降产**：分阶段降低产量，延缓水侵；
  - **控压方案**：限制最低 $p_{wf}$，延长无水采气期。
  - 输出各策略下的累计产气 $G_p$、$S_w$ 演化、$p_{wf}$ 对比图与文字建议。

### 运行方式

- **M5 训练完成后** `generate_report()` 会自动调用水侵分析；
- 或单独使用：`WaterInvasionAnalyzer(model, sampler, config).generate_all(output_dir)`。

### 输出

- `figs/M7_water_invasion_dashboard.png`（4 面板：$S_w(t)$、$R_w(t)$、见水时间柱状图、压力-Sw 双轴）
- `figs/M7_strategy_comparison.png`（稳产/阶梯/控压 产量与 $G_p$、$S_w$、$p_{wf}$ 对比）
- `reports/M7_water_invasion_report.md`

---

## 核心公式

### 井眼轨迹（最小曲率法）

$$\Delta \text{TVD} = \frac{1}{2}\Delta \text{MD}(\cos\alpha_1 + \cos\alpha_2) \cdot RF$$

### 厚度场

$$h(x,y) = z_{\text{top}}(x,y) - z_{\text{bot}}(x,y)$$

### PVT 2D 插值

沿 $p$ 方向 PCHIP（保持单调）+ 沿 $T$ 方向线性插值 → $f(p, T)$

### 两相流守恒方程（2.5D 厚度加权）

$$\nabla \cdot (h \rho_\alpha \mathbf{v}_\alpha) = \text{源项}_\alpha$$

### Darcy 速度

$$\mathbf{v}_\alpha = -\frac{k \cdot k_{r\alpha}(S_w)}{\mu_\alpha}(\nabla p - \rho_\alpha \mathbf{g})$$

### 井源项（高斯核）

井产量在空间上按高斯核分布到配点，参与 PDE 残差计算。

---

## 输出文件一览

**路径约定**：所有报告在 `outputs/<experiment_name>/reports/`，所有图件在 `outputs/<experiment_name>/figs/`，检查点在 `outputs/<experiment_name>/ckpt/`。

| 文件 | 说明 | 来源 |
|------|------|------|
| `data/clean/wellpath_stations.csv` | 井眼 3D 轨迹点 | M1 |
| `data/clean/mk_interval_points.csv` | MK 段代表点 | M1 |
| `data/clean/production_SY9.csv` | SY9 日生产数据 | M1 |
| `data/clean/normalization_params.json` | 坐标归一化参数 | M1 |
| `geo/surfaces/mk_*_surface.csv`、`mk_thickness.csv` | MK 顶/底面、厚度场 | M2 |
| `geo/grids/collocation_grid.csv`、`boundary_points.csv` | PINN 配点、边界点 | M2 |
| `geo/boundary/model_boundary.csv` | 模型边界多边形 | M2 |
| `reports/M1_*.md`、`figs/M1_validation.png` | M1 报告与验收图 | M1 |
| `reports/M2_*.md`、`figs/M2_*.png` | M2 报告与地质/配点/不确定性图 | M2 |
| `reports/M3_*.md`、`figs/M3_*.png` | M3 PVT/相渗报告与曲线 | M3 |
| `reports/M4_validation_report.md`、`figs/M4_*.png` | M4 验收报告与训练/压力图 | M4 |
| `ckpt/pinn_best.pt`、`pinn_final.pt` | M4 PINN 检查点 | M4 |
| `ckpt/m5_pinn_best.pt`、`m5_pinn_final.pt` | M5 PINN 检查点 | M5 |
| `figs/M5_*.png`、`reports/M5_*.md` | M5 产量对比/压力反演/训练曲线/PDE残差 | M5 |
| `reports/M5_inversion_params.json` | M5 反演参数快照 | M5 |
| `figs/M6_*.png`、`reports/M6_*.md` | M6 消融/UQ/连通性/Sw演化图与报告 | M6 |
| `reports/M6_connectivity_matrix.csv` | 井间连通性矩阵 | M6 |
| `figs/M7_*.png`、`reports/M7_*.md` | M7 水侵仪表盘与制度对比 | M7 |
| `resolved_config_m5.json` | M5 训练时完整合并配置 | M5 |

---

## 依赖

| 包 | 用途 | 最低版本 |
|----|------|---------|
| numpy | 数值计算 | ≥ 1.21 |
| pandas | 数据处理 | ≥ 1.3 |
| matplotlib | 可视化 | ≥ 3.4 |
| scipy | PCHIP 插值 | ≥ 1.7 |
| pyyaml | 配置加载 | ≥ 5.4 |
| shapely | 几何处理 | ≥ 1.8 |
| pykrige | Kriging 插值 | ≥ 1.6 |
| **torch** | **PINN 训练（推荐 CUDA 版）** | **≥ 2.0** |

---

## 配置说明

所有参数集中在 `config.yaml`，关键字段：

| 配置路径 | 说明 |
|----------|------|
| `meta.experiment_name` | 输出子目录名（如 `mk_pinn_dt_v2`） |
| `paths.raw_data` / `paths.clean_data` / `paths.geo_data` | 原始/清洗/地质数据目录 |
| `paths.outputs` / `paths.checkpoints` / `paths.reports` / `paths.figures` | 输出/检查点/报告/图表目录（支持 `${meta.experiment_name}`） |
| `coordinate_system` | 坐标基准（MSL）、z 向上为正、气水界面等 |
| `mk_formation` | 地层参数（压力/温度梯度、初始压力 76 MPa、储层温度 **140.32°C** 与 PVT 实测一致） |
| `data` | 数据模式、主井、井列表及各类数据源路径（PVT、相渗、生产数据等） |
| `m3_config` | 物性模块插值方法与单位因子 |
| `m4_config` | PINN 训练阶段与验收阈值 |
| `model.architecture` | 网络结构（隐层/激活/LayerNorm） |
| `train` | 训练参数（步数/学习率/batch_size/优化器） |
| `runtime` | 设备（cuda/cpu）/ 混合精度 / 线程数 |
| `reproducibility` | 随机种子、确定性、cudnn.benchmark 等 |
| **`m5_config`** | **井模型（r_w, skin）、p_wf 网络、源项高斯核、ReLoBRaLo、RAR** |
| **`m6_config`** | **域分解（none/xpinn/apinn）、XPINN/APINN 子网、UQ ensemble 数与种子** |

---

## 物理参数数据溯源

关键常数均来自附表实测或最小二乘拟合，保证数据→物性→PDE 同一数据链：

| 参数 | 值 | 来源 |
|------|-----|------|
| 储层温度 | 140.32°C | 附表5 PVT 实测最高温度 |
| Bg 参考值 | 0.002577 m³/m³ | 附表5-4 (75.7 MPa, 140.32°C) |
| Corey 指数 ng/nw | 1.08 / 4.41 | 附表7 联合最小二乘拟合 |
| Z 因子多项式 | 三次拟合 | 附表5-2 恒质膨胀数据 |
| 地层水粘度 μw | 0.25 mPa·s | 140°C 地层水 + 矿化度修正 |

先验值计算：`src/pinn/compute_priors.py` 从附表3+4+9 自动计算 k_eff、f_frac；`scripts/fit_z_factor_least_squares.py` 可复现 Z 因子拟合。

---

## 使用 Cursor 时选用已有 .venv

若在 Cursor 中使用 PyCharm 创建的 `.venv`：

1. **选择解释器**：`Ctrl+Shift+P` → `Python: Select Interpreter` → 选择项目下 `.venv\Scripts\python.exe`（或通过 “Find...” 指定路径）。
2. **项目内固定**：本仓库 `.vscode/settings.json` 已配置 `python.defaultInterpreterPath` 指向 `${workspaceFolder}/.venv/Scripts/python.exe` 与 `python.terminal.activateEnvironment: true`，新终端会自动激活该环境。

---

*CPEDC 创新组 — 边水缝洞强非均质气藏 PINN 数字孪生系统 v2.1（一等奖版：两相 PDE、连通性、水侵预警 | 仓库结构整理 2026-02-21）*
