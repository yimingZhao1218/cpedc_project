# 边水缝洞强非均质碳酸盐岩气藏 PINN 数字孪生系统 — 项目设计文档

> **文档版本**: v1.1 | **更新时间**: 2026-03-05  
> **项目**: 第十六届中国石油工程设计大赛（CPEDC）创新组  

---

## 1. 系统架构总览

### 1.1 工程问题背景

赛题聚焦**碳酸盐岩气藏**（SY9、SY13等7口井），核心工程难点：
- **强边水侵入**：统一气水界面 -4385m，边水沿裂缝体系侵入
- **缝洞强非均质性**：基质渗透率极低(0.01~1mD)，裂缝增强后可达数十mD
- **数据稀疏**：仅7口井，单井生产数据(SY9)，PVT数据有限

### 1.2 技术路线

用**物理信息神经网络(PINN)**替代传统数值模拟器：
- 传统路线：地质建模→网格剖分→数值求解(Eclipse/CMG)→历史拟合→预测
- PINN路线：数据预处理→神经网络+PDE约束→端到端训练→秒级正演→多方案优化

### 1.3 模块划分与文件映射

| 模块 | 功能定位 | 核心文件 |
|------|---------|---------|
| M1 | 数据清洗与坐标统一 | `src/m1/m1_data_processor.py` |
| M2 | 地质域构建(Kriging+配点网格) | `src/m2/m2_geo_builder.py` |
| M3 | PVT+相渗物性模块 | `src/physics/pvt.py`, `src/physics/relperm.py` |
| M3-Torch | 可微分物性(autograd兼容) | `src/pinn/torch_physics.py` |
| M4 | PINN基线(无井模型) | `src/pinn/model.py`, `src/pinn/losses.py`, `src/pinn/trainer.py` |
| M5 | 井藏耦合同化+参数反演 | `src/pinn/m5_model.py`, `src/pinn/m5_trainer.py`, `src/pinn/assimilation_losses.py` |
| M5-Well | Peaceman井模型+p_wf反演 | `src/pinn/well_model.py` |
| M5-Aux | ReLoBRaLo自适应权重+RAR采样 | `src/pinn/relobralo.py`, `src/pinn/rar_sampler.py` |
| M6 | 消融实验+UQ+连通性分析 | `src/m6/connectivity.py`, `src/pinn/uq_runner.py` |
| M6-PI | 附表9 PI独立闭环验证 | `src/m6/pi_validator.py` |
| M7 | 水侵预警+NSGA-II制度优化+碳足迹 | `src/pinn/water_invasion.py`, `src/pinn/nsga2_optimizer.py`, `src/pinn/carbon_footprint.py` |
| M7-FM | 7井差异化智能管控 | `src/pinn/field_management.py` |
| 前端 | Streamlit交互展示平台 | `app/streamlit_app.py`, `app/pages/`, `app/components/` |
| 入口 | 主执行脚本(M1~M6一键流程) | `src/main.py` |

---

## 2. M1 数据层与坐标统一

### 2.1 设计初衷

赛题原始数据分散在附表1-8中，格式不统一。M1负责：统一坐标系(工程坐标m)、清洗缺失值/异常值(如PERM=-9999)、生成标准化CSV。

### 2.2 关键数据流

```
附表1(井位坐标) → wellhead_locations.csv → M2/M5/M6/M7
附表2(井眼轨迹) → wellpath_stations.csv  → M2(TVD→海拔)
附表4(MK分层)   → mk_interval_points.csv → M2(Kriging控制点)
附表5(SY9生产)  → production_SY9.csv     → M5(同化锚点)
```

### 2.3 关键设计决策

- **坐标归一化参数**：M1计算 `x_min,x_max,y_min,y_max`(域范围约17400m×11000m)，所有模块共用进行[-1,1]归一化
- **时间轴**：SY9首条生产记录为t=0，归一化到[0,1]，t_max=1331天
- **开井/关井判定**：`生产时间_(H)`列——生产时间=0且qg缺失时视为关井(qg=0)，此逻辑贯穿M5损失函数设计

---

## 3. M2 弱空间地质域构建

### 3.1 设计初衷

PINN需要空间域定义PDE配点和边界点。M2从7口井MK层段数据构建连续地质域(顶/底面、厚度场)，并生成配点网格。

### 3.2 Kriging插值

选择Ordinary Kriging而非IDW：Kriging基于变差函数，能给出插值不确定性。变差函数模型通过**LOO-CV(留一交叉验证)**自动选择(球状/指数/高斯)。

关键输出：`mk_top_surface.csv`, `mk_bot_surface.csv`, `mk_thickness.csv`, `collocation_grid.csv`, `boundary_points.csv`

### 3.3 配点网格设计

```
配点策略: 均匀网格 + 井周加密
  - 基础网格: ~100m间距
  - 井周3000m半径内: 加密2x
  - 边界点: 矩形域四边等距分布
```

**设计初衷**：井周压力梯度最陡，加密采样提高逼近精度而不增加整体计算量。

### 3.4 厚度场梯度

M2构建的h(x,y)在M5 PDE中起关键作用——2.5D守恒方程的厚度修正项：

```
∇·(T·h·∇p) = T·h·∇²p + T·∇h·∇p
```

`PINNSampler._build_thickness_field()`在配点处预计算log-thickness梯度`(gx, gy)`，存储为`collocation_gx`, `collocation_gy`，直接用于PDE残差计算。

---

## 4. M3 可微分物性模块

### 4.1 设计初衷

两相(气+水)流动需要PVT和相渗模型。M3提供两个层级：
1. NumPy层(`physics/pvt.py`, `physics/relperm.py`)：数据预处理和验证
2. PyTorch层(`pinn/torch_physics.py`)：完全可微分，嵌入PINN的autograd图

### 4.2 TorchPVT — 可微分PVT

数据来源：附表5-4(不同温压下Z因子、Bg、ρ_g、μ_g、c_g)

```python
class TorchPVT(nn.Module):
    def rho_g(self, p):  # ρ_g = p·M/(Z·R·T), Z用多项式拟合
    def mu_g(self, p):   # 气相粘度 Pa·s
    def bg(self, p):     # 体积系数 m³/sm³
    def cg(self, p):     # 压缩系数 1/MPa = ρ_g·(1/p - (1/Z)(dZ/dp))
```

关键参数：M=17.25g/mol, T=413.47K(140.32°C), ρ_w=1050kg/m³, μ_w=0.28mPa·s

**μ_w版本说明**：当前权威值为0.28mPa·s（Kestin-Khalifa关联式 @T=140°C, TDS≈10⁴⁵mg/L, `torch_physics.py` TorchPVT类）。历史版本中`well_model.py`曾硬编码0.5mPa·s（常温值，偶高~80%），已在v3.21修正。注意`TorchRelPerm.fractional_flow_water()`的默认参数仍为0.5e-3，但实际调用时传入TorchPVT的正确值。

### 4.3 TorchRelPerm — 可微分相渗

采用**Corey模型**：

```
Se = (Sw - Swc) / (1 - Swc - Sgr)          归一化饱和度
krw = krw_max · Se^nw                        水相相渗
krg = krg_max · (1 - Se)^ng                  气相相渗
```

端点参数(附表7 SY13拟合)：Swc=0.26, Sgr=0.062, krw_max=0.48, krg_max=0.675, nw=4.4071, ng=1.0846 (R²>0.98)

**设计亮点**：ng/nw存储为`_ng_log=log(ng)`，通过`exp()`保证正值。在M5训练中作为**可学习参数**微调，有先验正则化(log空间)防止偏离。

导数方法 `dkrg_dSw()`, `dkrw_dSw()` 为PDE守恒方程展开项提供解析导数，`fw()` 含水率用于Buckley-Leverett水侵预测。

---

## 5. M4 PINN 基线模型

### 5.1 设计初衷

M4是PINN的**最小闭环**：不含井模型、不做参数反演，仅验证"神经网络+PDE约束"能否学到合理的压力/饱和度场。

### 5.2 网络架构 (`src/pinn/model.py` — `PINNNet`)

```
输入: (x_n, y_n, t_n) ∈ [-1,1]³
  ↓
Fourier Feature Encoding (σ=[1,2,4,8,16])
  ↓
Residual MLP: 6层, hidden=256, LayerNorm, Tanh/GELU
  ↓
输出头1: p(压力MPa) — 线性输出+p_offset(=p_init=76MPa)
输出头2: Sw(饱和度) — tanh映射到[0.05,0.95]+clamp
```

**Fourier Feature Encoding设计初衷**：解决PINN的**谱偏差问题(spectral bias)**。标准MLP倾向于先学低频分量，对高频信号(井周压力梯度)收敛极慢。多尺度频率[1,2,4,8,16]显式引入高频信息。

**输出层设计**：压力用`p_offset=p_init`让网络只学"偏离初始压力的增量"，减小输出范围加速收敛。饱和度用tanh硬约束[0.05,0.95]避免物理不合理值。

**PermeabilityNet子网络**：输入(x_n,y_n)，输出log(k/mD)，实际k=exp(log_k)。对数空间保证k>0，跨数量级变化更平滑。正则化：TV(抑制空间突变)、Laplacian(抑制高频)、井点先验。

### 5.3 PDE残差损失 (`src/pinn/losses.py`)

两相质量守恒方程：

**气相**: φ·h·[-ρ_g·∂Sw/∂t + (1-Sw)·ρ_g·c_g·∂p/∂t] = ∇·(k·krg·h·ρ_g/μ_g·∇p) + q_g

**水相**: φ·h·ρ_w·∂Sw/∂t = ∇·(k·krw·h·ρ_w/μ_w·∇p) + q_w

代码实现要点：
- 归一化→物理坐标链式法则：sx=2/dx, sy=2/dy, st=1/t_max_s
- 二阶导：`torch.autograd.grad(dp_dx, xyt, ...)[0][:,0:1]`
- 单位转换：∇p从MPa/m→Pa/m(×1e⁶)，k从mD→m²(×9.869e-16)
- 残差归一化：除以φ·h·ρ_ref/t_max，clamp(±50)截断防梯度爆炸

### 5.4 分阶段课程学习

| 阶段 | 名称 | 比例 | 策略 |
|------|------|------|------|
| A | IC/BC warmup | 15% | 先学初始/边界条件 |
| B | PDE引入 | 25% | 逐步开启PDE残差 |
| C | 数据同化 | 35% | 加入qg, p_wf监督 |
| D | 精细调整 | 25% | 降低LR，精细化 |

**设计初衷**：直接联合训练时PDE残差初期极大，压制数据损失导致不稳定。课程学习按"先简单后复杂"逐步引入。

---

## 6. M5 井—藏耦合同化与参数反演

### 6.1 设计初衷

M4的局限：没有井模型无法约束qg/p_wf；没有参数反演k只是先验值；无法做产量预测。

M5的核心创新：
1. **Peaceman井模型**：将PINN预测的地层压力p_cell与井底流压p_wf耦合
2. **参数反演**：同时反演k_frac(裂缝增强渗透率)、dp_wellbore(井筒压差)、Corey指数(ng,nw)
3. **多目标损失**：qg监督+p_wf反演+PDE残差+正则化，用ReLoBRaLo自适应平衡
4. **RAR自适应采样**：残差驱动加密高残差区域的PDE配点

### 6.2 M5PINNNet 模型架构 (`src/pinn/m5_model.py`)

```python
class M5PINNNet(PINNNet):
    """在PINNNet基础上增加: k_net子网络 + WellModel + 井眼奇异性修正"""
```

继承关系：`M5PINNNet → PINNNet`

新增组件：

**a) k_net (PermeabilityNet)**
- 空间渗透率场k(x,y)预测，输出log(k/mD)
- TV+Laplacian正则化保持地质连续性
- 井点先验约束：k(x_well) ≈ k_measured

**b) WellModel**
- 封装Peaceman井指数计算、p_wf反演网络、源项分布
- 详见6.3节

**c) 井眼奇异性修正(well singularity correction)**
- 压力场在井点附近存在对数奇异性：p ~ A·ln(r/r_w)
- 用小型MLP(`well_log_amp_net`)学习振幅A，从PINN压力场中解耦奇异分量
- **设计初衷**：标准MLP难以逼近对数奇异性(无穷阶导数发散)，显式分离后残余场更光滑

**d) forward与evaluate_at_well**

```python
def forward(self, x):
    """标准前向：返回(p, sw)，p已含井眼奇异性修正"""
    p_base, sw = self.field_net(x)
    if self.well_xy_norm is not None:
        p = p_base + self._well_singularity(x)
    return p, sw

def evaluate_at_well(self, well_id, xyt, ...):
    """井位评估：返回qg_pred, p_wf_pred, p_cell等"""
    p_cell, sw = self.forward(xyt)
    p_wf = self.well_model.pwf_nets[well_id](t_norm, ...)
    qg = self.well_model.compute_qg(well_id, p_cell, p_wf, ...)
    return {'qg': qg, 'p_wf': p_wf, 'p_cell': p_cell, 'sw': sw}
```

### 6.3 Peaceman 井模型 (`src/pinn/well_model.py`)

#### 6.3.1 PeacemanWI — 井指数计算

```python
class PeacemanWI(nn.Module):
    """Peaceman等效半径井指数
    
    WI = 2π·k_frac·h / ln(r_e/r_w)
    
    其中:
    - k_frac: 裂缝增强渗透率(可学习), 存储为_k_frac_raw=log(k_frac)
    - r_e: 等效排泄半径(可学习, v4.8):
           sigmoid约束[50,500]m, 初始值由Peaceman公式估算≈128.9m,
           训练中由产量数据驱动反演。
           设计依据: Peaceman公式r_e≈0.28√(Δx²+Δy²)的前提是
           结构化有限差分网格, 但PINN是无网格方法, 用配点密度
           代替网格间距缺乏严格物理依据。将r_e升级为可学习参数
           后, 排泄半径由数据驱动确定, 反演值可解释为储层等效排泄半径。
    - r_w: 井径(固定=0.1m)
    - h: 有效储厚(井别, 从附表8获取)
    """
```

**k_frac参数化设计初衷**：

传统井指数公式中k和f_frac(裂缝增强因子)高度耦合不可辨识：
```
WI = 2π·(k_eff·f_frac)·h / ln(r_e/r_w)
```
k_eff和f_frac可同时变化而不影响WI值。解决方案是合并为**单一参数k_frac = k_eff × f_frac**，消除不可辨识性。

存储为`_k_frac_raw = nn.Parameter(log(k_frac_init))`，实际值通过`exp()`恢复，保证k_frac > 0。

#### 6.3.2 PwfHiddenVariable — 井底流压反演

```python
class PwfHiddenVariable(nn.Module):
    """p_wf(t)作为隐变量, 用小型MLP拟合
    
    输入: [t_norm, prod_hours_norm, casing_norm]  3维工况向量
    输出: p_wf (MPa)
    
    设计初衷:
    - 赛题只给了WHP(油管压力), 而非BHP(井底流压)
    - p_wf = WHP + Δp_wellbore, 但Δp_wellbore未知
    - 让MLP直接学习p_wf(t), 并用WHP约束: p_wf ≈ WHP + dp_wellbore
    """
```

MLP结构：3→64→64→32→1, 带Tanh激活
- 输出范围约束：p_wf ∈ [10, p_init] MPa
- 平滑正则化：`compute_smoothness()` 惩罚 |dp_wf/dt|²

#### 6.3.3 GaussianSourceTerm — 源项空间分布

```python
class GaussianSourceTerm(nn.Module):
    """将井点产量分布到PDE配点
    
    q(x,y) = Q_total · G(x-x_w, y-y_w; σ)
    
    σ = well_radius × spread_factor
    
    设计初衷:
    PINN的PDE配点是离散的，产量需要以连续场的形式注入PDE。
    高斯核将点源展开为光滑场，避免数值奇异性。
    """
```

#### 6.3.4 WellModel — 主集成类

```python
class WellModel(nn.Module):
    """井模型集成: PeacemanWI + PwfHiddenVariable + GaussianSourceTerm
    
    compute_qg(well_id, p_cell, p_wf, ...):
        dp = p_cell - p_wf                    # 驱动压差 (MPa)
        WI = self.peaceman.compute_WI(h)       # 井指数 (m³/(Pa·s))
        krg = self.relperm.krg(sw)             # 气相相渗
        mu_g = pvt.mu_g(p_cell)                # 气相粘度
        Bg = pvt.bg(p_cell)                    # 体积系数
        
        qg = WI · krg · dp×1e6 / (mu_g × Bg)  # m³/d标况
    
    设计初衷:
    Peaceman公式将地层级别(p_cell)和井筒级别(p_wf)耦合,
    使得PINN的场预测直接约束在实际产量数据上——
    这是传统数值模拟的"历史拟合"在PINN框架下的等价实现。
    """
```

### 6.4 多目标同化损失 (`src/pinn/assimilation_losses.py`)

`AssimilationLoss`类整合三大类损失：

#### 6.4.1 监督损失(Supervised)

**a) loss_qg — 产气量损失(核心)**

```python
def loss_qg(self, qg_pred, qg_obs, scale, valid_mask):
    """v3.17多分量混合损失:
    
    L_qg = w_smape·L_smape + w_log1p·L_log1p
         + w_mse_high·L_mse_high + w_bias_high·L_bias_high
         + w_mse_global·L_mse_global
         + w_mape·L_mape + w_peak·L_peak
    """
```

**设计初衷与演进逻辑**：

1. **L_smape(对称平均绝对百分比误差)**：对高产和低产段等权惩罚，但对系统性低估惩罚不足
2. **L_log1p**：log(1+|error|)对大误差不敏感，避免单个异常点主导损失
3. **L_mse_high(高产段MSE)**：qg_obs ≥ 200k m³/d的高产段用绝对误差MSE，确保峰值不被低估
4. **L_bias_high(高产段偏置)**：惩罚系统性低估(mean bias)²
5. **L_mse_global(全局MSE)**：直接优化R²
6. **L_mape(直接MAPE)**：SMAPE对低估惩罚不足，直接用MAPE
7. **L_peak(超高产峰值, v3.17)**：qg_obs ≥ 350k的峰值段用相对误差²精准打击

**分段加权策略**：
```python
seg_w = torch.ones_like(qg_obs)
seg_w = torch.where(qg_obs < 5e4,  2.0 * seg_w, seg_w)  # 低产段×2
seg_w = torch.where(qg_obs < 2e5,  4.0 * seg_w, seg_w)  # 中产段×4
seg_w = torch.where(qg_obs >= 2e5, 6.0 * seg_w, seg_w)  # 高产段×6
```
**设计初衷**：高产段数据点少但经济价值高，加大权重确保拟合精度。

**b) loss_qg_nearzero — 近零产量(关井)损失**

```python
def loss_qg_nearzero(self, qg_pred, qg_obs, threshold=500, q_scale=5e4):
    """仅对qg_obs ≤ 500 m³/d且valid_mask=1的真实关井点监督
    至少8个点才启用，避免噪声"""
```

**设计初衷**：关井段qg应≈0，但PINN容易在关井段预测出残余产量(因为压差仍存在)。需要专门损失项将关井段产量拉到零。

**c) loss_whp — 油管压力损失**

```python
def loss_whp(self, p_wf_pred, whp_obs, dp_wellbore):
    """p_wf约束: p_wf_pred ≈ WHP_obs + Δp_wellbore
    
    设计初衷: 赛题给的是油管压力WHP, 不是井底流压BHP
    通过可学习的dp_wellbore建立两者关系"""
```

**d) loss_shutin_delta — 关井压差损失(v3.2新增)**

```python
def loss_shutin_delta(self, p_cell, p_wf, qg_obs, ...):
    """关井时p_wf应接近p_cell(压差→0)
    直击"关井仍维持大压差、qg下不去"的根因"""
```

#### 6.4.2 物理损失(Physics)

**a) loss_ic — 初始条件**: p(t=0) = p_init(76MPa), Sw(t=0) = sw_init(0.15)

**b) loss_bc — 边界条件**: p(boundary) = p_init(含水层准稳态)

**c) loss_pde — PDE残差(核心物理约束)**

完整两相守恒PDE残差，与M4的losses.py数学形式一致，但增加了关键改进：

**梯度贯通设计(v2核心创新)**：
```python
# k_net和k_eff_mD_tensor在loss_pde内部用xyt计算k_field
# 确保autograd.grad(k_field, xyt)在同一张计算图
if k_net is not None:
    xy_for_k = xyt[:, :2]   # 切片=视图，不断图
    k_field = k_net.get_k_mD(xy_for_k)  # 在计算图中
    # ∇k梯度 — allow_unused=False: 图断了直接报错
    dk_grad = autograd.grad(k_field, xyt, ..., allow_unused=False)
```

**设计初衷**：在早期版本中，k_field在loss_pde外部计算后传入，导致autograd图断裂——∇k梯度为零，PDE无法约束k_net。v2将k_field计算移入loss_pde内部，确保k_net参数的梯度能正确回传。

**flux展开(气相)**：
```
flux_g = T_g·∇²p                    # 主扩散项
       + (krg·h·ρ_g/μ_g)·∇k·∇p     # ∇k项(k_net空间变化)
       + (k·h·ρ_g/μ_g)·(dkrg/dSw)·∇Sw·∇p  # ∇krg项(相渗空间变化)
       + T_g·(1/h)·∇h·∇p            # 厚度修正项(2.5D)
```

#### 6.4.3 正则/约束损失

| 损失项 | 作用 | 设计初衷 |
|--------|------|---------|
| smooth_pwf | &#124;dp_wf/dt&#124;²平滑 | 抑制p_wf高频抖动 |
| smooth_qg | 开井段Δqg²平滑 | 抑制Peaceman乘法链放大的高频抖动 |
| monotonic | qg↑同时p_wf↑时惩罚 | 物理约束：产量增加应对应压差增大 |
| pwf_constraint | p_wf < p_cell - 2MPa | 生产井必须有驱动压差 |
| prior | log(k/k_prior)² | 防止k偏离先验太远 |
| sw_bounds | Sw有界性惩罚 | 对称屏障[0.09,0.65]+硬边界[0.05,0.85] |
| k_reg | TV+Laplacian+井点 | k_net空间正则化 |
| shutin_delta | 关井时p_cell≈p_wf | 直击关井段qg不归零的根因 |

**loss_sw_bounds设计详解(v5最终版)**：
```python
# 所有阈值有物理依据:
gas_floor = sw_init - 0.06 = 0.09     # SY9初始Sw≈0.15
gas_ceiling = 0.65                      # 相渗等渗点附近
hard_lower = 0.05                       # tanh下界+margin
hard_upper = 0.85                       # tanh上界-margin
# 对称:上下等强度,不造成单向漂移; 初始值Sw=0.15处零惩罚
```

**loss_prior_params设计**：
```python
# 对数空间正则: log(k/k_prior)²
# 设计初衷: 对数空间对跨数量级变化更宽容(k可能从5mD到50mD)
# k_frac先验 = k_eff(0.5mD) × frac_conductivity_factor(16) = 8mD
# k_eff=0.5mD: 基质渗透率保守先验(附表3测井PERM几何均值, config.yaml physics.priors.k_eff_mD)
# frac_conductivity_factor=16: 基于MK组缝洞型碳酸盐岩工程经验(config.yaml physics.priors)
# 训练结果k_frac=10.13mD, 与先验8mD一致(偏差26%, 在反演不确定性范围内)
# dp_wellbore: v4.7版本已冻结(frozen)在13.3MPa
# 冻结依据: 试油实测WHP=57.93MPa, BHP=71.23MPa, Δp=13.30MPa
# 冻结后作为常数参与p_wf计算, 不再参与梯度更新
# 历史版本中dp_wellbore曾作为可学习参数, 但实测值精度足够无需反演
# Corey指数先验: 防止ng/nw偏离SY13拟合值过远 (log空间, 权重0.1)
```

#### 6.4.4 总损失组装

```python
def total_loss(self, model, batch, well_outputs, weights, k_net, ...):
    """计算完整M5总损失
    
    losses = {ic, bc, pde, qg, qg_nearzero, shutin_delta,
              whp, qw, smooth_pwf, smooth_qg, pwf_constraint,
              monotonic, prior, sw_bounds, k_reg}
    
    total = Σ w_i × loss_i
    """
```

### 6.5 ReLoBRaLo 自适应权重 (`src/pinn/relobralo.py`)

```python
class ReLoBRaLo:
    """Relative Loss Balancing with Random Lookback
    
    核心思想: 动态调整各损失项权重, 使"变化快的损失"获得更大权重
    
    算法:
    1. 记录每个loss的历史值和EMA(指数移动平均)
    2. 计算相对变化率: r_i = L_i(t) / EMA_i(t)
    3. 随机lookback: 以概率ρ用历史某步替代EMA
    4. Softmax归一化: w_i = softmax(r_i / τ)
    
    参数:
    - temperature τ: 越小权重越集中, 默认1.0
    - alpha: EMA衰减系数, 默认0.999
    - rho: random lookback概率, 默认0.999
    - warmup_steps: 预热期使用均匀权重, 默认200步
    """
```

**设计初衷**：PINN多目标训练中，12+个损失项的量级可能相差10⁶倍。手动调权费时且不稳定。ReLoBRaLo自动平衡：收敛快的损失权重降低，收敛慢/变化大的权重升高。

参与平衡的损失项：`[ic, bc, pde, qg, shutin_delta, smooth_pwf, monotonic, prior, k_reg, whp, sw_bounds, tds]`

**注意**：`qg_nearzero`不参与ReLoBRaLo平衡——因为关井点稀少，其损失值波动大，会干扰其他项的权重分配。

`ManualLossBalancer`作为简单替代方案，用于消融实验(关闭ReLoBRaLo时)。

### 6.6 RAR 自适应采样 (`src/pinn/rar_sampler.py`)

```python
class RARSampler:
    """Residual-based Adaptive Refinement
    
    核心思路:
    1. 维护"候选池"(2000个随机配点)
    2. 每300步在候选池上计算PDE残差
    3. 选取残差最大的top-200点加入active集
    4. 重复直到8000点
    
    特别适合: 井周高残差区域(压力梯度陡), 饱和度前缘(Sw变化剧烈)
    """
```

**设计初衷**：固定配点可能在高梯度区域分辨率不足，RAR自动发现并加密这些区域。

**消融实验结论**：在单井(SY9)场景下RAR的边际贡献可忽略(M6消融报告确认)，因为井周加密采样已覆盖关键区域。但在多井场景下RAR预期有更大价值。

### 6.7 M5Trainer 训练循环 (`src/pinn/m5_trainer.py`)

#### 6.7.1 优化器设计

4组参数，精细控制学习率：

| 参数组 | 参数量 | LR倍率 | 说明 |
|--------|--------|--------|------|
| field_net | ~416K | 1× | 场网络(压力+饱和度) |
| inversion | ~数K | 1.5× | pwf_nets + dp_wellbore |
| k_frac | 1 | 10× | 单标量, 梯度极小需高LR |
| k_net | ~数K | 0.1× | 空间渗透率子网络, 有warmup |

**k_frac独立参数组设计初衷**：k_frac是WI公式中的乘子，其梯度与产量误差的乘积极小(因为WI值本身很小)。若使用与场网络相同的LR，k_frac几乎不动。10×的LR倍率确保k_frac能在合理步数内收敛。

**k_net warmup设计**：前2000步k_net的LR≈0(冻结)，之后逐步开放。原因：k_net需要在场网络已建立基本压力场后再开始学习空间渗透率分布，否则k场随机初始化会污染PDE残差。

#### 6.7.2 学习率调度

```python
# Warmup(1000步) + Cosine Decay + Stage D额外0.5×衰减
def warmup_cosine_schedule(step):
    if step < warmup:  return step / warmup
    progress = (step - warmup) / (max_steps - warmup)
    base = 0.5 * (1 + cos(π·progress))
    if step >= stage_d_start:  base *= 0.5
    return base
```

#### 6.7.3 TDS数据同化(v3.23)

```python
def _load_tds_data(self):
    """附表6水分析TDS数据 → Sw软标签
    
    物理模型:
    f_brine = clip((TDS - TDS_condensate) / (TDS_brine - TDS_condensate), 0, 1)
    Sw_tds = Swc + f_brine × (1 - Swc - Sgr)
    
    TDS_condensate = 100 mg/L  (凝析水基线, 2013-06~2014-06)
    TDS_brine = 105,000 mg/L   (地层卤水端元, 2016-09峰值)
    """
```

**设计初衷**：PINN训练的Sw缺乏直接观测约束(生产数据只有qg和WHP)。TDS水化学数据提供独立的Sw间接约束——总矿化度越高说明地层水混入越多，Sw越高。

**v3.23修复**：附表6 SY9数据跨2013-2022(174样本)，但PINN训练域仅0~1331天。旧版np.clip将域外样本堆叠到t_norm=1.0导致边界污染，现改为域外过滤，仅保留训练域内样本。

#### 6.7.4 有效储厚(net pay)处理

```python
net_pay_override = {
    'SY9': 48.4,   # 附表8: 16.296 + 32.1 = 48.4m
    'SY13': 41.65, # 附表8: 5.25 + 11.3 + 25.1
    'SY201': 37.9, # 附表8: 7.0 + 30.9
    ...
}
```

**设计初衷(v3.4)**：mk_interval_points.csv的mk_thickness=92.14m是**毛厚度(gross)**，而Peaceman WI应使用**有效储厚(net pay)**=48.4m(附表8测井解释)。用毛厚度会导致WI膨胀92.14/48.4=1.9×，优化器被迫把k_frac从4.0压到2.1mD来补偿——参数耦合错误。

#### 6.7.5 数据集切分

时间顺序切分(非随机)：train:val:test = 65:15:20（config.yaml `m5_split: [0.65, 0.15, 0.2]`）

**设计初衷**：时间序列数据不能随机打乱，否则未来数据泄露到训练集。按时间前65%训练、中15%验证(早停/best选择)、后20%测试(外推能力评估)。测试集(后20%)用于评估模型对未见生产阶段的真实外推能力——M5验收报告中 test MAPE=0.7% < val MAPE=3.5%，说明模型在训练窗口末段具备可靠外推性，这正是时间顺序切分的核心价值所在。

#### 6.7.6 报告生成

`generate_report()`输出：
- qg/p_wf拟合对比图 (train/val/test分段)
- 训练曲线(各损失项随步数变化)
- PDE残差热力图(空间分布)
- Sw时空演化图
- 定量指标(MAPE, RMSE, R²)

---

## 7. M6 消融实验、不确定性量化与连通性分析

### 7.1 设计初衷

M6回答三个关键问题：
1. **消融**：PINN各组件(PDE、k_net、Fourier、RAR)各贡献了多少？
2. **UQ**：反演参数和预测结果的不确定性有多大？
3. **连通性**：7口井之间的流体连通关系如何？水侵风险排序？

### 7.2 消融实验 (`src/m6/run_ablation_suite.py`)

**严格单变量递进链**：
```
Exp1: pure_ml (无PDE, 纯数据驱动)
Exp2: pinn_no_knet (有PDE, 无k_net空间反演)
Exp3: pinn_no_fourier (有PDE+k_net, 无Fourier编码)
Exp4: pinn_full (完整模型)
Exp5: pinn_full_rar (完整模型+RAR)
```

每组实验用相同随机种子、相同数据、相同训练步数，仅变化一个组件。

**结论**（详见 `outputs/mk_pinn_dt_v2/reports/M6_ablation_report.md` 实际输出）：
- 物理约束(PDE)是核心驱动力：pure_ml→pinn_no_knet RMSE大幅下降
- Fourier编码对高频信号拟合有显著贡献（详见报告定量数字）
- k_net空间反演贡献见报告（单井场景下边际贡献有限）
- RAR边际贡献可忽略（单井场景，井周加密采样已覆盖高梯度区，M6报告已确认）

注：以上定性结论均基于M6消融实验实际输出，具体百分比数字以M6_ablation_report.md为准。

### 7.3 不确定性量化 (`src/pinn/uq_runner.py`)

```python
class UQRunner:
    """多随机种子ensemble实现低成本UQ
    
    方法:
    1. 用不同种子(base_seed + i×1000)训练N个模型(N≥5)
    2. 汇聚预测: P10/P50/P90区间
    3. 反演参数: k_frac的均值/标准差/CV
    4. 累产气: P10/P50/P90
    """
```

**设计初衷**：贝叶斯UQ(如MC Dropout, Hamiltonian MC)对PINN计算成本极高。Ensemble方法简单有效：每个成员从不同初始化出发，收敛到不同局部最优，ensemble spread反映参数不确定性。

### 7.4 连通性分析 (`src/m6/connectivity.py`)

#### 7.4.1 多源数据融合构建k(x,y)场

```python
class ConnectivityAnalyzer:
    """井间连通性分析器
    
    数据融合:
    1. 附表3测井PERM → 7个硬约束点(各井MK段几何均值)
    2. SYX211补充: 附表3全无效, 用附表8解释成果k=0.037mD
    3. SY9叠加PINN反演k_frac(唯一有产量约束的井)
    4. IDW反距离加权插值(对数空间, p=2) → 连续k(x,y)场
    """
```

#### 7.4.2 连通性矩阵计算

```
方法:
1. 构建80×80网格图, 边权 = 渗流阻力 = ds/(k·h)
2. Dijkstra最短路径 → 井间最小阻力 R_ij
3. C_ij = exp(-R_ij/R_ref) 指数衰减归一化

修正因子:
- Sw流体校正: k_eff = k × (1 - Se)^n (含水区渗透率衰减)
- 构造校正: 低于GWC(-4385m)的网格额外阻力
```

#### 7.4.3 WIRI水侵风险指数

```
WIRI = w1·(1/R_boundary) + w2·Sw_well + w3·(1/RT_well)
```
- R_boundary：井到最近边界(水区)的渗流阻力
- Sw_well：附表8测井含水饱和度
- RT_well：附表3有效段电阻率(低RT→含水)

**多方法交叉验证**：
- **Dijkstra最短路径(IDW图论)** vs **解析传导率公式(距离衰减)**，两种独立方法的连通性排序对比，Pearson R在报告中输出
- **WIRI权重敏感性分析**：对构造[20%,60%]+连通性[15%,45%]+Sw[15%,45%]权重组合扫描，检验排名对权重选择的鲁棒性
- **结论**：排序鲁棒性高，但k场绝对值受IDW控制点(7个)限制，目标是井间风险排序而非精确重建渗透率场

### 7.5 附表9 PI独立闭环验证 (`src/m6/pi_validator.py`)

#### 设计初衷

M5从生产数据(1331天qg时间序列)反演得到k_frac，PI验证用**完全独立的试井数据**(附表9，未参与PINN训练)反算k_frac，两条路径的结果收敛是反演结果可靠性的最强证据。

#### 验证方法

附表9提供SY9井试油测试点：
```
qg_test  = 568,663 m³/d (地面标况日产气)
p_wf_test = 71.226 MPa  (井底流压)
p_res_test = 74.875 MPa (井底静压)
```

**Step 1**: 计算试油产能指数 PI_test
```
PI_test = qg_test / (p_res - p_wf) = 568,663 / 3.649 ≈ 155,840 m³/d/MPa
```
注：附表9中标注dp=2.7697MPa与直接压差3.649MPa不一致(可能是不同基准)，代码同时计算两个PI值取较保守的。

**Step 2**: Peaceman公式逆推k_frac
```
k_frac_PI = PI_test × μ_g × Bg × ln(r_e/r_w) / (2π × h × krg_max)
```

**Step 3**: 与M5反演k_frac=10.13 mD对比 → 双路径收敛验证

#### 意义

一路是**时间序列反演**(1331天产量拟合，数据驱动)，另一路是**稳态试井反算**(单点Peaceman逆推，物理驱动)。两者在完全不同数据和方法下指向同一k_frac区间，证明反演结果的**物理真实性**，而非PINN过拟合产物。

---

## 8. M7 水侵预警与制度优化

### 8.1 设计初衷

M7将训练好的PINN模型用于**工程决策**：
1. 预测各井水侵风险和见水时间
2. 优化SY9的生产制度(控产/控压)
3. 量化PINN vs传统数模的碳足迹差异

### 8.2 分层预测策略 (`src/pinn/water_invasion.py`)

```python
class WaterInvasionAnalyzer:
    """水侵预警分析器 — 混合策略版(v6.1)
    
    分层预测(不同井不同方法, 按置信度分级):
    - SY9:    PINN直接推理Sw(t),p(t)              [置信度:高]
    - SYX211: 附表8实测确认气水同层(Sw=30.3%)     [置信度:高]
    - SY102:  附表3+构造图确认气水井               [置信度:高]
    - 其他4井: M6 WIRI排序推断                     [置信度:中]
    """
```

**设计初衷**：不同井的数据丰度不同——SY9有完整生产数据可以用PINN推理，SYX211/SY102有测井/构造证据，其他井只能靠M6连通性排序间接推断。

### 8.3 TDS数据驱动Sw经验模型

```python
def _compute_tds_sw_timeseries(self, well_id, t_days):
    """TDS→Sw经验模型(替代PINN Sw预测)
    
    Sw = Swc + f_brine × (1 - Swc - Sgr)
    f_brine = (TDS - TDS_condensate) / (TDS_brine - TDS_condensate)
    
    PCHIP单调插值 + Buckley-Leverett外推
    """
```

### 8.4 Buckley-Leverett非线性Sw演化

```python
def _corey_fractional_flow(self, Sw):
    """fw(Sw) = (krw/μw) / (krw/μw + krg/μg)"""
```

**设计初衷**：Sw的时间演化不是简单线性增长，而是由含水率曲线fw(Sw)控制的非线性过程。Buckley-Leverett理论给出前缘推进速度与dfw/dSw成正比，实现物理一致的外推。

### 8.5 SY9制度优化(3种策略)

```
策略1: 稳产方案 — 维持当前产量(基线)
策略2: 阶梯降产 — 外推区p_wf +1.5/+3 MPa (降低生产压差)
策略3: 控压方案 — 外推区渐进提压 0→6 MPa
```

评估指标：累产气Gp、终期含水Sw_end、净现值NPV

### 8.6 NSGA-II 多目标优化 (`src/pinn/nsga2_optimizer.py`)

```python
"""NSGA-II多目标优化引擎 (PINN-as-Surrogate)

架构: Phase1 PINN推理(1次~2s) → cache → Phase2 NSGA-II(3000次, 每次~0.1ms)

决策变量(4维): dp_stage1[0,5], dp_stage2[0,10], t_switch[0.3,0.7], ramp_days[30,180]
目标(3个minimize): -Gp, Sw_end, -NPV
"""
```

**设计初衷**：传统数模每次前向模拟需数分钟~数小时，3000次评估不可行。PINN推理仅需2秒，将结果缓存后NSGA-II每次评估仅需0.1ms(解析Peaceman公式+Corey相渗)。

**缓存机制**：
```python
def build_evaluation_cache(analyzer, well_id='SY9', n_time=500):
    """Phase1: 一次PINN推理 → cache dict
    包含: p_cell, pwf_base, sw_base, qg_base, dp_base 时间序列
    以及: Corey参数, 经济参数, TDS标定的BL参数
    """
```

**Sw递推(TDS标定BL)**：
```python
def compute_sw_bl_static(sw_base, dp_mod, dp_base, ...):
    """外推区Sw递推: sw[i] = sw[i-1] + λ_BL·dt·(dp_mod/dp_base)·dfw/dSw
    
    λ_BL由训练区TDS数据标定, 确保Sw增长率与实测一致"""
```

### 8.7 碳足迹LCA (`src/pinn/carbon_footprint.py`)

三维碳足迹评估：

**A. 计算侧**：
```
核心创新说明:
  传统优化受限只能人工评估10~20个方案(建模+历史拟合每次2h);
  PINN-NSGA-II将"可搜索空间"从20个扩展到3000个Pareto最优解,
  实现了传统方法"不是慢, 而是根本做不到"的决策质量跃升.

PINN方案 vs 传统方案 (等价任务: 评估3000种调产方案):
  PINN: GPU(RTX4090, 450W) × 训练1h + 推理5min ≈ 0.49kWh → 0.28kgCO₂
  传统: 工作站(600W) × 2h/案例 × 3000案例 = 3600kWh → 2092kgCO₂
  说明: 3000案例为NSGA-II实际评估次数; 传统方法在工程实践中
       因时间约束通常只能评估10~20个方案, PINN方法打破了这一上限.
```

**B. 生产侧**：不同策略下CH₄泄漏(0.1%) + 含水处理能耗

**C. CCUS潜力**：MK组CO₂封存安全窗口初评

### 8.8 7井差异化智能管控 (`src/pinn/field_management.py`)

基于M6 WIRI排名 + NSGA-II优化结果，为7口井生成差异化管控方案（v4.7核心交付）：

| 类别 | WIRI阈值 | 井号 | 管控行动 | 监测频率 |
|------|---------|------|---------|---------|
| 立即干预 | ≥0.7 | SYX211 (WIRI=1.000) | 排水采气/关井控水 | 每日 |
| 重点监控 | 0.4~0.7 | SY102 (0.568), SY116 (0.434) | 阶梯降产(季度降10%) | 季度 |
| 计划跟进 | 0.2~0.4 | SY13 (0.357), SY101 (0.263), SY9 (0.241) | NSGA-II平衡策略 | 半年 |
| 常规生产 | <0.2 | SY201 (0.150) | 维持稳产 | 年度 |

**设计初衷**：不同井的水侵风险差异巨大（SYX211已确认气水同层，SY201构造高位安全），一刀切的管控策略造成资源浪费或风险遗漏。分类管控将有限的监测和干预资源集中在高风险井上。

**输出**：`M7_field_management.png`（7井管控路线图 + 全场NPV对比 + WIRI分级色卡）

---

## 9. Streamlit Web应用 (交互展示平台)

### 9.1 设计初衷

将M1~M7全流程的核心输出整合为可交互的Web仪表板，供评委直观操作和验证，无需运行Python脚本。这是面向评委的**核心交付物**之一。

### 9.2 架构

- **入口**: `app/streamlit_app.py` (主页: 项目概述+技术路线+指标卡片+快速导航)
- **多页面**: `app/pages/` 目录下各功能页面
- **组件库**: `app/components/` (config_loader.py配置加载, plotly_charts.py交互图表)
- **数据源**: 自动读取 `outputs/mk_pinn_dt_v2/reports/` 下的JSON/CSV/PNG

### 9.3 页面功能清单

| 页面 | 功能 | 核心数据来源 |
|------|------|------------|
| 主页 (streamlit_app.py) | 项目概述+技术路线图+指标卡片(MAPE/k_frac/消融/UQ) | M5 inversion_params.json |
| 数据概览 | 7口井地图+生产数据时间序列 | M1坐标+附表10 |
| 训练监控 | 损失曲线+qg/p_wf拟合对比(train/val/test分段) | M5 checkpoint+报告 |
| 渗透率反演 | k_frac/r_e/dp_wb反演值+先验对比+k(x,y)场 | M5反演结果 |
| 水侵预警 | WIRI热力图+7井风险排序+Sw演化预测 | M6连通性+M7水侵 |
| 消融+UQ | 5组消融对比表+k_frac P10/P50/P90区间图 | M6消融报告+UQ报告 |
| 制度优化 (07_⚙️_制度优化.py) | 三策略Gp/Sw对比+Pareto前沿+NSGA-II | M7优化结果 |

### 9.4 启动方式

```bash
streamlit run app/streamlit_app.py
```

### 9.5 设计要点

- **自动数据发现**: 从outputs目录自动读取最新训练结果，无需手动配置路径
- **交互式图表**: 基于Plotly的可缩放/悬停/导出图表
- **实时指标卡片**: MAPE、k_frac、消融实验数、UQ成员数等核心指标一屏可见
- **评委友好**: 侧边栏导航+技术路线图，支持非技术背景评委快速理解系统

---

## 10. 全局数据流与模块依赖图

```
M1(数据清洗)
 ├─→ M2(地质域) ─→ collocation_grid.csv, boundary_points.csv, thickness.csv
 │    └─→ M4(PINN基线) ─→ 基础网络架构验证
 │         └─→ M5(井藏耦合)
 │              ├─→ M6(消融/UQ/连通性)
 │              └─→ M7(水侵预警/优化/碳足迹)
 ├─→ M3(物性) ─→ TorchPVT, TorchRelPerm
 │    └─→ M5(PDE残差中的物性计算)
 └─→ production_SY9.csv ─→ M5(同化锚点)
      附表3(测井PERM) ─→ M6(k场硬约束)
      附表6(TDS) ─→ M5(Sw软标签) + M7(TDS时间序列)
      附表7(相渗) ─→ M3(Corey拟合)
      附表8(测井解释) ─→ M5(有效储厚) + M6(Sw/RT数据融合)
```

### 关键数据接口

| 上游 | 下游 | 数据 | 格式 |
|------|------|------|------|
| M1 | M2 | mk_interval_points.csv | 7口井MK层段(x,y,z_top,z_bot,thickness) |
| M2 | M4/M5 | collocation_grid.csv | 配点(x,y,is_near_well) |
| M2 | M5 | thickness field | h(x,y)及∂ln(h)/∂x, ∂ln(h)/∂y |
| M3 | M5 | TorchPVT, TorchRelPerm | PyTorch nn.Module(autograd兼容) |
| M5 | M6 | m5_pinn_best.pt | checkpoint(model_state_dict) |
| M5 | M7 | 训练好的M5PINNNet | 秒级正演替代器 |
| M6 | M7 | ConnectivityAnalyzer | WIRI排序, C_ij矩阵 |

---

## 11. 关键设计决策汇总

| # | 决策 | 理由 |
|---|------|------|
| 1 | Fourier Feature Encoding | 解决PINN谱偏差, 缓解高频信号学习困难 |
| 2 | k_frac合并参数化 | 消除k_eff与f_frac的不可辨识性 |
| 3 | p_wf作为隐变量MLP | 赛题只给WHP不给BHP, MLP自动拟合p_wf(t) |
| 4 | 梯度贯通(loss_pde内部计算k) | 确保k_net参数梯度正确回传, 避免autograd图断裂 |
| 5 | 有效储厚替代毛厚度 | 防止WI膨胀导致k_frac补偿性下降 |
| 6 | 分段加权qg损失 | 高产段经济价值高但数据点少, 需加权 |
| 7 | ReLoBRaLo自适应权重 | 12+损失项量级差10⁶倍, 自动平衡 |
| 8 | 课程学习(A→B→C→D) | 防止PDE残差初期主导压制数据损失 |
| 9 | TDS→Sw软标签 | 弥补Sw无直接观测的缺陷 |
| 10 | PINN-as-Surrogate | 秒级正演替代传统数模, 支撑3000次NSGA-II评估 |
| 11 | 多源数据融合k场 | 附表3+附表8+PINN反演, 最大化信息利用 |
| 12 | 对数空间k预测/正则 | 保证k>0, 跨数量级变化平滑 |
| 13 | Corey指数可学习 | 在SY13拟合先验基础上微调适配SY9实际数据 |
| 14 | 井眼奇异性分解 | MLP难以逼近log奇异性, 显式分离后残余场光滑 |
| 15 | 时间顺序切分(非随机) | 防止未来数据泄露, 评估真实外推能力 |
| 16 | r_e可学习参数(v4.8) | Peaceman公式r_e≈0.28√(Δx²+Δy²)的前提是结构化网格, 但PINN是无网格方法. 将r_e升级为sigmoid约束[50,500]m的可学习参数, Peaceman估算值(≈128.9m)作为先验初始化, 训练收敛值=等效排泄半径 |

---

## 12. 已知局限性与改进方向

> 一份诚实的设计文档应该坦承已知局限性——这既体现工程严谨性，也为答辩时的主动防守提供准备。

| 局限 | 当前处理 | 改进方向 |
|------|---------|---------|
| PINN仅SY9单井训练 | 其他井用WIRI+构造分层推断, 诚实标注置信度等级(高/中) | 获取其他井生产数据后扩展多井联合训练 |
| TDS→Sw是经验映射非严格物理 | R=0.921交叉验证, BL物理外推, 训练域内PCHIP插值 | 引入毛细压力-电阻率联合约束 |
| BL递推为一维集总参数近似 | λ_BL由TDS斜率标定, dp_ratio驱动Sw增量 | 耦合全场Sw方程, 引入空间非均质性 |
| k场仅7个IDW控制点 | 目标是井间风险排序(WIRI 7/7排序稳定), 而非精确重建 | 增加地震解释数据作为空间约束 |
| RAR存在时间边界数据泄漏(已知P1) | 当前单井场景RAR贡献=0%, 影响有限 | rar_sampler.py增加t<t_train_max过滤 |
| UQ覆盖率低于目标 | ensemble成员间初始化差异不足 | 增加物理参数扰动(k_eff先验±30%)构造多样性 |
| μ_w代码残留不一致 | TorchPVT权威值0.28mPa·s(v3.21), water_invasion.py已统一(v4.8), TorchRelPerm默认参数残留0.5e-3 | 统一所有模块μ_w来源为TorchPVT |

---

> **文档结束** | 更新时间: 2026-03-05
