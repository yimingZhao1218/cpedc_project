# M5 调参历程与问题总结 (v3.6)
> 更新时间: 2026-02-21 15:00  
> 目标: MAPE < 15%, R² > 0.85

---

## 一、各版本结果对比

| 版本 | 步数 | MAPE | R² | k_frac | 主要变更 |
|------|------|------|----|--------|---------|
| v3.3 (基线) | 10000 | 21.3% | 0.593 | 2.31 mD | 旧配置 |
| v3.4 (5k步) | 5000 | 28.5% | 0.478 | 3.08 mD | h_well修复+prior=1.0+PDE权重过高 |
| **v3.5 (10k步)** | 10000 | **17.2%** | **0.652** | 3.27 mD | PDE权重降10x+prior=0.1 |
| **目标** | ≥10000 | **<15%** | **>0.85** | ~4-5 mD | — |

---

## 二、已完成的修复（v3.4 + v3.5）

### 修复1：h_well 毛厚度→有效储厚 ✅
- **文件**: `src/pinn/m5_trainer.py` (line ~399)
- **问题**: `mk_interval_points.csv` 的 `mk_thickness=92.14m` 是毛厚度，Peaceman WI ∝ k×h，用毛厚度导致 WI 膨胀 1.9x，优化器被迫把 k_frac 从 4.0 压到 2.1 mD
- **修复**: 添加 `net_pay_override` 字典，基于附表8测井解释成果表
  ```python
  net_pay_override = {
      'SY9': 48.4,    # 附表8: 16.296 + 32.1 = 48.4 m
      'SY13': 41.65,  # 附表8: 5.25 + 11.3 + 25.1 = 41.65 m
      ...
  }
  ```

### 修复2：prior_k_weight 0.005→0.1 ✅
- **文件**: `src/pinn/assimilation_losses.py` (line ~137)
- **问题**: 旧值 0.005 导致 prior 比 qg 弱 150 万倍（完全无效）；v3.4 改为 1.0 矫枉过正，把 k_frac 锚死在 3.27 mD
- **当前值**: 0.1（引导但不压制寻优）

### 修复3：config PDE 权重降 10x ✅
- **文件**: `config.yaml` (line ~440)
- **问题**: 旧值 pde=0.001~0.003 导致 PDE:Qg=19:1，Qg 4500步零进展
- **修复**: 降到 0.0001~0.0005，使 PDE 贡献与 Qg 同量级

### 修复4：M4 Sw 平线修复 ✅
- **文件**: `src/pinn/model.py` (line ~257)
- **问题**: sw_center=0.50, sw_scale=0.38，tanh 上界=0.88，Sw 漂移到 0.88 平线
- **修复**: sw_center→0.33, sw_scale→0.13，范围缩窄到 (0.20, 0.46)
- **文件**: `src/pinn/losses.py` — anchor 权重 0.3→5.0
- **文件**: `config.yaml` — Stage A sw_phys 0→2.0

### 修复5：k_frac_bounds 缩窄 ✅
- **文件**: `src/pinn/well_model.py`
- **修复**: [0.1, 500]→[0.1, 100]，ratio clamp 0.01→0.001

---

## 三、当前残余问题（v3.5 仍未达标）

### 问题A：k_frac 仍然锁死（最关键）
```
Step 8000: k_frac=3.2630 mD, grad=-1.84
Step 9000: k_frac=3.2667 mD, grad=+0.94
Step 9999: k_frac=3.2668 mD  ← 完全锁死
```
- **根因**: `prior_k_weight=0.1` 仍然过强。prior 先验中心 = k_eff_prior × f_frac_prior = 5.0 × 10.0 = **50 mD**（！），而 k_frac 实际在 3~4 mD 附近，log(3.27/50)² = 5.56，prior 贡献 = 0.05 × 0.1 × 5.56 = **0.028**，这个梯度方向是把 k_frac 往 50 mD 推，与 qg 梯度方向相反，导致拮抗锁死。
- **需要检查**: `k_eff_prior` 和 `f_frac_prior` 的实际取值是否合理

### 问题B：Qg 权重 Stage D=800 仍不够主导
```
Stage D: PDE贡献 = 0.0005 × 239k = 120
         Qg贡献  = 800 × 0.130 = 104
         比值 ≈ 1.15:1 (接近均衡，但 Qg 仍未主导)
```
- Qg 从 step 2000 到 step 9999 几乎没有下降（0.141→0.129），说明优化器还是在 PDE 和 Qg 之间摇摆

### 问题C：dp_wellbore 持续漂移（次要）
```
Step 0:    dp=13.30 MPa
Step 5000: dp=16.21 MPa  (+2.91 MPa)
Step 9999: dp=16.97 MPa  (+3.67 MPa, 仍在漂移)
```
- dp_wellbore 从 13.3 漂到 17.0 MPa，增加了 28%，说明模型在用 dp 补偿 k_frac 不足导致的 WI 偏低
- 这是 k_frac 锁死的间接证据

### 问题D：val R²=0.24 远低于 train R²=0.61（过拟合）
- train MAPE=20.0% vs val MAPE=12.2%（val 反而更好？）
- val R²=0.24 极低，说明 val 段预测方差很小（预测值接近常数），并非真正拟合好
- best_step=3000（10000步中最优在 step 3000），说明后 7000 步在退化

---

## 四、下一轮修复方案（v3.6）

### 方案A：彻底解决 prior 先验中心问题（最高优先级）

**需要检查** `config.yaml` 中 `k_eff_prior` 和 `f_frac_prior` 的值：
```yaml
physics:
  priors:
    k_eff_mD:
      value: ???  # 如果是 5.0，则 k_frac_prior = 5.0 × 10.0 = 50 mD，完全错误！
    frac_conductivity_factor:
      value: ???  # 应为 ~1.0，而非 10.0
```
- 正确的 k_frac_prior 应该 ≈ 4.0 mD（历史最优收敛值）
- 建议直接在 `assimilation_losses.py` 中把 `k_frac_prior` 硬编码为 4.0 mD，或者把 `prior_k_weight` 设为 0.0 彻底禁用

### 方案B：提升 Stage D Qg 权重到 1500
```python
# m5_trainer.py 中 w_d
w_d = {..., 'qg': 1500.0, ...}  # 800→1500
```

### 方案C：增加训练步数到 20000
- best_step=3000 说明模型在 3000 步后开始退化
- 需要更长的 warmup 或更小的 lr 来让后期继续收敛

### 方案D：考虑禁用 k_net（实验性）
- k_net 引入了 15297 个额外参数，可能与 k_frac 标量产生干扰
- 可以尝试 `--no-knet` 对比实验

---

## 五、立即可执行的命令

### 检查 prior 先验中心（先做这个）
```powershell
python -c "
import yaml
cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
priors = cfg.get('physics', {}).get('priors', {})
print('k_eff_prior:', priors.get('k_eff_mD'))
print('f_frac_prior:', priors.get('frac_conductivity_factor'))
m5 = cfg.get('m5_config', {})
print('inversion_prior:', m5.get('inversion_prior'))
"
```

### 下一轮训练（修复 prior 后）
```powershell
python src/m5/run_m5_single_well.py --well SY9 --steps 20000 --device cuda
```

---

## 六、关键数值参考

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `prior_k_weight` | 0.1 | 保持 | 先验中心=4.0 mD (正确, 非50mD) |
| Stage D `qg` 权重 | 800 | **1200** | 让 Qg 真正主导 |
| `max_steps` | 10000 | **20000** | best_step=3000，需更多步数 |
| `warmup_steps` | 1000 | **2000** | 配合 20k 步延长 warmup |
| PDE 权重 (D) | 0.0005 | 保持 | 已合理 |
| `h_well[SY9]` | 48.4 m ✅ | 保持 | 已修复 |

---

## 七、v3.6 深度分析 (2026-02-21 15:00)

### v3.5 结果回顾 (10000步)
```
MAPE = 17.2% (目标 <15%)
R²   = 0.652 (目标 >0.85)
k_frac = 3.2668 mD (step 8000+ 完全锁死)
dp_wellbore = 16.97 MPa (从 13.3 漂移 +28%)
best_step = 3000 (后 7000 步退化)
高产段: mean_error = -101,019 m³/d (系统性低估)
```

### 根因诊断: dp_wellbore 漂移是 R² 低的直接原因

**关键链路**: WHP loss 定义了 `p_wf_obs = WHP + dp_wellbore`，Peaceman 产量 `qg ∝ (p_cell - p_wf)`

dp_wellbore 从 13.3 漂到 17.0 MPa 时，驱动压差坍缩：
```
dp=13.3: 驱动压差 = p_cell - (WHP+13.3) = 76 - 58 - 13.3 = 4.7 MPa
dp=17.0: 驱动压差 = p_cell - (WHP+17.0) = 76 - 58 - 17.0 = 1.0 MPa
```
**驱动压差下降 4.7x → 高产段系统性低估 -101k m³/d**

为什么漂移？
- `dp_wellbore bounds = [2.0, 35.0]` 太宽
- `prior_dp_weight = 0.01` 完全无效
- WHP 权重 (10-22) 太高，优化器用 dp_wellbore 吸收 p_wf 误差

### 根因诊断: Qg 损失函数不匹配 R²

R² 基于 MSE，但 loss_qg 以 SMAPE² + log1p² 为主 (各占 w=0.5)。
MSE_high 仅 w=0.15，bias_high 仅 w=0.10。
优化 SMAPE 不等于优化 MSE → MAPE 可以不错但 R² 很低。

## 八、v3.6 已实施修复

### 修复1: dp_wellbore 强约束 (最关键)
- **`m5_model.py`**: bounds `[2.0, 35.0]` → `[10.0, 17.0]`
- **`assimilation_losses.py`**: 
  - `prior_dp_center` 12.0 → **13.3** (试油实测值)
  - `prior_dp_weight` 0.01 → **0.5** (强约束近实测值)

### 修复2: Qg 损失增强 MSE 分量
- **`assimilation_losses.py`**:
  - `w_mse_high` 0.15 → **0.40** (MSE 是 R² 的直接优化目标)
  - `w_bias_high` 0.10 → **0.30** (消除高产段系统性低估)
  - 新增 `w_mse_global = 0.20` (全局 MSE，不限高产段)

### 修复3: WHP 权重降低
- **`m5_trainer.py`**: 各阶段 WHP 权重
  - A: 10→5, B: 15/20→8/10, C: 18/20→10/12, D: 22→12
  - 降低 WHP 权重让优化器不再把 dp_wellbore 当逃逸通道

### 修复4: Stage D qg 权重提升
- **`m5_trainer.py`**: D 阶段 qg 800 → **1200**

### 修复5: 训练配置
- **`config.yaml`**:
  - `max_steps` 12000 → **20000**
  - `warmup_steps` 1000 → **2000**
  - `inversion_lr_factor` 2.0 → **1.5** (配合 dp 约束)

## 九、v3.6 预期效果

dp_wellbore 约束后:
- dp ≈ 13.3~14.5 (不再漂移到17)
- 驱动压差恢复到 3.5~4.7 MPa (之前仅 1.0 MPa)
- 高产段低估从 -101k 降到 -30k 以内
- k_frac 有更大上升空间 (不再需要被压低来补偿dp漂移)

MSE 分量增强后:
- R² 直接被优化 (之前仅优化 SMAPE)
- 全局 MSE 确保整体拟合精度

预期指标: **MAPE ≤ 12%, R² ≥ 0.80**

---

## 十一、v3.7 深度分析 (2026-02-21 17:00)

### v3.6 结果回顾 (10000步, 注意: --steps覆盖了config的20000)
```
MAPE = 16.9% (v3.5: 17.2%, 改善极微)
R²   = 0.699 (v3.5: 0.652, 改善极微)
k_frac = 3.3846 mD
dp_wellbore = 16.24 MPa (仍远离实测 13.3!)
best_step = 3600
高产段: mean_error = -93,863 m³/d (仍严重低估)
```

### 根因: prior约束方案彻底失败

dp_wellbore 单调漂移轨迹 (无任何回头趋势):
```
Step 0→1000: 13.30→13.50    (+0.20)
Step 1000→3000: 13.50→14.95 (+1.45)
Step 3000→5000: 14.95→15.66 (+0.71)
Step 5000→8000: 15.66→16.22 (+0.56)
Step 8000→10000: 16.22→16.24 (锁死)
```

**prior dp 贡献量化**:
- loss_prior_dp = stage_w(0.05) × dp_w(0.5) × ((16.24-13.3)/13.3)² = **0.001**
- loss_qg = stage_w(1200) × qg(0.14) = **168**
- **比值: 168,000:1 — prior 对优化器完全不可见!**

根本原因: WHP loss 让 dp 成为逃逸通道。dp↑ → p_wf_target↑ → WHP loss↓ (容易满足)，但 驱动压差↓ → qg↓。优化器选择牺牲 qg 来换取 WHP loss 下降。

### 冻结dp的物理论证
```
dp=16.24 (当前): 驱动压差 = p_cell - (WHP+dp) = 75 - 74.24 = 0.76 MPa
dp=13.30 (实测): 驱动压差 = p_cell - (WHP+dp) = 75 - 71.30 = 3.70 MPa
→ 提升 4.9x
```
试油实测: WHP=57.93, BHP=71.23, Δp=13.3 MPa。这是直接物理测量值，无需"学习"。

## 十二、v3.7 已实施修复

### 修复1: 冻结 dp_wellbore = 13.3 MPa (最关键)
- **`m5_model.py`**: `nn.Parameter(torch.tensor(13.3))` → `register_buffer('_dp_wellbore_raw', torch.tensor(13.3))`
- **`m5_model.py`**: dp_wellbore 属性直接返回 buffer (无 clamp, 无梯度)
- **`m5_model.py`**: count_parameters dp_wellbore=0 (不再可训练)
- **`m5_trainer.py`**: `isinstance(model._dp_wellbore_raw, nn.Parameter)` 检查, dp 不进入 optimizer

### TDD 验证 (6项测试全部通过)
```
[PASS] dp_wellbore = 13.30 MPa
[PASS] dp_wellbore NOT in model.parameters()
[PASS] dp_wellbore IS in model.buffers()
[PASS] dp_wellbore.requires_grad = False
[PASS] _dp_wellbore_raw is NOT an nn.Parameter
[PASS] dp_wellbore NOT in optimizer param groups
[PASS] dp_wellbore unchanged after backward
```

## 十三、v3.7 预期效果

dp 冻结后:
- 驱动压差恢复 4.9x (0.76→3.70 MPa)
- 高产段低估从 -94k 大幅缩小 (驱动压差提升直接线性增加 qg)
- k_frac 不再需要被压低补偿 dp 漂移
- WHP loss 变为 p_wf 的真约束 (不再有逃逸通道)
- Qg loss 有望突破 0.14 plateau (之前被 dp 漂移锁死)

预期指标: **MAPE ≤ 10%, R² ≥ 0.85**

## 十四、训练命令 (注意: 必须显式传 --steps!)

```powershell
python src/m5/run_m5_single_well.py --well SY9 --steps 20000 --device cuda
```
> ⚠️ 上次 config 写了 20000 但实际跑了 10000 (命令行 --steps 覆盖 config). 必须显式传 `--steps 20000`.
