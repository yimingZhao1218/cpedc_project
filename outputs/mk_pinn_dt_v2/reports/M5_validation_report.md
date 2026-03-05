# M5 井—藏耦合同化验收报告 (v3)

## 参数来源 (final checkpoint)
- Source tag: final
- Step: final
- Checkpoint 时间: 2026-03-05 08:29:17

## 训练配置
- 总步数: 80000
- 学习率: 0.001
- ReLoBRaLo: True
- RAR: True
- RAR 点数: {'n_rar_points': 8000, 'total_points': 8000}
- Fourier Features: True
- k_net: True
- 参数量: {'field_net': 416258, 'well_model': 4420, 'k_net': 15297, 'well_singularity': 1153, 'dp_wellbore': 0, 'total': 437128}

## 反演参数
- k_frac: 10.1299 mD
- f_frac: (已合并入 k_frac)
- r_e: 128.9 m
- dp_wellbore: 13.30 MPa (WHP→p_wf 井筒压差)

## 最终损失 (final checkpoint)
- Total: 3.106025e+02
- PDE: 3.730155e+02
- qg: 8.328896e-02
- k_reg: 1.192521e-01

## 当前模型 qg 拟合指标 (final checkpoint)
- 井 SY9: MAPE=4.6%, RMSE=41752 m³/d, R²=0.9350
  - train: MAPE=5.2%, RMSE=45051, R²=0.9263
  - val: MAPE=3.5%, RMSE=28342, R²=0.8816
  - test: MAPE=0.7%, RMSE=9808, R²=0.9222
  - val 开井点数: 32, test 开井点数: 121
  - 关井段 (q≤1): n=378, MAE=189 m³/d, pred_mean=189 m³/d
  - 高产段 (q≥504878): n=239, MAPE=3.8%, mean_error=-14076 m³/d

## 当前模型 p_wf 拟合指标 (final checkpoint, 相对 WHP+dp_wellbore)
- 井 SY9: MAPE=0.5%, RMSE=1.18 MPa, R²=0.9555
  - train: MAPE=0.5%, RMSE=1.28 MPa, R²=0.8977
  - val: MAPE=0.3%, RMSE=0.35 MPa, R²=0.6795
  - test: MAPE=0.3%, RMSE=0.26 MPa, R²=0.8832

## 对照 checkpoint 诊断 (best 对照)
- 用途: 与报告主模型对照，辅助判断过拟合/欠拟合。
- 最终步损失: Total=3.106025e+02, PDE=3.730155e+02, qg=8.328896e-02
- 井 SY9 qg (best): MAPE=4.6%, RMSE=41774 m³/d, R²=0.9349
- 井 SY9 p_wf (best): MAPE=0.5%, RMSE=1.18 MPa, R²=0.9555