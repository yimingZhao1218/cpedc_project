# M5 井—藏耦合同化验收报告 (v3)

## 参数来源 (final checkpoint)
- Source tag: final
- Step: final
- Checkpoint 时间: 2026-03-04 17:31:02

## 训练配置
- 总步数: 20000
- 学习率: 0.0005
- ReLoBRaLo: True
- RAR: True
- RAR 点数: {'n_rar_points': 7800, 'total_points': 7800}
- Fourier Features: True
- k_net: True
- 参数量: {'field_net': 416258, 'well_model': 4420, 'k_net': 15297, 'well_singularity': 1153, 'dp_wellbore': 0, 'total': 437128}

## 反演参数
- k_frac: 9.6803 mD
- f_frac: (已合并入 k_frac)
- r_e: 128.9 m
- dp_wellbore: 13.30 MPa (WHP→p_wf 井筒压差)

## 最终损失 (final checkpoint)
- Total: 3.469061e+02
- PDE: 6.621138e+02
- qg: 1.135855e-01
- k_reg: 1.343427e-08

## 当前模型 qg 拟合指标 (final checkpoint)
- 井 SY9: MAPE=8.4%, RMSE=49596 m³/d, R²=0.9083
  - train: MAPE=9.5%, RMSE=52601, R²=0.8995
  - val: MAPE=5.5%, RMSE=46241, R²=0.6849
  - test: MAPE=1.8%, RMSE=22614, R²=0.5865
  - val 开井点数: 32, test 开井点数: 121
  - 关井段 (q≤1): n=378, MAE=206 m³/d, pred_mean=206 m³/d
  - 高产段 (q≥504878): n=239, MAPE=4.3%, mean_error=-21193 m³/d

## 当前模型 p_wf 拟合指标 (final checkpoint, 相对 WHP+dp_wellbore)
- 井 SY9: MAPE=0.5%, RMSE=1.19 MPa, R²=0.9547
  - train: MAPE=0.5%, RMSE=1.29 MPa, R²=0.8959
  - val: MAPE=0.3%, RMSE=0.42 MPa, R²=0.5205
  - test: MAPE=0.2%, RMSE=0.22 MPa, R²=0.9157

## 对照 checkpoint 诊断 (best 对照)
- 用途: 与报告主模型对照，辅助判断过拟合/欠拟合。
- 最终步损失: Total=3.469061e+02, PDE=6.621138e+02, qg=1.135855e-01
- 井 SY9 qg (best): MAPE=8.4%, RMSE=49676 m³/d, R²=0.9080
- 井 SY9 p_wf (best): MAPE=0.5%, RMSE=1.19 MPa, R²=0.9547