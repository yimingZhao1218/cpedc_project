# M6 验收/签收报告

**签收结果: ✅ 通过**

## 验收标准

1. **连通性分析（必选）**: M6_connectivity_report.md 或 k 场图、连通性矩阵 CSV
2. **消融实验（推荐）**: M6_ablation 下至少一组有 checkpoint/报告，或 reports/M6_ablation_report.md
3. **UQ（可选）**: M6_uq_report.md 或 M6_uq_*.png

签收条件：满足第 1 项即可签收；第 2、3 项为加分项。

## 检查结果

- ✅ M6 连通性报告存在: reports/M6_connectivity_report.md
- ✅ M6 消融报告存在: reports/M6_ablation_report.md
- ✅ M6 UQ 报告或图件存在

## 说明

- 连通性产出由 M5 训练后 `generate_report()` 自动调用，或单独运行连通性分析。
- 消融产出由 `python main.py --stage m6` 或 `python src/run_ablation_suite.py` 生成。
- UQ 产出由 `python src/run_uq_ensemble.py` 生成。
