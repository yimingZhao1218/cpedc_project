"""
M6 验收/签收
============
检查 M6 交付物是否齐全，便于签收。

验收项（满足其一即可签收，推荐两项都有）:
  1. 连通性分析: M6_connectivity_report.md 或 M6_k_field_channels.png、M6_connectivity_matrix.csv
  2. 消融实验: M6_ablation 下至少一组有 checkpoint/报告，或主 reports 下有 M6_ablation_report.md
  3. 可选: M6_uq_report.md、M6_uq_*.png
"""

import os
import sys
from pathlib import Path

_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import setup_chinese_support, setup_logger, load_config, ensure_dir, write_markdown_report


class M6Validator:
    """M6 验收/签收器（仅检查输出文件，不加载模型）"""

    def __init__(self, config_filename: str = 'config.yaml'):
        setup_chinese_support()
        self.logger = setup_logger('M6Validator')
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / config_filename
        self.config = load_config(str(config_path))
        for key, value in self.config['paths'].items():
            self.config['paths'][key] = str(project_root / value)
        self.output_dir = self.config['paths']['outputs']
        self.report_dir = self.config['paths'].get('reports', os.path.join(self.output_dir, 'reports'))
        self.fig_dir = self.config['paths'].get('figures', os.path.join(self.output_dir, 'figs'))
        self.m6_ablation_dir = os.path.join(self.output_dir, 'M6_ablation')

    def run(self) -> bool:
        self.logger.info("=" * 60)
        self.logger.info("M6 验收/签收开始")
        self.logger.info("=" * 60)

        checks = []
        conn_ok = False
        ablation_ok = False
        uq_ok = False

        # ---------- 1. 连通性分析（核心交付物）----------
        conn_report = os.path.join(self.report_dir, 'M6_connectivity_report.md')
        conn_fig_k = os.path.join(self.fig_dir, 'M6_k_field_channels.png')
        conn_fig_c = os.path.join(self.fig_dir, 'M6_connectivity_matrix.png')
        conn_csv = os.path.join(self.report_dir, 'M6_connectivity_matrix.csv')
        if os.path.isfile(conn_report):
            checks.append("[PASS] M6 连通性报告存在: reports/M6_connectivity_report.md")
            conn_ok = True
        elif os.path.isfile(conn_fig_k) or os.path.isfile(conn_csv):
            checks.append("[PASS] M6 连通性图件或矩阵存在 (k 场图或 CSV)")
            conn_ok = True
        else:
            checks.append("[FAIL] 未找到 M6 连通性交付物 (需 M5 训练后 generate_report 或单独跑连通性)")

        # ---------- 2. 消融实验 ----------
        ablation_report = os.path.join(self.report_dir, 'M6_ablation_report.md')
        if os.path.isfile(ablation_report):
            checks.append("[PASS] M6 消融报告存在: reports/M6_ablation_report.md")
            ablation_ok = True
        elif os.path.isdir(self.m6_ablation_dir):
            subdirs = [d for d in os.listdir(self.m6_ablation_dir)
                       if os.path.isdir(os.path.join(self.m6_ablation_dir, d))]
            for d in subdirs:
                ckpt = os.path.join(self.m6_ablation_dir, d, 'ckpt', 'm5_pinn_best.pt')
                rpt = os.path.join(self.m6_ablation_dir, d, 'reports', 'M5_inversion_params.json')
                if os.path.isfile(ckpt) or os.path.isfile(rpt):
                    ablation_ok = True
                    break
            if ablation_ok:
                checks.append(f"[PASS] M6 消融至少一组有产出: M6_ablation/ 下 {len(subdirs)} 组")
            else:
                checks.append("[FAIL] M6_ablation 下无有效 checkpoint 或报告")
        else:
            checks.append("[INFO] 未找到 M6 消融产出 (可选，需 main --stage m6 或 src/m6/run_ablation_suite.py)")

        # ---------- 3. UQ（可选）----------
        uq_report = os.path.join(self.report_dir, 'M6_uq_report.md')
        uq_fig = os.path.join(self.fig_dir, 'M6_uq_qg_p10p50p90.png')
        if os.path.isfile(uq_report) or os.path.isfile(uq_fig):
            checks.append("[PASS] M6 UQ 报告或图件存在")
            uq_ok = True
        else:
            checks.append("[INFO] 未找到 M6 UQ 产出 (可选，需 src/m6/run_uq_ensemble.py)")

        # 签收条件：连通性必须通过；消融或 UQ 至少有一项有则更佳
        passed = conn_ok
        if not passed:
            self.logger.warning("M6 签收未通过：缺少连通性交付物（请先完成 M5 并生成报告，或运行连通性分析）")
        else:
            if ablation_ok or uq_ok:
                self.logger.info("M6 签收通过（连通性 + 消融/UQ 均有产出）")
            else:
                self.logger.info("M6 签收通过（连通性满足；消融/UQ 可选）")

        self._generate_report(passed, checks)
        self.logger.info("=" * 60)
        return passed

    def _generate_report(self, passed: bool, checks: list):
        lines = [
            "# M6 验收/签收报告",
            "",
            f"**签收结果: {'✅ 通过' if passed else '❌ 未通过'}**",
            "",
            "## 验收标准",
            "",
            "1. **连通性分析（必选）**: M6_connectivity_report.md 或 k 场图、连通性矩阵 CSV",
            "2. **消融实验（推荐）**: M6_ablation 下至少一组有 checkpoint/报告，或 reports/M6_ablation_report.md",
            "3. **UQ（可选）**: M6_uq_report.md 或 M6_uq_*.png",
            "",
            "签收条件：满足第 1 项即可签收；第 2、3 项为加分项。",
            "",
            "## 检查结果",
            "",
        ]
        for check in checks:
            if check.startswith("[PASS]"):
                lines.append(f"- ✅ {check[7:]}")
            elif check.startswith("[FAIL]"):
                lines.append(f"- ❌ {check[7:]}")
            elif check.startswith("[INFO]"):
                lines.append(f"- ℹ️ {check[7:]}")
            else:
                lines.append(f"- {check}")
        lines.extend([
            "",
            "## 说明",
            "",
            "- 连通性产出由 M5 训练后 `generate_report()` 自动调用，或单独运行连通性分析。",
            "- 消融产出由 `python main.py --stage m6` 或 `python src/m6/run_ablation_suite.py` 生成。",
            "- UQ 产出由 `python src/m6/run_uq_ensemble.py` 生成。",
            "",
        ])
        ensure_dir(self.report_dir)
        report_path = os.path.join(self.report_dir, 'M6_validation_report.md')
        write_markdown_report(lines, report_path)
        self.logger.info(f"M6 验收报告已保存: {report_path}")


if __name__ == "__main__":
    import sys
    setup_chinese_support()
    validator = M6Validator()
    success = validator.run()
    sys.exit(0 if success else 1)
