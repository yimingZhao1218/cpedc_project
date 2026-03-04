"""
M4 PINN 基线验收
验收 1: SY9 压力趋势可拟合
验收 2: Sw 输出在 [0, 1] 内（sigmoid 输出层机制保证）
"""

import os
import sys
import numpy as np
from pathlib import Path

# 修复 OpenMP 库冲突（Windows + Anaconda + PyTorch 常见问题）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import torch
except ImportError:
    torch = None

# 脚本位于 src/m4/，需将 src 加入 path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import (
    setup_chinese_support, setup_logger, load_config,
    ensure_dir, write_markdown_report
)


class M4Validator:
    """M4 PINN 基线验收器"""
    
    def __init__(self, config_filename: str = 'config.yaml'):
        setup_chinese_support()
        self.logger = setup_logger('M4Validator')
        
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / config_filename
        self.config = load_config(str(config_path))
        for key, value in self.config['paths'].items():
            self.config['paths'][key] = str(project_root / value)
        
        self.output_dir = self.config['paths']['outputs']
        ensure_dir(self.output_dir)
        ensure_dir(os.path.join(self.output_dir, 'reports'))
    
    def _resolve_device(self, preferred: str = None):
        """
        解析运行设备。若指定了 preferred 则优先使用；
        否则尝试 CUDA，若当前 PyTorch 不支持本机 GPU（如 sm_120 等）则回退到 CPU。
        """
        if preferred is not None:
            preferred = preferred.strip().lower()
            if preferred in ('cuda', 'cpu'):
                return preferred
        if not torch.cuda.is_available():
            return 'cpu'
        # 先尝试在 GPU 上跑一个简单 op，避免“CUDA available 但内核不支持本机显卡”导致后面崩溃
        try:
            t = torch.zeros(1, device='cuda')
            _ = t + 1
            return 'cuda'
        except RuntimeError as e:
            msg = str(e).lower()
            if 'cuda' in msg or 'kernel' in msg or 'device' in msg:
                self.logger.warning(
                    "当前 PyTorch 无法在本机 GPU 上运行（可能显卡架构过新，如 RTX 50 系 sm_120），已回退到 CPU 验收。"
                )
                self.logger.warning(f"  错误信息: {e}")
            return 'cpu'
    
    def run(self, device: str = None) -> bool:
        """
        执行 M4 完整验收。
        device: 可选，'cuda'/'cpu'；不传则自动选择（CUDA 不可用时用 CPU，避免 PyTorch 与显卡架构不兼容导致崩溃）。
        """
        if torch is None:
            self.logger.error("PyTorch 未安装，无法执行 M4 验收")
            return False
        
        self.logger.info("=" * 60)
        self.logger.info("M4 PINN 基线验收开始")
        self.logger.info("=" * 60)
        
        all_pass = True
        checks = []
        self._pressure_mape_threshold = (
            self.config.get('m4_config', {}).get('acceptance', {}).get('pressure_mape_threshold', 30.0)
        )
        
        # 加载模型
        from pinn.model import PINNNet
        from pinn.sampler import PINNSampler
        
        device = self._resolve_device(device)
        self.logger.info(f"  验收设备: {device}")
        
        model = PINNNet(self.config).to(device)
        sampler = PINNSampler(config=self.config)
        
        # P0/P1: 优先 pinn_best.pt，缺失则 pinn_final.pt，再缺失则验收不通过（不沿用随机权重判通过）
        ckpt_dir = self.config['paths'].get('checkpoints',
                                             os.path.join(self.output_dir, 'ckpt'))
        ckpt_best = os.path.join(ckpt_dir, 'pinn_best.pt')
        ckpt_final = os.path.join(ckpt_dir, 'pinn_final.pt')
        ckpt_loaded = False
        if os.path.exists(ckpt_best):
            ckpt = torch.load(ckpt_best, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            self.logger.info(f"已加载最佳模型: {ckpt_best}")
            ckpt_loaded = True
        elif os.path.exists(ckpt_final):
            ckpt = torch.load(ckpt_final, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            self.logger.info(f"已加载最终模型（best 缺失）: {ckpt_final}")
            ckpt_loaded = True
        if not ckpt_loaded:
            self.logger.error("未找到训练检查点 (pinn_best.pt / pinn_final.pt)，验收不通过")
            checks.append("[FAIL] 未找到训练检查点 (pinn_best.pt / pinn_final.pt)，验收不通过")
            all_pass = False
        
        model.eval()
        
        # 井筒压差：模型输出地层压力, p_obs 是井口压力
        # dp_wellbore = 地层压力 - 井口压力 ≈ 18 MPa (从 config 读取)
        mk = self.config.get('mk_formation', {})
        dp_wb_cfg = mk.get('dp_wellbore_MPa', {})
        self.dp_wellbore = (dp_wb_cfg.get('value', 18.0)
                            if isinstance(dp_wb_cfg, dict) else float(dp_wb_cfg))
        self.logger.info(f"  dp_wellbore = {self.dp_wellbore} MPa (地层→井口压差修正)")
        
        # ============= 验收 1: SY9 压力趋势 =============
        self.logger.info("\n>>> 验收 1: SY9 压力趋势可拟合")
        
        well_data = sampler.sample_well_data('SY9')
        self._sy9_nan_ratio = well_data.get('nan_ratio') if well_data else None
        self._sy9_missing_runs = well_data.get('missing_runs', []) if well_data else []
        self._sy9_total_points = well_data.get('total_points') if well_data else None
        self._sy9_valid_points = well_data.get('valid_points') if well_data else None
        if well_data:
            xyt_tensor = torch.from_numpy(well_data['xyt']).float().to(device)
            p_obs = well_data['p_obs']  # 井口压力 (WHP)
            
            with torch.no_grad():
                p_pred_form, sw_pred = model(xyt_tensor)
                p_pred_form_np = p_pred_form.cpu().numpy().flatten()  # 地层压力
                sw_pred_np = sw_pred.cpu().numpy().flatten()
            
            # 关键修正：将模型输出的地层压力转为井口压力再与观测对比
            p_pred_whp = p_pred_form_np - self.dp_wellbore  # 预测井口压力
            
            self.logger.info(f"  模型地层压力范围: [{p_pred_form_np.min():.2f}, {p_pred_form_np.max():.2f}] MPa")
            self.logger.info(f"  修正后井口压力范围: [{p_pred_whp.min():.2f}, {p_pred_whp.max():.2f}] MPa")
            self.logger.info(f"  观测井口压力范围: [{p_obs.min():.2f}, {p_obs.max():.2f}] MPa")
            
            # 计算指标（用修正后的井口压力对比）
            valid = np.isfinite(p_obs) & np.isfinite(p_pred_whp) & (p_obs > 0)
            if np.any(valid):
                # MAPE 带 ε 避免低压点爆炸: |p - p_hat| / (p + ε)
                _eps_mape = 1.0  # MPa，防止 p→0 时除零
                mape = np.mean(np.abs((p_obs[valid] - p_pred_whp[valid])
                                      / (p_obs[valid] + _eps_mape))) * 100
                rmse = np.sqrt(np.mean((p_obs[valid] - p_pred_whp[valid]) ** 2))
                
                # 趋势相关性（常数列时 pearsonr 会报错或返回 nan，做防护）
                p_obs_v = p_obs[valid]
                p_pred_v = p_pred_whp[valid]
                corr = np.nan
                if np.var(p_obs_v) > 1e-12 and np.var(p_pred_v) > 1e-12:
                    try:
                        from scipy.stats import pearsonr
                        corr, _ = pearsonr(p_obs_v, p_pred_v)
                        if not np.isfinite(corr):
                            corr = np.nan
                    except Exception:
                        corr = np.nan
                
                # 基线验收: MAPE 与阈值比较（阈值已在 run 开头从 config 读入）
                pressure_pass = mape < self._pressure_mape_threshold
                checks.append(f"[{'PASS' if pressure_pass else 'FAIL'}] "
                              f"SY9 压力 MAPE = {mape:.2f}% (阈值 < {self._pressure_mape_threshold}%, dp_wellbore={self.dp_wellbore}MPa)")
                checks.append(f"[INFO] RMSE = {rmse:.2f} MPa")
                checks.append(f"[INFO] Pearson 相关系数 = {corr:.4f}" if np.isfinite(corr) else "[INFO] Pearson 相关系数 = N/A (常数列或计算异常)")
                checks.append(f"[INFO] 地层压力预测范围: [{p_pred_form_np.min():.2f}, {p_pred_form_np.max():.2f}] MPa")
                checks.append(f"[INFO] 井口压力预测范围: [{p_pred_whp.min():.2f}, {p_pred_whp.max():.2f}] MPa")
                
                # P1: 有效性门槛 — 预测非常数、相关系数可计算
                if np.var(p_pred_whp) < 1e-12:
                    checks.append("[FAIL] 预测压力为常数，未学到时间趋势")
                    all_pass = False
                if not np.isfinite(corr):
                    checks.append("[FAIL] Pearson 相关系数不可计算（预测或观测为常数列），趋势未拟合")
                    all_pass = False
                if not pressure_pass:
                    all_pass = False
            else:
                checks.append("[FAIL] 无有效压力数据进行对比")
                all_pass = False
        else:
            checks.append("[FAIL] 无法加载 SY9 数据")
            all_pass = False
        
        # ============= 验收 2: Sw 范围 =============
        self.logger.info("\n>>> 验收 2: Sw 输出在 [0, 1] 内")
        
        # 大规模随机采样检查
        np.random.seed(42)
        x_check = sampler.sample_domain(5000, seed=42)
        x_tensor = torch.from_numpy(x_check).float().to(device)
        
        with torch.no_grad():
            _, sw_all = model(x_tensor)
            sw_np = sw_all.cpu().numpy().flatten()
        
        sw_in_range = np.all((sw_np >= 0) & (sw_np <= 1))
        sw_min = sw_np.min()
        sw_max = sw_np.max()
        
        checks.append(f"[{'PASS' if sw_in_range else 'FAIL'}] "
                       f"Sw ∈ [0, 1] (实际范围: [{sw_min:.6f}, {sw_max:.6f}])")
        
        if not sw_in_range:
            all_pass = False
        
        # 输出层机制说明（bounded-tanh + clamp）
        checks.append("[INFO] Sw 使用 bounded-tanh + clamp，保证 [0, 1] 物理范围")
        
        # ============= 验收 3: 模型结构检查 =============
        self.logger.info("\n>>> 验收 3: 模型结构检查")
        
        n_params = model.count_parameters()
        checks.append(f"[PASS] 模型参数量: {n_params:,}")
        
        # 输出维度检查
        test_input = torch.randn(10, 3).to(device)
        with torch.no_grad():
            p_test, sw_test = model(test_input)
        
        shape_ok = (p_test.shape == (10, 1)) and (sw_test.shape == (10, 1))
        checks.append(f"[{'PASS' if shape_ok else 'FAIL'}] "
                       f"输出维度: p={p_test.shape}, Sw={sw_test.shape}")
        if not shape_ok:
            all_pass = False
        
        # ============= 生成报告 =============
        self._generate_report(all_pass, checks)
        
        # 日志
        for check in checks:
            self.logger.info(f"  {check}")
        
        self.logger.info("=" * 60)
        if all_pass:
            self.logger.info("M4 验收结果: ✅ 全部通过")
        else:
            self.logger.info("M4 验收结果: ❌ 存在未通过项")
        self.logger.info("=" * 60)
        
        return all_pass
    
    def _generate_report(self, passed: bool, checks: list):
        """生成 M4 验收报告"""
        lines = [
            "# M4 PINN 基线验收报告",
            "",
            f"**验收结果: {'✅ 全部通过' if passed else '❌ 存在未通过项'}**",
            "",
            "## 验收标准",
            "",
            f"1. SY9 压力趋势可拟合 (MAPE < {getattr(self, '_pressure_mape_threshold', 30.0)}%)",
            "2. Sw 输出严格在 [0, 1] 内",
            "3. 模型结构正确（输入3D → 输出 p + Sw）",
            "",
            "## 检查结果",
            "",
        ]
        
        for check in checks:
            if check.startswith('[PASS]'):
                lines.append(f"- ✅ {check[7:]}")
            elif check.startswith('[FAIL]'):
                lines.append(f"- ❌ {check[7:]}")
            elif check.startswith('[INFO]'):
                lines.append(f"- {check[7:]}")
            else:
                lines.append(check)
        
        lines.extend([
            "",
            "## SY9 井口压力观测缺失审计",
            "",
        ])
        if self._sy9_nan_ratio is not None:
            lines.append(f"- **nan_ratio（缺失率）**: {self._sy9_nan_ratio:.2%}")
        if self._sy9_total_points is not None and self._sy9_valid_points is not None:
            lines.append(f"- **总点数 / 有效点数**: {self._sy9_total_points} / {self._sy9_valid_points}")
        if self._sy9_missing_runs:
            lines.append("- **缺失段清单（起止日期、长度/天）**:")
            for run in self._sy9_missing_runs:
                start_idx, end_idx, start_date, end_date, length_days = run
                lines.append(f"  - [{start_date} ~ {end_date}] 长度 {length_days} 天")
        lines.extend([
            "",
            "说明：MAPE/RMSE 仅在有效观测点计算；缺失区间为模型在物理约束下的插值/外推。",
            "",
            "## 训练策略",
            "",
            "分阶段训练（防翻车）：",
            "- Stage A: IC + BC 预训练（场形态先像物理）",
            "- Stage B: 逐步引入 PDE 残差",
            "- Stage C: 加入数据同化（SY9 压力锚点）",
            "- Stage D: 全损失微调",
            "",
            "## 图件",
            "",
            "- M4_training_history.png: 训练曲线（损失、Sw范围、学习率）",
            "- M4_pressure_comparison.png: SY9 压力预测 vs 观测（缺失区间已用灰色阴影标注）",
        ])
        
        report_path = os.path.join(self.output_dir, 'reports', 'M4_validation_report.md')
        write_markdown_report(lines, report_path)
        self.logger.info(f"M4 验收报告已保存: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="M4 PINN 基线验收")
    parser.add_argument("--device", default=None, help="运行设备: cuda 或 cpu（不传则自动选择，CUDA 不可用时用 CPU）")
    args = parser.parse_args()
    validator = M4Validator()
    success = validator.run(device=args.device)
    sys.exit(0 if success else 1)
