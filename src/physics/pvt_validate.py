"""
M3 PVT 验收模块
自动出图（不同温度下各物性曲线）+ 自动检查报告

检查项:
    1. 所有输出有限且非 NaN/Inf
    2. Z > 0, Bg > 0, rho > 0, cg >= 0
    3. 随机采样 + 边界点全部通过
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_chinese_support, setup_logger, ensure_dir, write_markdown_report

# 中文字体必须在 import plt 之前配置
setup_chinese_support()
import matplotlib.pyplot as plt


class PVTValidator:
    """PVT 物性验收器"""
    
    def __init__(self, gas_pvt, output_dir: str = None):
        """
        Args:
            gas_pvt: GasPVT 实例
            output_dir: 输出图件和报告目录
        """
        setup_chinese_support()
        self.pvt = gas_pvt
        self.logger = setup_logger('PVTValidator')
        
        if output_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_dir = str(project_root / 'outputs' / 'mk_pinn_dt_v2')
        self.output_dir = output_dir
        self.fig_dir = os.path.join(output_dir, 'figs')
        ensure_dir(output_dir)
        ensure_dir(self.fig_dir)
        ensure_dir(os.path.join(output_dir, 'reports'))
    
    def validate_all(self, n_random: int = 200) -> Tuple[bool, List[str]]:
        """
        执行全部验收检查
        
        Args:
            n_random: 随机采样点数
            
        Returns:
            (是否通过, 检查项列表)
        """
        checks = []
        all_pass = True
        
        # 生成测试点
        p_min, p_max = self.pvt.get_pressure_range()
        T_min, T_max = self.pvt.get_temperature_range()
        
        np.random.seed(42)
        p_test = np.random.uniform(p_min, p_max, n_random)
        T_test = np.random.uniform(T_min, T_max, n_random)
        
        # 加入边界点
        p_boundary = np.array([p_min, p_max, p_min, p_max])
        T_boundary = np.array([T_min, T_min, T_max, T_max])
        p_all = np.concatenate([p_test, p_boundary])
        T_all = np.concatenate([T_test, T_boundary])
        
        # 查询所有物性
        props = self.pvt.query_all(p_all, T_all)
        
        # 检查 1: 有限值（非 NaN/Inf）
        for name, values in props.items():
            is_finite = np.all(np.isfinite(values))
            status = "PASS" if is_finite else "FAIL"
            checks.append(f"[{status}] {name}: 所有值有限（非NaN/Inf）")
            if not is_finite:
                all_pass = False
                n_bad = int(np.sum(~np.isfinite(values)))
                self.logger.error(f"{name}: {n_bad}/{len(values)} 个值非有限")
        
        # 检查 2: Z > 0
        z_positive = np.all(props['Z'] > 0)
        checks.append(f"[{'PASS' if z_positive else 'FAIL'}] Z > 0 (min={props['Z'].min():.6f})")
        if not z_positive:
            all_pass = False
        
        # 检查 3: Bg > 0
        bg_positive = np.all(props['Bg'] > 0)
        checks.append(f"[{'PASS' if bg_positive else 'FAIL'}] Bg > 0 (min={props['Bg'].min():.6e})")
        if not bg_positive:
            all_pass = False
        
        # 检查 4: rho > 0
        rho_positive = np.all(props['rho'] > 0)
        checks.append(f"[{'PASS' if rho_positive else 'FAIL'}] rho > 0 (min={props['rho'].min():.4f} kg/m³)")
        if not rho_positive:
            all_pass = False
        
        # 检查 5: cg >= 0
        cg_nonneg = np.all(props['cg'] >= 0)
        checks.append(f"[{'PASS' if cg_nonneg else 'FAIL'}] cg >= 0 (min={props['cg'].min():.6e} 1/MPa)")
        if not cg_nonneg:
            all_pass = False
        
        # 检查 6: alphaT > 0
        aT_positive = np.all(props['alphaT'] > 0)
        checks.append(f"[{'PASS' if aT_positive else 'FAIL'}] alphaT > 0 (min={props['alphaT'].min():.6e} 1/℃)")
        if not aT_positive:
            all_pass = False
        
        # 统计信息
        checks.append("")
        checks.append("--- 统计汇总 ---")
        for name, values in props.items():
            checks.append(
                f"  {name}: min={values.min():.6g}, max={values.max():.6g}, "
                f"mean={values.mean():.6g}"
            )
        
        if all_pass:
            self.logger.info("PVT 验收全部通过!")
        else:
            self.logger.warning("PVT 验收有未通过项!")
        
        return all_pass, checks
    
    def plot_curves(self, save: bool = True) -> str:
        """
        绘制不同温度下的 PVT 曲线（5 个子图）
        
        Returns:
            图片路径
        """
        self.logger.info("生成 PVT 曲线图...")
        
        p_min, p_max = self.pvt.get_pressure_range()
        p_arr = np.linspace(p_min, p_max, 200)
        temps = self.pvt.temperatures
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('M3 PVT 多温度物性曲线', fontsize=16, fontweight='bold')
        
        prop_info = [
            ('Z', 'Z 偏差系数', '', self.pvt.z),
            ('Bg', '体积系数 Bg', 'm³/m³', self.pvt.bg),
            ('cg', '压缩系数 cg', '1/MPa', self.pvt.cg),
            ('rho', '密度 ρ', 'kg/m³', self.pvt.rho),
            ('alphaT', '热膨胀系数 αT', '1/℃', self.pvt.alpha_T),
        ]
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))
        
        for idx, (key, title, unit, func) in enumerate(prop_info):
            ax = axes.flat[idx]
            T_list = temps if key != 'alphaT' else self.pvt.temperatures_alphaT
            colors_i = plt.cm.coolwarm(np.linspace(0, 1, len(T_list)))
            
            for j, T in enumerate(T_list):
                T_arr_full = np.full_like(p_arr, T)
                values = func(p_arr, T_arr_full)
                ax.plot(p_arr, values, color=colors_i[j], linewidth=1.5,
                        label=f'T={T:.1f}℃')
            
            ax.set_xlabel('压力 p (MPa)', fontsize=10)
            ylabel = f'{title}' + (f' ({unit})' if unit else '')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
        
        # 空白子图用于说明
        ax_info = axes.flat[5]
        ax_info.axis('off')
        info_text = (
            "M3 PVT 模块验收\n\n"
            f"压力范围: [{p_min:.1f}, {p_max:.1f}] MPa\n"
            f"温度数量: {len(temps)} 个\n"
            f"插值方法: PCHIP (单调保持)\n"
            f"温度方向: 线性插值\n\n"
            "特性:\n"
            "- 保持数据单调性\n"
            "- 避免过冲/振荡\n"
            "- 支持 2D 连续查询"
        )
        ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes,
                     fontsize=11, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.fig_dir, 'M3_pvt_curves.png')
            fig.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"PVT 曲线图已保存: {filepath}")
            return filepath
        
        plt.close(fig)
        return ''
    
    def generate_report(self) -> str:
        """生成 PVT 验收报告"""
        passed, checks = self.validate_all()
        
        lines = [
            "# M3 PVT 物性验收报告",
            "",
            f"**验收结果: {'✅ 全部通过' if passed else '❌ 存在未通过项'}**",
            "",
            "## 验收标准",
            "",
            "给定 (p, T) 能稳定返回全部物性，且满足：",
            "- Z > 0",
            "- Bg > 0", 
            "- ρ > 0",
            "- cg ≥ 0",
            "- αT > 0",
            "- 所有值有限（非 NaN/Inf）",
            "",
            "## 检查结果",
            "",
        ]
        
        for check in checks:
            if check.startswith('[PASS]'):
                lines.append(f"- ✅ {check[7:]}")
            elif check.startswith('[FAIL]'):
                lines.append(f"- ❌ {check[7:]}")
            else:
                lines.append(check)
        
        lines.extend([
            "",
            "## 插值方法",
            "",
            "- 压力方向: PCHIP (分段三次 Hermite, 保持单调)",
            "- 温度方向: 线性插值",
            f"- 压力范围: [{self.pvt.p_min:.1f}, {self.pvt.p_max:.1f}] MPa",
            f"- 温度: {self.pvt.temperatures} ℃",
            "",
            "## 图件",
            "",
            "- M3_pvt_curves.png: 各物性随压力变化曲线（不同温度）",
        ])
        
        report_path = os.path.join(self.output_dir, 'reports', 'M3_pvt_report.md')
        write_markdown_report(lines, report_path)
        self.logger.info(f"PVT 验收报告已保存: {report_path}")
        
        return report_path
