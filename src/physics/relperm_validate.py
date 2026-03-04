"""
M3 气水相渗验收模块
自动检查: 单调性、端点条件、非负性
自动出图: 气水相渗曲线（插值后）

验收条件:
    - krw(Sw) 随 Sw 单调递增
    - krg(Sw) 随 Sw 单调递减
    - 端点: krw(Swr) ≈ 0, krg(1-Sgr) ≈ 0
    - 全域: 0 ≤ kr ≤ 1
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_chinese_support, setup_logger, ensure_dir, write_markdown_report

# 中文字体必须在 import plt 之前配置
setup_chinese_support()
import matplotlib.pyplot as plt


class RelPermValidator:
    """气水相渗验收器"""
    
    def __init__(self, relperm, output_dir: str = None):
        """
        Args:
            relperm: RelPermGW 实例
            output_dir: 输出目录
        """
        setup_chinese_support()
        self.rp = relperm
        self.logger = setup_logger('RelPermValidator')
        
        if output_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_dir = str(project_root / 'outputs' / 'mk_pinn_dt_v2')
        self.output_dir = output_dir
        self.fig_dir = os.path.join(output_dir, 'figs')
        ensure_dir(output_dir)
        ensure_dir(self.fig_dir)
        ensure_dir(os.path.join(output_dir, 'reports'))
    
    def validate_all(self, n_grid: int = 500) -> Tuple[bool, List[str]]:
        """
        执行全部相渗验收检查
        
        Args:
            n_grid: Sw 网格点数
            
        Returns:
            (是否通过, 检查项列表)
        """
        checks = []
        all_pass = True
        
        sw_min, sw_max = self.rp.get_sw_range()
        sw_grid = np.linspace(sw_min, sw_max, n_grid)
        
        krw_vals = self.rp.krw(sw_grid)
        krg_vals = self.rp.krg(sw_grid)
        
        # 检查 1: krw 单调递增
        krw_diff = np.diff(krw_vals)
        krw_mono = np.all(krw_diff >= -1e-10)
        checks.append(f"[{'PASS' if krw_mono else 'FAIL'}] krw(Sw) 单调递增 "
                       f"(min_diff={krw_diff.min():.2e})")
        if not krw_mono:
            all_pass = False
        
        # 检查 2: krg 单调递减
        krg_diff = np.diff(krg_vals)
        krg_mono = np.all(krg_diff <= 1e-10)
        checks.append(f"[{'PASS' if krg_mono else 'FAIL'}] krg(Sw) 单调递减 "
                       f"(max_diff={krg_diff.max():.2e})")
        if not krg_mono:
            all_pass = False
        
        # 检查 3: 端点 - krw(Swr) ≈ 0
        krw_at_swr = np.asarray(self.rp.krw(sw_min)).item()
        ep_krw = krw_at_swr < 0.01
        checks.append(f"[{'PASS' if ep_krw else 'FAIL'}] krw(Swr={sw_min:.4f}) = "
                       f"{krw_at_swr:.6f} ≈ 0")
        if not ep_krw:
            all_pass = False
        
        # 检查 4: 端点 - krg(1-Sgr) ≈ 0
        krg_at_max = np.asarray(self.rp.krg(sw_max)).item()
        ep_krg = krg_at_max < 0.02
        checks.append(f"[{'PASS' if ep_krg else 'FAIL'}] krg(Sw={sw_max:.4f}) = "
                       f"{krg_at_max:.6f} ≈ 0")
        if not ep_krg:
            all_pass = False
        
        # 检查 5: 全域非负 krw
        krw_nonneg = np.all(krw_vals >= 0)
        checks.append(f"[{'PASS' if krw_nonneg else 'FAIL'}] krw >= 0 "
                       f"(min={krw_vals.min():.6e})")
        if not krw_nonneg:
            all_pass = False
        
        # 检查 6: 全域非负 krg
        krg_nonneg = np.all(krg_vals >= 0)
        checks.append(f"[{'PASS' if krg_nonneg else 'FAIL'}] krg >= 0 "
                       f"(min={krg_vals.min():.6e})")
        if not krg_nonneg:
            all_pass = False
        
        # 检查 7: 上限 krw <= 1
        krw_upper = np.all(krw_vals <= 1.0)
        checks.append(f"[{'PASS' if krw_upper else 'FAIL'}] krw <= 1 "
                       f"(max={krw_vals.max():.6f})")
        if not krw_upper:
            all_pass = False
        
        # 检查 8: 上限 krg <= 1
        krg_upper = np.all(krg_vals <= 1.0)
        checks.append(f"[{'PASS' if krg_upper else 'FAIL'}] krg <= 1 "
                       f"(max={krg_vals.max():.6f})")
        if not krg_upper:
            all_pass = False
        
        # 端点信息
        Swr, Sgr, krw_max, krg_max = self.rp.endpoints()
        checks.append("")
        checks.append("--- 端点参数 ---")
        checks.append(f"  束缚水饱和度 Swr = {Swr:.4f}")
        checks.append(f"  残余气饱和度 Sgr = {Sgr:.4f}")
        checks.append(f"  最大水相相渗 krw_max = {krw_max:.4f}")
        checks.append(f"  最大气相相渗 krg_max = {krg_max:.4f}")
        
        if all_pass:
            self.logger.info("相渗验收全部通过!")
        else:
            self.logger.warning("相渗验收有未通过项!")
        
        return all_pass, checks
    
    @staticmethod
    def _corey_curves(sw_grid, Swc, Sgr, krg_max, krw_max, ng=2.0, nw=3.0):
        """Corey-Brooks 解析计算 kr 和 dkr/dSw"""
        denom = 1.0 - Swc - Sgr
        Se_g = np.clip((1.0 - sw_grid - Sgr) / denom, 0, 1)
        Se_w = np.clip((sw_grid - Swc) / denom, 0, 1)
        
        krg_c = krg_max * Se_g ** ng
        krw_c = krw_max * Se_w ** nw
        
        # 解析导数
        Se_g_safe = np.clip(Se_g, 1e-8, 1.0)
        Se_w_safe = np.clip(Se_w, 1e-8, 1.0)
        dkrg_c = krg_max * ng * Se_g_safe ** (ng - 1) * (-1.0 / denom)
        dkrw_c = krw_max * nw * Se_w_safe ** (nw - 1) * (1.0 / denom)
        
        return krg_c, krw_c, dkrg_c, dkrw_c

    def plot_curves(self, save: bool = True) -> str:
        """
        绘制气水相渗曲线（PCHIP 插值 + Corey 拟合 + 解析导数）
        
        Returns:
            图片路径
        """
        self.logger.info("生成气水相渗曲线图...")
        
        sw_min, sw_max = self.rp.get_sw_range()
        sw_grid = np.linspace(sw_min, sw_max, 500)
        
        krw_interp = self.rp.krw(sw_grid)
        krg_interp = self.rp.krg(sw_grid)
        
        Swr, Sgr, krw_max, krg_max = self.rp.endpoints()
        
        # Corey 参数 (附表7 SY13 21点最小二乘拟合, 与 PINN TorchRelPerm 一致)
        ng, nw = 1.0846, 4.4071
        krg_corey, krw_corey, dkrg_corey, dkrw_corey = self._corey_curves(
            sw_grid, Swr, Sgr, krg_max, krw_max, ng, nw
        )
        
        # ===== 配色方案 =====
        c_gas = '#D32F2F'       # 气相红
        c_gas_light = '#FFCDD2' # 气相淡红
        c_water = '#1565C0'     # 水相蓝
        c_water_light = '#BBDEFB'  # 水相淡蓝
        c_corey = '#FF9800'     # Corey 拟合橙
        c_corey_w = '#4CAF50'   # Corey 拟合绿
        c_bg = '#FAFAFA'
        c_grid = '#E0E0E0'
        c_annot = '#616161'
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('white')
        fig.suptitle('M3 气水两相相渗曲线', fontsize=18, fontweight='bold', y=0.98)
        
        # =================== 左图: 相渗曲线 ===================
        ax1 = axes[0]
        ax1.set_facecolor(c_bg)
        
        # 端点区域填充
        ax1.axvspan(0, Swr, alpha=0.06, color=c_water, zorder=0)
        ax1.axvspan(1 - Sgr, 1, alpha=0.06, color=c_gas, zorder=0)
        
        # PCHIP 插值曲线
        ax1.plot(sw_grid, krg_interp, color=c_gas, linewidth=2.5,
                 label='krg (PCHIP 插值)', zorder=3)
        ax1.plot(sw_grid, krw_interp, color=c_water, linewidth=2.5,
                 label='krw (PCHIP 插值)', zorder=3)
        
        # Corey 拟合曲线
        ax1.plot(sw_grid, krg_corey, color=c_corey, linewidth=1.5,
                 linestyle='--', alpha=0.85, label=f'krg Corey (ng={ng:.1f})', zorder=2)
        ax1.plot(sw_grid, krw_corey, color=c_corey_w, linewidth=1.5,
                 linestyle='--', alpha=0.85, label=f'krw Corey (nw={nw:.1f})', zorder=2)
        
        # 原始数据点
        ax1.scatter(self.rp.sw_data, self.rp.krg_data, c=c_gas, s=40, zorder=5,
                    edgecolors='#B71C1C', linewidths=0.8, marker='o', label='krg 实测 (附表7)')
        ax1.scatter(self.rp.sw_data, self.rp.krw_data, c=c_water, s=40, zorder=5,
                    edgecolors='#0D47A1', linewidths=0.8, marker='s', label='krw 实测 (附表7)')
        
        # 端点标注线
        ax1.axvline(x=Swr, color=c_annot, linestyle='--', alpha=0.6, linewidth=1)
        ax1.axvline(x=1 - Sgr, color=c_annot, linestyle=':', alpha=0.6, linewidth=1)
        ax1.annotate(f'Swr={Swr:.3f}', xy=(Swr, 0.92), fontsize=9,
                     color=c_annot, ha='left', va='top',
                     xytext=(Swr + 0.02, 0.92))
        ax1.annotate(f'1-Sgr={1 - Sgr:.3f}', xy=(1 - Sgr, 0.92), fontsize=9,
                     color=c_annot, ha='right', va='top',
                     xytext=(1 - Sgr - 0.02, 0.92))
        
        # 等渗点标注
        diff_kr = np.abs(krg_interp - krw_interp)
        idx_cross = np.argmin(diff_kr)
        sw_cross = sw_grid[idx_cross]
        kr_cross = 0.5 * (krg_interp[idx_cross] + krw_interp[idx_cross])
        ax1.plot(sw_cross, kr_cross, 'k*', markersize=12, zorder=6)
        ax1.annotate(f'等渗点\nSw={sw_cross:.3f}\nkr={kr_cross:.3f}',
                     xy=(sw_cross, kr_cross), fontsize=8, color='#333',
                     xytext=(sw_cross + 0.06, kr_cross + 0.12),
                     arrowprops=dict(arrowstyle='->', color='#666', lw=1),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                               edgecolor='#FBC02D', alpha=0.9))
        
        ax1.set_xlabel('含水饱和度 Sw', fontsize=13)
        ax1.set_ylabel('相对渗透率 kr', fontsize=13)
        ax1.set_title('气水相渗曲线 (SY13 MK组)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=8.5, loc='center right', framealpha=0.9,
                   edgecolor=c_grid, fancybox=True)
        ax1.grid(True, alpha=0.4, color=c_grid, linewidth=0.5)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([-0.02, 1.02])
        ax1.tick_params(labelsize=11)
        
        # =================== 右图: 解析导数 (Corey) ===================
        ax2 = axes[1]
        ax2.set_facecolor(c_bg)
        
        # Corey 解析导数（光滑）
        ax2.plot(sw_grid, dkrg_corey, color=c_gas, linewidth=2.5,
                 label='dkrg/dSw (Corey 解析)')
        ax2.plot(sw_grid, dkrw_corey, color=c_water, linewidth=2.5,
                 label='dkrw/dSw (Corey 解析)')
        
        # PCHIP 导数（淡色参考）
        dkrw_pchip = self.rp.dkrw_dsw(sw_grid)
        dkrg_pchip = self.rp.dkrg_dsw(sw_grid)
        ax2.plot(sw_grid, dkrg_pchip, color=c_gas_light, linewidth=1.2,
                 linestyle=':', alpha=0.7, label='dkrg/dSw (PCHIP)')
        ax2.plot(sw_grid, dkrw_pchip, color=c_water_light, linewidth=1.2,
                 linestyle=':', alpha=0.7, label='dkrw/dSw (PCHIP)')
        
        ax2.axhline(y=0, color='#9E9E9E', linestyle='-', alpha=0.4, linewidth=0.8)
        
        # 填充正负区域
        ax2.fill_between(sw_grid, 0, dkrw_corey,
                         where=(dkrw_corey > 0), alpha=0.08, color=c_water)
        ax2.fill_between(sw_grid, dkrg_corey, 0,
                         where=(dkrg_corey < 0), alpha=0.08, color=c_gas)
        
        ax2.set_xlabel('含水饱和度 Sw', fontsize=13)
        ax2.set_ylabel('导数 dkr/dSw', fontsize=13)
        ax2.set_title('相渗导数 (Corey-Brooks 解析)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=8.5, loc='upper left', framealpha=0.9,
                   edgecolor=c_grid, fancybox=True)
        ax2.grid(True, alpha=0.4, color=c_grid, linewidth=0.5)
        ax2.tick_params(labelsize=11)
        
        # 公式标注
        formula = (f'Corey-Brooks:\n'
                   f'  krg = {krg_max:.3f}·Se_g^{ng:.1f}\n'
                   f'  krw = {krw_max:.3f}·Se_w^{nw:.1f}')
        ax2.text(0.97, 0.03, formula, transform=ax2.transAxes,
                 fontsize=8, fontfamily='monospace', color=c_annot,
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=c_grid, alpha=0.9))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            filepath = os.path.join(self.fig_dir, 'M3_relperm_curves.png')
            fig.savefig(filepath, dpi=200, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            self.logger.info(f"相渗曲线图已保存: {filepath}")
            return filepath
        
        plt.close(fig)
        return ''
    
    def generate_report(self) -> str:
        """生成相渗验收报告"""
        passed, checks = self.validate_all()
        Swr, Sgr, krw_max, krg_max = self.rp.endpoints()
        
        lines = [
            "# M3 气水相渗验收报告",
            "",
            f"**验收结果: {'✅ 全部通过' if passed else '❌ 存在未通过项'}**",
            "",
            "## 数据来源",
            "",
            "- 附表7-相对渗透率数据表 (SY13井, MK组)",
            f"- 孔隙度: 2.16%, 绝对渗透率: 3.20×10⁻³ μm²",
            f"- 数据点数: {len(self.rp.sw_data)}",
            "",
            "## 端点参数",
            "",
            f"| 参数 | 值 |",
            f"|------|------|",
            f"| 束缚水饱和度 Swr | {Swr:.4f} |",
            f"| 残余气饱和度 Sgr | {Sgr:.4f} |",
            f"| 最大水相相渗 krw_max | {krw_max:.4f} |",
            f"| 最大气相相渗 krg_max | {krg_max:.4f} |",
            "",
            "## 验收检查",
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
            "- PCHIP (分段三次 Hermite, 保持单调性)",
            "- 超范围自动 clamp 到端点",
            "- 支持解析导数 dkr/dSw",
            "",
            "## 图件",
            "",
            "- M3_relperm_curves.png: 气水相渗曲线（含原始数据点 + 导数）",
        ])
        
        report_path = os.path.join(self.output_dir, 'reports', 'M3_relperm_report.md')
        write_markdown_report(lines, report_path)
        self.logger.info(f"相渗验收报告已保存: {report_path}")
        
        return report_path
