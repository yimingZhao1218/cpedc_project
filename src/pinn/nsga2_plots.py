"""
NSGA-II 可视化模块
===================
绘制Pareto前沿2x2面板图 + 生成报告markdown段落
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_pareto_results(results, save_dir):
    """绘制NSGA-II Pareto结果: 2x2面板"""
    os.makedirs(save_dir, exist_ok=True)

    Gp = results['Gp_M']
    Sw = results['Sw_end']
    NPV = results['NPV_M']
    top3 = results['top3']
    existing = results['existing_strategies']

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # (a) Gp vs Sw_end
    ax = axes[0, 0]
    sc = ax.scatter(Gp, Sw, c=NPV, cmap='RdYlGn', s=20, alpha=0.6, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='NPV (M yuan)', shrink=0.8)
    _mark_points(ax, top3, existing, 'Gp_M', 'Sw_end')
    ax.set_xlabel('Gp (M m3)', fontsize=11)
    ax.set_ylabel('Sw_end', fontsize=11)
    ax.set_title('(a) Pareto: Gp vs Sw', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # (b) Gp vs NPV
    ax = axes[0, 1]
    sc = ax.scatter(Gp, NPV, c=Sw, cmap='RdYlBu_r', s=20, alpha=0.6, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Sw_end', shrink=0.8)
    _mark_points(ax, top3, existing, 'Gp_M', 'NPV_M')
    ax.set_xlabel('Gp (M m3)', fontsize=11)
    ax.set_ylabel('NPV (M yuan)', fontsize=11)
    ax.set_title('(b) Pareto: Gp vs NPV', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) TOP-3 vs existing table
    ax = axes[1, 0]
    ax.axis('off')
    cols = ['Strategy', 'dp1(MPa)', 'dp2(MPa)', 'Gp(M m3)', 'Sw_end', 'NPV(M yuan)']
    rows = []
    for t in top3:
        rows.append([f"NSGA-II {t['label']}", f"{t['dp1']:.1f}", f"{t['dp2']:.1f}",
                     f"{t['Gp_M']:.1f}", f"{t['Sw_end']:.3f}", f"{t['NPV_M']:.1f}"])
    for nm, s in existing.items():
        rows.append([f"Manual {nm}", '-', '-',
                     f"{s['Gp_M']:.1f}", f"{s['Sw_end']:.3f}", f"{s['NPV_M']:.1f}"])

    tb = ax.table(cellText=rows, colLabels=cols, loc='center',
                  cellLoc='center', colColours=['#D5E8D4'] * 6)
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1.0, 1.6)
    for i in range(len(top3)):
        for j in range(len(cols)):
            tb[i + 1, j].set_facecolor('#E8F5E9')
    ax.set_title('(c) NSGA-II TOP-3 vs Manual', fontsize=12, fontweight='bold', pad=20)

    # (d) efficiency stats
    ax = axes[1, 1]
    ax.axis('off')
    el = results['elapsed']
    ne = results['n_eval']
    trad_h = ne * 3.0
    speedup = trad_h * 3600 / max(el, 0.01)

    txt = (
        f"NSGA-II Optimization Stats\n"
        f"{'=' * 34}\n"
        f"Evaluations:  {ne:,d}\n"
        f"PINN time:    {el:.1f} s\n"
        f"Traditional:  {trad_h:,.0f} h (3h/case)\n"
        f"Speedup:      {speedup:,.0f}x\n"
        f"{'=' * 34}\n"
        f"Pareto sols:  {len(Gp)}\n"
        f"Dec. space:   4D continuous\n"
        f"Obj. space:   3D (Gp/Sw/NPV)\n"
    )
    ax.text(0.1, 0.9, txt, transform=ax.transAxes, fontsize=11,
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    ax.set_title('(d) Computational Efficiency', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(save_dir, 'M7_nsga2_pareto.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Pareto figure saved: {path}")


def _mark_points(ax, top3, existing, xk, yk):
    mk = ['*', 'D', 's']
    cl = ['#C0392B', '#2980B9', '#16A085']
    for i, t in enumerate(top3):
        ax.scatter(t[xk], t[yk], marker=mk[i], s=200, c=cl[i],
                   edgecolors='black', linewidths=1.2, zorder=5,
                   label=f"TOP-{i+1}: {t['label']}")
    for nm, s in existing.items():
        ax.scatter(s[xk], s[yk], marker='X', s=120, c=s['color'],
                   edgecolors='black', linewidths=0.8, zorder=4,
                   label=f"Manual: {nm}")
    ax.legend(fontsize=8, loc='best', framealpha=0.8)


def generate_nsga2_report(results):
    """生成NSGA-II markdown报告段落"""
    top3 = results['top3']
    ex = results['existing_strategies']
    el = results['elapsed']
    ne = results['n_eval']

    lines = [
        "## NSGA-II Multi-Objective Optimization (v4.7)",
        "",
        f"- Algorithm: NSGA-II (pop=100, {ne} evaluations)",
        "- Surrogate: M5 PINN (cached, ~0.1ms/eval)",
        "- Variables: 4D (dp_stage1, dp_stage2, t_switch, ramp_days)",
        "- Objectives: max Gp / min Sw_end / max NPV",
        f"- Total time: **{el:.1f}s** (vs traditional {ne*3:,d}h)",
        "",
        "### Pareto TOP-3",
        "",
        "| Strategy | dp1 | dp2 | Gp(M) | Sw | NPV(M) |",
        "|----------|-----|-----|-------|------|--------|",
    ]
    for t in top3:
        lines.append(
            f"| NSGA-II {t['label']} | {t['dp1']:.1f} | {t['dp2']:.1f} "
            f"| {t['Gp_M']:.1f} | {t['Sw_end']:.3f} | {t['NPV_M']:.1f} |"
        )
    lines.append("")
    lines.append("### vs Manual Strategies")
    lines.append("")
    lines.append("| Strategy | Gp(M) | Sw | NPV(M) |")
    lines.append("|----------|-------|------|--------|")
    for nm, s in ex.items():
        lines.append(f"| {nm} | {s['Gp_M']:.1f} | {s['Sw_end']:.3f} | {s['NPV_M']:.1f} |")

    return "\n".join(lines)
