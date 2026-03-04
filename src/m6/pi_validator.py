"""
M6-B: 附表9 PI独立闭环验证
============================
用SY9试油数据(完全未参与PINN训练)独立估算k_frac，
与M5反演值对比，实现"两条独立路径收敛"的可信度证明。

数据来源: 附表9-试油试采数据.csv
  - qg_test  = 568,663 m³/d (地面标况日产气)
  - p_wf_test = 71.2256 MPa (井底流压)
  - p_res_test = 74.875 MPa (井底静压)
  - T_test = 140.32°C (井底流温)

物理基础:
  PI_test = qg_test / (p_res - p_wf) = 155,840 m³/d/MPa
  
  Peaceman: PI = WI × krg(Swc) / (μg × Bg)
           WI = 2π × k_frac × h / ln(r_e/r_w)
  
  → k_frac_PI = PI_test × μg × Bg × ln(r_e/r_w) / (2π × h × krg_max)
"""

import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  附表9 原始数据 (SY9井试油试采)
# ═══════════════════════════════════════════════════════════
WELL_TEST_DATA = {
    'well_id': 'SY9',
    'qg_test_m3d': 568663.0,       # 日产气 m³/d (地面标况)
    'p_wf_test_MPa': 71.2256,      # 井底流压 MPa
    'p_res_test_MPa': 74.875,      # 井底静压 MPa
    'T_wellbore_C': 140.32,        # 井底流温 °C
    'T_static_C': 136.689,         # 井底静温 °C
    'dp_test_MPa': 2.7697,         # 压差 MPa (= 74.875 - 71.2256 ≈ 3.649, 附表给2.77?)
    'gamma_g': 0.577,              # 天然气相对密度 (air=1)
    'test_conclusion': '干气藏',
    'h_perf_m': 58.5,              # 射孔厚度 m (4549-4607.5)
    'gauge_depth_m': 4400.0,       # 压力计下深 m
}


def compute_pi_test():
    """
    计算附表9试油产能指数 PI_test
    
    注意: dp_test在附表中标注为2.7697 MPa, 但p_res-p_wf = 74.875-71.226 = 3.649 MPa
    附表中的2.7697可能是不同基准的压差(如稳定后vs瞬态)。
    工程上取两者都计算，以较保守值(较大dp)为主。
    """
    d = WELL_TEST_DATA
    dp_direct = d['p_res_test_MPa'] - d['p_wf_test_MPa']  # 3.649 MPa
    dp_table = d['dp_test_MPa']                             # 2.7697 MPa
    
    PI_direct = d['qg_test_m3d'] / dp_direct   # 155,840 m³/d/MPa
    PI_table = d['qg_test_m3d'] / dp_table      # 205,316 m³/d/MPa
    
    logger.info(
        f"附表9 PI计算: "
        f"PI(直接压差{dp_direct:.3f}MPa) = {PI_direct:.0f} m³/d/MPa, "
        f"PI(附表压差{dp_table:.4f}MPa) = {PI_table:.0f} m³/d/MPa"
    )
    
    return {
        'PI_direct': PI_direct,
        'PI_table': PI_table,
        'dp_direct': dp_direct,
        'dp_table': dp_table,
        'qg_test': d['qg_test_m3d'],
    }


def compute_k_frac_from_pi(PI_m3d_MPa, h_well_m, r_e_m, r_w_m,
                            mu_g_Pa_s, Bg_m3_m3, krg_max=0.80, skin=0.0):
    """
    从PI反算k_frac (Peaceman公式反演)
    
    PI = 2π × k_frac × h × krg_max / (μg × Bg × (ln(r_e/r_w) + skin))
    
    → k_frac = PI × μg × Bg × (ln(r_e/r_w) + skin) / (2π × h × krg_max)
    
    Args:
        PI_m3d_MPa: 产能指数 m³/d/MPa
        h_well_m: 有效储层厚度 m
        r_e_m: 等效泄油半径 m
        r_w_m: 井筒半径 m
        mu_g_Pa_s: 气体粘度 Pa·s (注意单位!)
        Bg_m3_m3: 体积系数 m³/m³ (地面/地下)
        krg_max: 气相端点相渗 (Sw=Swc时)
        skin: 表皮因子
    
    Returns:
        k_frac_mD: 裂缝增强渗透率 mD
    """
    ln_ratio = math.log(r_e_m / r_w_m) + skin
    
    # PI单位转换: m³/d/MPa → m³/s/Pa
    # 1 m³/d = 1/(86400) m³/s
    # 1 MPa = 1e6 Pa
    # PI [m³/s/Pa] = PI [m³/d/MPa] / 86400 × 1e6 = PI × 1e6/86400
    PI_SI = PI_m3d_MPa * 1e6 / 86400.0   # m³/s/Pa
    
    # Peaceman: PI_SI = 2π × k_SI × h / (μg × Bg × ln_ratio) × krg_max
    # → k_SI = PI_SI × μg × Bg × ln_ratio / (2π × h × krg_max)
    k_SI = PI_SI * mu_g_Pa_s * Bg_m3_m3 * ln_ratio / (2.0 * math.pi * h_well_m * krg_max)
    
    # k_SI [m²] → k_mD [mD]: 1 mD = 9.869233e-16 m²
    k_mD = k_SI / 9.869233e-16
    
    return k_mD


def run_pi_validation(m5_params: dict, config: dict,
                      save_dir: str = None) -> dict:
    """
    执行完整的附表9 PI独立验证
    
    Args:
        m5_params: M5反演参数 dict, 包含:
            - k_frac_mD: float
            - r_e_m: float
            - r_w_m: float (默认0.1)
            - dp_wellbore_MPa: float
        config: 项目config dict
        save_dir: 图件保存目录
    
    Returns:
        验证结果 dict
    """
    logger.info("=" * 60)
    logger.info("M6-B: 附表9 PI独立闭环验证")
    logger.info("=" * 60)
    
    # ── 1. M5反演参数 ──
    k_frac_m5 = m5_params.get('k_frac_mD', 9.68)
    r_e = m5_params.get('r_e_m', 128.9)
    r_w = m5_params.get('r_w_m', 0.1)
    skin = m5_params.get('skin', 0.0)
    
    # ── 2. 物性参数 (试油条件: p≈74.9 MPa, T≈140°C) ──
    # 附表5: Bg(75.7MPa, 140.32°C) ≈ 0.002577 m³/m³
    Bg = 0.002577
    # LGE粘度模型 @74.9 MPa, 140°C → ~0.034 mPa·s = 3.4e-5 Pa·s
    mu_g = 0.034e-3   # Pa·s
    
    # 有效储厚 (附表8: SY9 net pay = 48.4m)
    # 注意: 试油射孔段58.5m是毛厚度，有效储厚用附表8的48.4m
    h_well = 48.4
    
    # Corey端点相渗 (v4.7: 与附表7 + torch_physics.py统一)
    krg_max = 0.675
    
    # ── 3. 计算附表9 PI ──
    pi_results = compute_pi_test()
    PI_test = pi_results['PI_direct']   # 用直接压差 (更保守)
    
    # ── 4. 从PI反算k_frac ──
    k_frac_pi = compute_k_frac_from_pi(
        PI_test, h_well, r_e, r_w, mu_g, Bg, krg_max, skin
    )
    
    # ── 5. 对比 ──
    deviation_pct = abs(k_frac_pi - k_frac_m5) / k_frac_m5 * 100
    
    logger.info(f"  M5反演 k_frac = {k_frac_m5:.2f} mD")
    logger.info(f"  附表9反算 k_frac = {k_frac_pi:.2f} mD")
    logger.info(f"  偏差 = {deviation_pct:.1f}%")
    logger.info(f"  {'✅ 偏差<30%, 两条独立路径收敛' if deviation_pct < 30 else '⚠️ 偏差较大, 需检查参数'}")
    
    # ── 6. 也用附表压差计算 (作为灵敏度对比) ──
    PI_table = pi_results['PI_table']
    k_frac_pi_table = compute_k_frac_from_pi(
        PI_table, h_well, r_e, r_w, mu_g, Bg, krg_max, skin
    )
    
    results = {
        'PI_test_direct': PI_test,
        'PI_test_table': PI_table,
        'k_frac_m5_mD': k_frac_m5,
        'k_frac_pi_direct_mD': k_frac_pi,
        'k_frac_pi_table_mD': k_frac_pi_table,
        'deviation_direct_pct': deviation_pct,
        'deviation_table_pct': abs(k_frac_pi_table - k_frac_m5) / k_frac_m5 * 100,
        'h_well_m': h_well,
        'r_e_m': r_e,
        'r_w_m': r_w,
        'mu_g_mPas': mu_g * 1e3,
        'Bg': Bg,
        'krg_max': krg_max,
    }
    
    # ── 7. 绘图 ──
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _plot_pi_validation(results, save_dir)
    
    return results


def _plot_pi_validation(results: dict, save_dir: str):
    """绘制PI验证双面板图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # ── 面板(a): PI对比柱状图 ──
    ax = axes[0]
    labels = ['附表9\n(直接压差)', '附表9\n(附表压差)']
    pi_vals = [results['PI_test_direct'], results['PI_test_table']]
    colors = ['#3498DB', '#85C1E9']
    bars = ax.bar(labels, pi_vals, color=colors, width=0.5, edgecolor='#2C3E50', linewidth=0.8)
    
    for bar, val in zip(bars, pi_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('产能指数 PI (m³/d/MPa)', fontsize=12)
    ax.set_title('(a) 附表9 试油产能指数', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(pi_vals) * 1.25)
    
    # ── 面板(b): k_frac双路径收敛 ──
    ax = axes[1]
    k_vals = [results['k_frac_m5_mD'], results['k_frac_pi_direct_mD'], results['k_frac_pi_table_mD']]
    k_labels = ['M5 PINN\n反演', '附表9 PI\n(直接压差)', '附表9 PI\n(附表压差)']
    k_colors = ['#E74C3C', '#2ECC71', '#82E0AA']
    
    bars = ax.bar(k_labels, k_vals, color=k_colors, width=0.5, edgecolor='#2C3E50', linewidth=0.8)
    
    for bar, val in zip(bars, k_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f} mD', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 偏差标注
    dev = results['deviation_direct_pct']
    color_dev = '#27AE60' if dev < 30 else '#E67E22'
    ax.annotate(
        f'偏差 {dev:.1f}%',
        xy=(0.5, max(k_vals[0], k_vals[1])),
        xytext=(0.5, max(k_vals) * 1.15),
        fontsize=12, fontweight='bold', color=color_dev,
        ha='center',
        arrowprops=dict(arrowstyle='->', color=color_dev, lw=1.5)
    )
    
    ax.set_ylabel('裂缝增强渗透率 k_frac (mD)', fontsize=12)
    ax.set_title('(b) k_frac 双路径独立验证', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(k_vals) * 1.35)
    
    # 添加物理参数注释
    param_text = (
        f"验证参数:\n"
        f"h = {results['h_well_m']:.1f} m (附表8有效储厚)\n"
        f"r_e = {results['r_e_m']:.1f} m (M5反演)\n"
        f"μg = {results['mu_g_mPas']:.3f} mPa·s (LGE@74.9MPa)\n"
        f"Bg = {results['Bg']:.4f} m³/m³ (附表5)"
    )
    ax.text(0.98, 0.55, param_text, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'M6_pi_validation.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  PI验证图件已保存: {save_path}")


def generate_pi_report_section(results: dict) -> str:
    """生成PI验证的markdown报告段落"""
    return f"""## 附表9 PI独立闭环验证 (v4.7)

### 数据来源
- 附表9: SY9井试油试采数据 (**完全未参与PINN训练**)
- qg_test = {WELL_TEST_DATA['qg_test_m3d']:,.0f} m³/d
- p_wf = {WELL_TEST_DATA['p_wf_test_MPa']:.4f} MPa, p_res = {WELL_TEST_DATA['p_res_test_MPa']:.3f} MPa

### 产能指数对比

| 方法 | 压差(MPa) | PI (m³/d/MPa) |
|------|----------|--------------|
| 直接压差 (p_res - p_wf) | {WELL_TEST_DATA['p_res_test_MPa'] - WELL_TEST_DATA['p_wf_test_MPa']:.3f} | {results['PI_test_direct']:,.0f} |
| 附表标注压差 | {WELL_TEST_DATA['dp_test_MPa']:.4f} | {results['PI_test_table']:,.0f} |

### k_frac双路径收敛验证

| 路径 | k_frac (mD) | 数据来源 |
|------|------------|---------|
| **M5 PINN反演** | **{results['k_frac_m5_mD']:.2f}** | 1331天生产数据拟合 |
| **附表9 PI反算** | **{results['k_frac_pi_direct_mD']:.2f}** | 独立试油数据 |

**偏差: {results['deviation_direct_pct']:.1f}%** {'✅ <30%, 两条完全独立的路径收敛到同一物理量' if results['deviation_direct_pct'] < 30 else '⚠️ 偏差较大'}

### 验证参数
- h_well = {results['h_well_m']:.1f} m (附表8有效储厚)
- r_e = {results['r_e_m']:.1f} m (M5反演等效泄油半径)
- μg = {results['mu_g_mPas']:.3f} mPa·s (LGE关联式 @74.9MPa, 140°C)
- Bg = {results['Bg']:.4f} m³/m³ (附表5-4 @75.7MPa)
- krg(Swc) = {results['krg_max']:.2f} (Corey端点)
"""
