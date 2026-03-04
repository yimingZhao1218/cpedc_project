"""
M7-D: 碳足迹全生命周期评估 (LCA)
==================================
v4.7 2026-03-04

三维碳足迹评估:
  A. 计算侧: PINN vs 传统数模 (GPU单卡 vs CPU集群)
  B. 生产侧: 不同策略下CH4泄漏+含水处理能耗
  C. CCUS潜力: MK组CO2封存安全窗口初评

数据来源:
  - 全国电网排放因子: 0.581 kgCO2/kWh (2023年生态环境部公告)
  - GPU TDP: NVIDIA RTX 4090 = 450W
  - CPU集群基准: 双路至强×8节点, 200核×200W/核 (Eclipse/CMG典型配置)
  - CH4泄漏率: 0.1% (天然气生产行业平均, EPA 2023)
  - CH4 GWP100: 28 (IPCC AR5)
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 全局常量
GRID_EMISSION_FACTOR = 0.581     # kgCO2/kWh (2023年生态环境部)
CH4_GWP100 = 28                  # IPCC AR5
CH4_DENSITY_KG_M3 = 0.717       # 标况 kg/m3
CH4_LEAK_RATE = 0.001            # 0.1% 泄漏率

# Corey fw内联计算 (避免循环import nsga2_optimizer)
# v4.7: 参数与附表7拟合值+torch_physics.py TorchRelPerm完全一致
_COREY = dict(Swc=0.26, Sgr=0.062, nw=4.4071, ng=1.0846,
              krw_max=0.48, krg_max=0.675, mu_w=0.30, mu_g=0.025)

def _simple_corey_fw(Sw):
    """Corey含水率 — 内联版，用于碳足迹估算(精度要求不高)"""
    c = _COREY
    Se = np.clip((Sw - c['Swc']) / (1.0 - c['Swc'] - c['Sgr']), 0, 1)
    krw = c['krw_max'] * np.power(Se, c['nw'])
    krg = c['krg_max'] * np.power(1.0 - Se, c['ng'])
    mob_w = krw / c['mu_w']
    mob_g = krg / c['mu_g']
    return mob_w / (mob_w + mob_g + 1e-15)


def compute_computational_carbon(nsga2_results: Optional[dict] = None,
                                  pinn_train_hours: float = 1.0,
                                  gpu_tdp_kw: float = 0.450) -> dict:
    """
    A. 计算侧碳足迹对比

    PINN方案: 训练 + NSGA-II推理
    传统方案: Eclipse/CMG CPU集群建模+求解

    Args:
        nsga2_results: NSGA-II结果(含n_eval, elapsed)
        pinn_train_hours: PINN训练时间(小时)
        gpu_tdp_kw: GPU功率(kW)
    """
    # ── PINN方案 ──
    pinn_train_kwh = pinn_train_hours * gpu_tdp_kw
    nsga2_hours = nsga2_results['elapsed'] / 3600 if nsga2_results else 5.0 / 60
    pinn_optim_kwh = nsga2_hours * gpu_tdp_kw
    pinn_total_kwh = pinn_train_kwh + pinn_optim_kwh
    pinn_co2_kg = pinn_total_kwh * GRID_EMISSION_FACTOR

    # ── 传统方案 (Eclipse/CMG) ──
    # 赛题规模: 碳酸盐岩双孔双渗, 80×80×单层网格
    # 单台高性能工作站: 双路至强16核, 整机功耗~600W, 单案例~2h
    # 单方案: 0.6kW × 2h = 1.2 kWh (非集群, 赛题规模不需要HPC)
    n_scenarios = nsga2_results['n_eval'] if nsga2_results else 3000
    trad_per_case_kwh = 0.6 * 2.0  # v4.7: 120→1.2 kWh (单台工作站600W×2h)
    # 历史拟合: 手动调参约10次迭代
    trad_histmatch_kwh = 10 * trad_per_case_kwh  # 12 kWh
    # 多方案评估
    trad_optim_kwh = n_scenarios * trad_per_case_kwh
    # 建模人工(不计碳, 仅记录工时)
    trad_total_kwh = trad_histmatch_kwh + trad_optim_kwh
    trad_co2_kg = trad_total_kwh * GRID_EMISSION_FACTOR

    reduction_kg = trad_co2_kg - pinn_co2_kg
    reduction_pct = reduction_kg / trad_co2_kg * 100 if trad_co2_kg > 0 else 0

    result = {
        'pinn_train_kwh': pinn_train_kwh,
        'pinn_optim_kwh': pinn_optim_kwh,
        'pinn_total_kwh': pinn_total_kwh,
        'pinn_co2_kg': pinn_co2_kg,
        'trad_histmatch_kwh': trad_histmatch_kwh,
        'trad_optim_kwh': trad_optim_kwh,
        'trad_total_kwh': trad_total_kwh,
        'trad_co2_kg': trad_co2_kg,
        'reduction_kg': reduction_kg,
        'reduction_pct': reduction_pct,
        'n_scenarios': n_scenarios,
    }

    logger.info(
        f"计算侧碳足迹: PINN={pinn_co2_kg:.2f}kgCO2 vs "
        f"传统={trad_co2_kg:.1f}kgCO2, 减排{reduction_kg:.1f}kg ({reduction_pct:.1f}%)"
    )
    return result


def compute_production_carbon(strategies: dict, t_days: np.ndarray) -> dict:
    """
    B. 生产侧碳排放对比

    不同策略下:
      1. CH4泄漏: Gp × 泄漏率 × CH4密度 × GWP
      2. 含水处理能耗: qw × 处理电耗(5 kWh/m3) × 电网排放因子

    Args:
        strategies: {name: {qg, sw, t_days, ...}} 从evaluate_production_strategy
        t_days: 时间序列
    """
    results = {}
    water_treatment_kwh_per_m3 = 5.0  # 含水处理电耗 kWh/m3 (脱水+防腐+排放)

    for name, s in strategies.items():
        qg = s.get('qg', np.zeros_like(t_days))
        dt = np.diff(s.get('t_days', t_days), prepend=0)

        # 累计产气 m3
        Gp_m3 = np.sum(qg * dt)

        # CH4泄漏碳排放
        ch4_leak_m3 = Gp_m3 * CH4_LEAK_RATE
        ch4_leak_kg = ch4_leak_m3 * CH4_DENSITY_KG_M3
        ch4_co2eq_kg = ch4_leak_kg * CH4_GWP100

        # 含水处理碳排放
        sw = s.get('sw', np.full_like(qg, 0.26))
        fw = _simple_corey_fw(sw)
        qw = qg * fw / (1.0 - fw + 1e-10)
        qw = np.clip(qw, 0, qg * 5)
        Qw_m3 = np.sum(qw * dt)
        water_kwh = Qw_m3 * water_treatment_kwh_per_m3
        water_co2_kg = water_kwh * GRID_EMISSION_FACTOR

        total_co2_kg = ch4_co2eq_kg + water_co2_kg

        results[name] = {
            'Gp_M_m3': Gp_m3 / 1e6,
            'ch4_leak_kg': ch4_leak_kg,
            'ch4_co2eq_kg': ch4_co2eq_kg,
            'Qw_M_m3': Qw_m3 / 1e6,
            'water_kwh': water_kwh,
            'water_co2_kg': water_co2_kg,
            'total_co2_kg': total_co2_kg,
        }

        logger.info(
            f"  {name}: CH4泄漏={ch4_co2eq_kg:.0f}kgCO2eq, "
            f"水处理={water_co2_kg:.1f}kgCO2, 合计={total_co2_kg:.0f}kgCO2"
        )

    return results


def estimate_ccus_potential(p_res_MPa: float = 76.0,
                            p_frac_MPa: float = 110.0,
                            pore_volume_m3: float = None,
                            porosity: float = 0.05,
                            area_m2: float = None,
                            thickness_m: float = 90.0) -> dict:
    """
    C. CCUS潜力初评

    基于PINN压力场评估MK组CO2封存安全窗口:
      安全注入压力: p_res < p_inj < 0.9 × p_frac
      可用压力窗口: dp_safe = 0.9×p_frac - p_res
      储存量估算: V_co2 = E × PV × (dp_safe/p_res) × rho_co2

    Args:
        p_res_MPa: 当前地层压力
        p_frac_MPa: 破裂压力 (MK组碳酸盐岩, 根据地应力梯度估算)
        porosity: 平均孔隙度
        area_m2: 储层面积 (若None则用井位凸包估算)
        thickness_m: 储层厚度
    """
    # 储层面积估算 (井位凸包: ~17km × 10km = 170 km2)
    if area_m2 is None:
        area_m2 = 170e6  # 170 km2 = 170×10^6 m2

    if pore_volume_m3 is None:
        pore_volume_m3 = area_m2 * thickness_m * porosity

    # 安全注入压力窗口
    p_safe_max = 0.9 * p_frac_MPa
    dp_safe = p_safe_max - p_res_MPa

    if dp_safe <= 0:
        logger.warning("CCUS: 安全压力窗口不足, 不适合CO2注入")
        return {'feasible': False, 'dp_safe_MPa': dp_safe}

    # CO2密度 @储层条件 (超临界, ~76MPa, 140C)
    # 近似: rho_co2 ≈ 500 kg/m3 (超临界态)
    rho_co2 = 500.0  # kg/m3

    # 储存效率因子 E (碳酸盐岩: 1-4%, 取2%)
    E_factor = 0.02

    # CO2储存量 = E × PV × rho_co2 (简化, 不考虑压力变化)
    co2_storage_kg = E_factor * pore_volume_m3 * rho_co2
    co2_storage_Mt = co2_storage_kg / 1e9  # 百万吨

    result = {
        'feasible': True,
        'p_res_MPa': p_res_MPa,
        'p_frac_MPa': p_frac_MPa,
        'p_safe_max_MPa': p_safe_max,
        'dp_safe_MPa': dp_safe,
        'pore_volume_Mm3': pore_volume_m3 / 1e6,
        'E_factor': E_factor,
        'rho_co2_kg_m3': rho_co2,
        'co2_storage_Mt': co2_storage_Mt,
        'co2_storage_kg': co2_storage_kg,
    }

    logger.info(
        f"CCUS潜力: PV={pore_volume_m3/1e6:.1f}M m3, "
        f"dp_safe={dp_safe:.1f}MPa, "
        f"CO2={co2_storage_Mt:.2f} Mt (E={E_factor*100:.0f}%)"
    )
    return result


def plot_carbon_footprint(comp_carbon, prod_carbon, ccus, save_dir):
    """绘制碳足迹LCA三面板图"""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── (a) 计算侧对比 ──
    ax = axes[0]
    cats = ['PINN\n(GPU)', 'Traditional\n(CPU cluster)']
    vals = [comp_carbon['pinn_co2_kg'], comp_carbon['trad_co2_kg']]
    colors = ['#27AE60', '#E74C3C']
    bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor='#2C3E50', linewidth=0.8)
    for b, v in zip(bars, vals):
        label = f"{v:.2f}" if v < 10 else f"{v:.0f}"
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                f"{label} kgCO2", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('CO2 (kg)', fontsize=11)
    ax.set_title(f'(a) Computational Carbon\nReduction: {comp_carbon["reduction_pct"]:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # ── (b) 生产侧对比 ──
    ax = axes[1]
    names = list(prod_carbon.keys())
    ch4_vals = [prod_carbon[n]['ch4_co2eq_kg'] for n in names]
    wat_vals = [prod_carbon[n]['water_co2_kg'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, ch4_vals, w, label='CH4 leak', color='#E67E22', edgecolor='#2C3E50')
    b2 = ax.bar(x + w/2, wat_vals, w, label='Water treatment', color='#3498DB', edgecolor='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_ylabel('CO2eq (kg)', fontsize=11)
    ax.set_title('(b) Production Carbon by Strategy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── (c) CCUS潜力 ──
    ax = axes[2]
    if ccus.get('feasible', False):
        labels = ['Reservoir\nPressure', 'Safe Max\nInjection', 'Fracture\nPressure']
        pressures = [ccus['p_res_MPa'], ccus['p_safe_max_MPa'], ccus['p_frac_MPa']]
        colors_c = ['#3498DB', '#27AE60', '#E74C3C']
        bars = ax.barh(labels, pressures, color=colors_c, height=0.5, edgecolor='#2C3E50')
        for b, v in zip(bars, pressures):
            ax.text(v + 1, b.get_y() + b.get_height()/2,
                    f"{v:.1f} MPa", va='center', fontsize=10, fontweight='bold')
        ax.axvline(ccus['p_safe_max_MPa'], color='green', ls='--', alpha=0.5)
        ax.set_xlabel('Pressure (MPa)', fontsize=11)
        storage_txt = f"CO2 Storage: {ccus['co2_storage_Mt']:.2f} Mt\n(E={ccus['E_factor']*100:.0f}%)"
        ax.text(0.95, 0.15, storage_txt, transform=ax.transAxes, fontsize=11,
                ha='right', fontweight='bold', color='#27AE60',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    else:
        ax.text(0.5, 0.5, 'CCUS Not Feasible\n(Pressure window insufficient)',
                transform=ax.transAxes, ha='center', va='center', fontsize=14, color='red')
    ax.set_title('(c) CCUS Potential (MK Formation)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'M7_carbon_footprint_lca.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Carbon footprint figure: {path}")


def generate_carbon_report(comp, prod, ccus):
    """生成碳足迹markdown报告"""
    lines = [
        "## Carbon Footprint LCA (v4.7)",
        "",
        "### A. Computational Carbon",
        "",
        f"| Item | PINN | Traditional | Reduction |",
        f"|------|------|------------|-----------|",
        f"| Energy (kWh) | {comp['pinn_total_kwh']:.2f} | {comp['trad_total_kwh']:,.0f} | {comp['reduction_pct']:.1f}% |",
        f"| CO2 (kg) | {comp['pinn_co2_kg']:.2f} | {comp['trad_co2_kg']:,.0f} | {comp['reduction_kg']:,.0f} kg |",
        "",
        "### B. Production Carbon by Strategy",
        "",
        "| Strategy | CH4 leak (kgCO2eq) | Water treat (kgCO2) | Total |",
        "|----------|-------------------|--------------------|----- |",
    ]
    for n, p in prod.items():
        lines.append(f"| {n} | {p['ch4_co2eq_kg']:.0f} | {p['water_co2_kg']:.1f} | {p['total_co2_kg']:.0f} |")

    if ccus.get('feasible'):
        lines.extend([
            "",
            "### C. CCUS Potential",
            "",
            f"- Safe injection window: {ccus['dp_safe_MPa']:.1f} MPa",
            f"- CO2 storage capacity: **{ccus['co2_storage_Mt']:.2f} Mt**",
            f"- Storage efficiency: {ccus['E_factor']*100:.0f}%",
        ])

    return "\n".join(lines)
