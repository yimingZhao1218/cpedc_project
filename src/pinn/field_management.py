"""
M7-C: 全场7井差异化智能管控
==============================
v4.7 2026-03-04

基于M6 WIRI风险排名 + NSGA-II优化结果, 为7口井生成差异化管控方案。

管控分类:
  立即干预: WIRI>0.7              → 排水采气/关井控水
  重点监控: WIRI 0.4-0.7          → 阶梯降产+季度监测
  计划跟进: WIRI 0.2-0.4          → NSGA-II平衡区方案
  常规生产: WIRI<0.2              → 维持稳产+年度监测

数据来源:
  - WIRI: M6连通性报告 (构造40%+连通性30%+Sw30%)
  - NSGA-II TOP-3: M7-A多目标优化结果
  - 各井属性: 附表8测井解释, M6渗透率场
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# WIRI排名 (来自M6 connectivity报告 v3.19)
DEFAULT_WIRI = {
    'SYX211': {'wiri': 1.000, 'risk': 'high',   'gwc_dist_m': -32,  'note': 'gas-water well'},
    'SY102':  {'wiri': 0.568, 'risk': 'medium',  'gwc_dist_m': 20,   'note': 'gas-water well'},
    'SY116':  {'wiri': 0.434, 'risk': 'medium',  'gwc_dist_m': -11,  'note': 'below GWC'},
    'SY13':   {'wiri': 0.357, 'risk': 'low',     'gwc_dist_m': 15,   'note': ''},
    'SY101':  {'wiri': 0.263, 'risk': 'low',     'gwc_dist_m': 24,   'note': ''},
    'SY9':    {'wiri': 0.241, 'risk': 'low',     'gwc_dist_m': 74,   'note': 'highest structure'},
    'SY201':  {'wiri': 0.150, 'risk': 'low',     'gwc_dist_m': 61,   'note': ''},
}


def classify_wells(wiri_data: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    基于WIRI将7口井分为4个管控类别
    
    Returns:
        {well_id: {wiri, category, action, monitoring, color}}
    """
    if wiri_data is None:
        wiri_data = DEFAULT_WIRI
    
    results = {}
    for well_id, info in wiri_data.items():
        w = info['wiri']
        
        if w >= 0.7:
            cat = 'immediate'
            action = 'drainage gas recovery / shut-in water control'
            monitor = 'daily'
            color = '#E74C3C'
        elif w >= 0.4:
            cat = 'priority'
            action = 'stepped decline (10%/quarter)'
            monitor = 'quarterly'
            color = '#E67E22'
        elif w >= 0.2:
            cat = 'planned'
            action = 'NSGA-II balanced strategy'
            monitor = 'semi-annual'
            color = '#F1C40F'
        else:
            cat = 'normal'
            action = 'maintain production'
            monitor = 'annual'
            color = '#27AE60'
        
        results[well_id] = {
            'wiri': w,
            'category': cat,
            'action': action,
            'monitoring': monitor,
            'color': color,
            'gwc_dist_m': info.get('gwc_dist_m', 0),
            'note': info.get('note', ''),
        }
    
    return results


def estimate_field_npv(well_plans: Dict, nsga2_top3: Optional[List] = None) -> Dict:
    """
    估算全场NPV (各井差异化 vs 全部稳产)
    
    简化模型: 各井按管控类别分配不同的产量衰减系数
    """
    # 各类别的产量保留系数 (相对稳产)
    retention = {
        'immediate': 0.30,   # 排水采气, 产量大幅下降
        'priority': 0.70,    # 阶梯降产
        'planned': 0.85,     # NSGA-II平衡
        'normal': 1.00,      # 稳产
    }
    
    # 各井基础日产能估算 (基于附表10 SY9数据外推, 简化)
    # SY9是主力井(~500k m3/d), 其他井按渗透率比例估算
    base_production = {
        'SY9': 500000, 'SY13': 50000, 'SY201': 80000,
        'SY101': 60000, 'SY102': 40000, 'SY116': 30000,
        'SYX211': 20000,
    }
    
    gas_price = 2.50  # yuan/m3
    years = 5
    days = years * 365
    
    # 全部稳产 NPV
    npv_all_steady = sum(q * days * gas_price for q in base_production.values()) / 1e6
    
    # 差异化管控 NPV
    npv_diff = 0
    for well_id, plan in well_plans.items():
        q_base = base_production.get(well_id, 50000)
        ret = retention.get(plan['category'], 0.85)
        npv_diff += q_base * ret * days * gas_price / 1e6
    
    return {
        'npv_all_steady_M': npv_all_steady,
        'npv_differentiated_M': npv_diff,
        'delta_M': npv_diff - npv_all_steady,
        'years': years,
    }


def plot_field_management(well_plans: Dict, npv_est: Dict, save_path: str):
    """绘制全场管控方案图 (2x1布局)"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ── (a) 7井管控优先级矩阵 ──
    ax = axes[0]
    
    wells = sorted(well_plans.keys(), key=lambda w: well_plans[w]['wiri'], reverse=True)
    y_pos = np.arange(len(wells))
    wiri_vals = [well_plans[w]['wiri'] for w in wells]
    colors = [well_plans[w]['color'] for w in wells]
    
    bars = ax.barh(y_pos, wiri_vals, color=colors, height=0.6, 
                   edgecolor='#2C3E50', linewidth=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wells, fontsize=11)
    ax.set_xlabel('WIRI Risk Index', fontsize=11)
    ax.set_title('(a) Well Priority by WIRI', fontsize=12, fontweight='bold')
    
    # 分类线
    ax.axvline(0.7, color='red', ls='--', alpha=0.5, label='Immediate (>0.7)')
    ax.axvline(0.4, color='orange', ls='--', alpha=0.5, label='Priority (>0.4)')
    ax.axvline(0.2, color='gold', ls='--', alpha=0.5, label='Planned (>0.2)')
    
    # 动作标注
    for i, w in enumerate(wells):
        plan = well_plans[w]
        action_short = plan['action'][:25]
        ax.text(max(plan['wiri'] + 0.02, 0.05), i, action_short,
                va='center', fontsize=8, style='italic')
    
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 1.15)
    ax.grid(axis='x', alpha=0.3)
    
    # ── (b) 管控策略汇总表 ──
    ax = axes[1]
    ax.axis('off')
    
    cols = ['Well', 'WIRI', 'Category', 'Action', 'Monitor']
    rows = []
    for w in wells:
        p = well_plans[w]
        cat_cn = {'immediate': 'Immediate', 'priority': 'Priority',
                  'planned': 'Planned', 'normal': 'Normal'}[p['category']]
        rows.append([w, f"{p['wiri']:.3f}", cat_cn, 
                     p['action'][:30], p['monitoring']])
    
    tb = ax.table(cellText=rows, colLabels=cols, loc='center',
                  cellLoc='center', colColours=['#D5E8D4'] * 5)
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1.0, 1.5)
    
    # 颜色编码 (转RGBA元组, 兼容所有matplotlib版本)
    from matplotlib.colors import to_rgba
    for i, w in enumerate(wells):
        c = well_plans[w]['color']
        rgba = to_rgba(c, alpha=0.2)
        for j in range(5):
            tb[i + 1, j].set_facecolor(rgba)
    
    ax.set_title('(b) Differentiated Management Plan', 
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Field management figure: {save_path}")


def generate_management_report(well_plans: Dict, npv_est: Dict) -> str:
    """生成7井管控markdown报告段落"""
    wells = sorted(well_plans.keys(), key=lambda w: well_plans[w]['wiri'], reverse=True)
    
    lines = [
        "## 7-Well Differentiated Management (v4.7)",
        "",
        "Based on M6 WIRI risk ranking + NSGA-II optimization results:",
        "",
        "| Well | WIRI | Category | Action | Monitoring |",
        "|------|------|----------|--------|-----------|",
    ]
    
    cat_cn = {'immediate': 'Immediate', 'priority': 'Priority',
              'planned': 'Planned', 'normal': 'Normal'}
    
    for w in wells:
        p = well_plans[w]
        lines.append(
            f"| {w} | {p['wiri']:.3f} | {cat_cn[p['category']]} "
            f"| {p['action']} | {p['monitoring']} |"
        )
    
    lines.extend([
        "",
        "### Key Recommendations",
        "",
        "- **SYX211**: Immediate drainage gas recovery (WIRI=1.000, confirmed water invasion)",
        "- **SY102/SY116**: Priority monitoring + stepped decline",  
        "- **SY9/SY13**: Apply NSGA-II balanced strategy",
        "- **SY201/SY101**: Maintain current production with routine monitoring",
    ])
    
    return "\n".join(lines)
