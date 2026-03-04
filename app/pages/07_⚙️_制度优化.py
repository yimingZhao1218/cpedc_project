"""
M7 制度优化页面 (v3.17)
=======================
2×2 叠加面板策略对比 + 决策支持
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import re

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'app'))

from components.config_loader import get_loader
from PIL import Image

st.set_page_config(page_title="制度优化", page_icon="⚙️", layout="wide")

st.title("⚙️ M7 制度优化与决策推荐系统")
st.markdown("**PINN秒级策略筛选 | 仅外推区施加策略 | Peaceman物理驱动**")
st.markdown("---")

loader = get_loader()

experiments = loader.list_experiments()
selected_exp = st.selectbox("选择实验", experiments,
                            index=experiments.index('mk_pinn_dt_v2')
                            if 'mk_pinn_dt_v2' in experiments else 0) if experiments else 'mk_pinn_dt_v2'

# ── 从报告自动读取策略数据 ──
strategy_data = {}
try:
    report_path = project_root / 'outputs' / selected_exp / 'reports' / 'M7_water_invasion_report.md'
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = f.read()
        section_key = '## 4. 三策略定量对比'
        if section_key in report:
            section = report.split(section_key)[1].split('\n## ')[0]
            for line in section.strip().split('\n'):
                line = line.strip()
                if not line.startswith('|') or line.startswith('|--') or '策略' in line:
                    continue
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if len(cells) >= 4:
                    name = cells[0]
                    try:
                        gp = float(cells[1])
                    except:
                        gp = 0
                    try:
                        sw = float(cells[2])
                    except:
                        sw = 0
                    strategy_data[name] = {'Gp_M': gp, 'Sw_end': sw, 'dsw': cells[3]}
except Exception:
    pass

st.markdown("---")

# ── 三种策略卡片 ──
st.subheader("📋 三种策略对比")

gp_steady = strategy_data.get('稳产方案', {}).get('Gp_M', 1196)
gp_decay = strategy_data.get('阶梯降产', {}).get('Gp_M', 912)
gp_ctrl = strategy_data.get('控压方案', {}).get('Gp_M', 954)
sw_steady = strategy_data.get('稳产方案', {}).get('Sw_end', 0.443)
sw_decay = strategy_data.get('阶梯降产', {}).get('Sw_end', 0.321)
sw_ctrl = strategy_data.get('控压方案', {}).get('Sw_end', 0.332)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔴 稳产方案")
    st.markdown("""
    **策略描述**:
    - 维持当前产量不变 (基线)
    - p_wf 按 PINN 预测自然演化
    
    **优点**: 累计产气最大
    
    **缺点**: 水侵风险最高, Sw超危险线
    
    **适用**: 短期评价
    """)
    
    st.metric("累计产气", f"{gp_steady:.0f}M m³", delta="基准")
    st.metric("末期 Sw", f"{sw_steady:.3f}", delta=f"+{sw_steady-0.26:.3f}", delta_color="inverse")

with col2:
    st.markdown("### 🟠 阶梯降产 ⭐")
    st.markdown("""
    **策略描述 (仅外推区生效)**:
    - 中期: p_wf +1.5 MPa
    - 后期: p_wf +3.0 MPa
    
    **优点**: 产量与水侵最优平衡
    
    **特点**: Sw显著降低, 推荐方案
    
    **推荐指数**: ⭐⭐⭐⭐⭐
    """)
    
    pct_decay = (1 - gp_decay / gp_steady) * 100 if gp_steady > 0 else 0
    st.metric("累计产气", f"{gp_decay:.0f}M m³", delta=f"-{pct_decay:.0f}% vs 稳产", delta_color="off")
    st.metric("末期 Sw", f"{sw_decay:.3f}", delta=f"{sw_decay-sw_steady:+.3f} vs 稳产", delta_color="normal")

with col3:
    st.markdown("### 🟢 控压方案")
    st.markdown("""
    **策略描述 (仅外推区生效)**:
    - 渐进提压 0 → 4 MPa
    - 平滑保守策略
    
    **优点**: 水侵抑制最强
    
    **缺点**: 产量降幅较大
    
    **适用**: 保守开发
    """)
    
    pct_ctrl = (1 - gp_ctrl / gp_steady) * 100 if gp_steady > 0 else 0
    st.metric("累计产气", f"{gp_ctrl:.0f}M m³", delta=f"-{pct_ctrl:.0f}% vs 稳产", delta_color="off")
    st.metric("末期 Sw", f"{sw_ctrl:.3f}", delta=f"{sw_ctrl-sw_steady:+.3f} vs 稳产", delta_color="normal")

st.markdown("---")

# ── 策略对比图 (2×2 叠加面板) ──
st.subheader("📊 策略对比 (2×2 叠加面板)")

fig_path = loader.get_figure_path(selected_exp, 'M7_strategy_comparison.png')
if fig_path and fig_path.exists():
    img = Image.open(fig_path)
    st.image(img, caption='制度优化 2×2 叠加面板 — PINN秒级策略筛选', 
            use_column_width=True)
    
    with st.expander("📖 图表解读指南"):
        st.markdown("""
        #### 面板结构
        
        - **(a) 日产气量 qg(t)**: 三策略叠加, 历史实线+外推虚线, 灰色底色标注外推区
        - **(b) 累计产气 Gp(t)**: 三策略叠加, 末端标注最终Gp值(百万m³)
        - **(c) 含水饱和度 Sw(t)**: 三策略叠加, 含预警线(0.35)和危险线(0.50)
        - **(d) ΔSw差异放大**: 仅外推区, 阶梯降产/控压 vs 稳产的Sw差值, 量化延缓效果
        
        #### 关键观察
        
        1. **历史区(70%)**: 三策略完全重合 — 策略仅在外推区生效
        2. **外推区(30%)**: 策略分叉, 稳产Sw最高(红), 阶梯降产/控压Sw更低
        3. **ΔSw面板**: 负值=延缓水侵, 面积越大效果越好
        4. **策略验证**: 稳产Gp > 阶梯/控压Gp (物理合理)
        """)
else:
    st.warning("策略对比图未生成, 请运行 `python scripts/regen_m7_water_invasion.py`")

st.markdown("---")

# ── 决策推荐 ──
st.subheader("🎯 决策推荐")

col1, col2 = st.columns([2, 1])

with col1:
    decision_df = pd.DataFrame({
        '开发目标': ['产量最大化', '产量与水侵平衡', '延长无水采气期', '探井短期评价'],
        '推荐策略': ['稳产方案', '阶梯降产 ⭐', '控压方案', '稳产方案'],
        '概率分位': ['P90', 'P50（推荐）', 'P10', 'P90'],
        '适用场景': ['短期开发（1-2年）', '长期稳产（3-5年）', '边水强压（保守）', '经济评价'],
        '风险等级': ['高', '中', '低', '高']
    })
    
    st.dataframe(
        decision_df.style.map(
            lambda x: 'background-color: #d4edda; font-weight: bold' if '⭐' in str(x) else '',
            subset=['推荐策略']
        ),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.success(f"""
    **P50 基准方案: 阶梯降产** ⭐
    
    - ✅ 保留 {gp_decay/gp_steady*100:.0f}% 稳产累产
    - ✅ Sw降低 {sw_steady-sw_decay:.3f} (延缓水侵)
    - ✅ 仅外推区生效, 不改历史拟合
    - ✅ 产量与安全最优平衡
    """)

st.markdown("---")

# ── 技术优势 ──
st.subheader("🔬 PINN vs 传统数值模拟")

col1, col2 = st.columns(2)

with col1:
    compare_df = pd.DataFrame({
        '维度': ['单次推演', '3策略全评', '参数反演', '硬件需求'],
        'PINN': ['< 1秒', '< 2秒', '训练自动完成', '单GPU/CPU'],
        '数值模拟': ['2~4小时', '6~12小时', '手动历史拟合', '计算集群'],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("""
    #### 本研究创新点
    
    1. **仅外推区施加策略** — 不改变历史拟合区, 物理可解释
    2. **水侵/干化分离** — dsw>0受策略影响, dsw<0保持原速率
    3. **Peaceman物理驱动** — qg ∝ (p_cell - p_wf), 非简化乘数
    4. **Sw clip** — 消除PINN伪影, Sw严格在[Swc, 1-Sgr]内
    """)

st.markdown("---")

# ── 完整报告 ──
try:
    report_path = project_root / 'outputs' / selected_exp / 'reports' / 'M7_water_invasion_report.md'
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        with st.expander("📄 查看完整 M7 报告（含定量对比数据）"):
            st.markdown(report_content)
except Exception:
    pass

st.caption("🔬 PINN秒级正演替代器 | v3.17 制度优化 | 仅外推区策略施加")
