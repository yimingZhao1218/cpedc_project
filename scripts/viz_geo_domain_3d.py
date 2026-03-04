"""
M2 地质域3D可视化 — 评委展示版
将 geo/ 目录下的全部数据（边界、顶底面、厚度、配点网格、井点）渲染为交互式3D HTML。
顶面(暖色) + 底面(冷色) 同时展示，井柱贯穿，边界围栏清晰。
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.spatial import Delaunay

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEO = PROJECT_ROOT / 'geo'
DATA = PROJECT_ROOT / 'data'
OUT_HTML = PROJECT_ROOT / 'outputs' / 'mk_pinn_dt_v2' / 'geo_domain_3d.html'

# ── 加载数据 ──
boundary   = pd.read_csv(GEO / 'boundary' / 'model_boundary.csv')
top_surf   = pd.read_csv(GEO / 'surfaces' / 'mk_top_surface.csv')
bot_surf   = pd.read_csv(GEO / 'surfaces' / 'mk_bot_surface.csv')
thickness  = pd.read_csv(GEO / 'surfaces' / 'mk_thickness.csv')
top_var    = pd.read_csv(GEO / 'surfaces' / 'mk_top_surface_variance.csv')
grid_pts   = pd.read_csv(GEO / 'grids' / 'collocation_grid.csv')
bnd_pts    = pd.read_csv(GEO / 'grids' / 'boundary_points.csv')
mk_pts     = pd.read_csv(DATA / 'clean' / 'mk_interval_points.csv')
wells      = pd.read_csv(DATA / 'staged' / 'wells_staged.csv')

# ── 坐标中心化（大坐标 → 局部 km）──
x0 = mk_pts['x_mid'].mean()
y0 = mk_pts['y_mid'].mean()
S  = 1000.0  # m → km

def lx(x): return (x - x0) / S
def ly(y): return (y - y0) / S

# ── 颜色 ──
BG       = '#080c14'
GRID_BG  = '#0b1018'
WELL_CLR = {
    'SY9': '#4FC3F7', 'SY13': '#FF8A65', 'SY201': '#81C784',
    'SY101': '#BA68C8', 'SY102': '#FFD54F', 'SY116': '#4DB6AC',
    'SYX211': '#E57373'
}
GWC_Z = -4385.0

# ── 自定义色阶 ──
# 顶面：暖色调（棕→米→浅绿），突出构造高点
TOP_COLORSCALE = [
    [0.0, 'rgb(120,70,20)'],
    [0.2, 'rgb(170,120,55)'],
    [0.4, 'rgb(210,175,110)'],
    [0.6, 'rgb(235,225,175)'],
    [0.8, 'rgb(170,205,165)'],
    [1.0, 'rgb(60,150,130)'],
]
# 底面：冷色调（深蓝→青→浅蓝），与顶面形成鲜明对比
BOT_COLORSCALE = [
    [0.0, 'rgb(10,30,80)'],
    [0.2, 'rgb(20,60,130)'],
    [0.4, 'rgb(35,100,170)'],
    [0.6, 'rgb(60,145,200)'],
    [0.8, 'rgb(100,185,220)'],
    [1.0, 'rgb(155,215,235)'],
]

# ── Figure ──
fig = go.Figure()

# ═══════════════════════════════════════════════════════════
# 1. MK 顶面 (Delaunay mesh, 暖色)
# ═══════════════════════════════════════════════════════════
valid_top = top_surf.dropna(subset=['z'])
step = max(1, len(valid_top) // 10000)
vt = valid_top.iloc[::step].reset_index(drop=True)

pts2d = np.column_stack([lx(vt['x'].values), ly(vt['y'].values)])
tri = Delaunay(pts2d)

fig.add_trace(go.Mesh3d(
    x=pts2d[:, 0], y=pts2d[:, 1], z=vt['z'].values,
    i=tri.simplices[:, 0], j=tri.simplices[:, 1], k=tri.simplices[:, 2],
    intensity=vt['z'].values,
    colorscale=TOP_COLORSCALE, opacity=0.92,
    colorbar=dict(
        title=dict(text='顶面标高(m)', font=dict(size=11, color='#E0C080')),
        x=-0.05, len=0.4, y=0.72, tickfont=dict(size=9, color='#B0BEC5'),
        bordercolor='#333', borderwidth=1,
    ),
    name='MK顶面',
    hovertemplate='X: %{x:.2f} km<br>Y: %{y:.2f} km<br>Z: %{z:.1f} m<extra>MK顶面</extra>',
    lighting=dict(ambient=0.65, diffuse=0.55, specular=0.25, roughness=0.5, fresnel=0.3),
    lightposition=dict(x=5000, y=-5000, z=15000),
))

# ═══════════════════════════════════════════════════════════
# 2. MK 底面 (Delaunay mesh, 冷色) — 默认显示
# ═══════════════════════════════════════════════════════════
valid_bot = bot_surf.dropna(subset=['z'])
step_b = max(1, len(valid_bot) // 10000)
vb = valid_bot.iloc[::step_b].reset_index(drop=True)

pts2d_b = np.column_stack([lx(vb['x'].values), ly(vb['y'].values)])
tri_b = Delaunay(pts2d_b)

fig.add_trace(go.Mesh3d(
    x=pts2d_b[:, 0], y=pts2d_b[:, 1], z=vb['z'].values,
    i=tri_b.simplices[:, 0], j=tri_b.simplices[:, 1], k=tri_b.simplices[:, 2],
    intensity=vb['z'].values,
    colorscale=BOT_COLORSCALE, opacity=0.80,
    colorbar=dict(
        title=dict(text='底面标高(m)', font=dict(size=11, color='#60A0D0')),
        x=-0.12, len=0.4, y=0.28, tickfont=dict(size=9, color='#B0BEC5'),
        bordercolor='#333', borderwidth=1,
    ),
    name='MK底面',
    hovertemplate='X: %{x:.2f} km<br>Y: %{y:.2f} km<br>Z: %{z:.1f} m<extra>MK底面</extra>',
    lighting=dict(ambient=0.55, diffuse=0.45, specular=0.15, roughness=0.6),
    lightposition=dict(x=5000, y=-5000, z=15000),
))

# ═══════════════════════════════════════════════════════════
# 3. 边界围栏竖线（顶-底垂直连线，半透明围栏效果）
# ═══════════════════════════════════════════════════════════
bx = lx(boundary['x'].values)
by = ly(boundary['y'].values)

# 沿边界每隔若干个点画一根竖线，连接顶底
z_top_mean = valid_top['z'].mean()
z_bot_mean = valid_bot['z'].mean()

fence_step = max(1, len(bx) // 40)
fence_x, fence_y, fence_z = [], [], []
for i in range(0, len(bx), fence_step):
    fence_x.extend([bx[i], bx[i], None])
    fence_y.extend([by[i], by[i], None])
    fence_z.extend([z_top_mean, z_bot_mean, None])

fig.add_trace(go.Scatter3d(
    x=fence_x, y=fence_y, z=fence_z,
    mode='lines',
    line=dict(color='rgba(100,180,255,0.3)', width=1.5),
    name='边界围栏',
    hoverinfo='skip',
    showlegend=False,
))

# ═══════════════════════════════════════════════════════════
# 4. GWC 平面（气水界面）
# ═══════════════════════════════════════════════════════════
fig.add_trace(go.Mesh3d(
    x=bx, y=by, z=np.full(len(bx), GWC_Z),
    color='rgba(30,136,229,0.20)',
    opacity=0.25,
    name=f'GWC ({GWC_Z}m)',
    hovertemplate=f'气水界面 GWC = {GWC_Z} m<extra></extra>',
    flatshading=True,
))

# ═══════════════════════════════════════════════════════════
# 5. 模型边界（3D围栏线：顶面+底面轮廓）
# ═══════════════════════════════════════════════════════════
for z_lev, nm, clr, dash, w in [
    (z_top_mean, '边界@顶面', '#FF9800', 'solid', 3.5),
    (z_bot_mean, '边界@底面', '#2196F3', 'dash', 3),
    (GWC_Z,      '边界@GWC',  '#1565C0', 'dot',  2),
]:
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=np.full(len(bx), z_lev),
        mode='lines',
        line=dict(color=clr, width=w, dash=dash),
        name=nm,
        hoverinfo='skip',
    ))

# ═══════════════════════════════════════════════════════════
# 6. 配点网格 (PDE + WELL_NEAR 分色)
# ═══════════════════════════════════════════════════════════
pde_pts  = grid_pts[~grid_pts['is_near_well']]
well_pts = grid_pts[grid_pts['is_near_well']]

pde_step = max(1, len(pde_pts) // 2000)
pde_show = pde_pts.iloc[::pde_step]

fig.add_trace(go.Scatter3d(
    x=lx(pde_show['x'].values), y=ly(pde_show['y'].values),
    z=np.full(len(pde_show), z_top_mean - 20),
    mode='markers',
    marker=dict(size=1.5, color='#546E7A', opacity=0.4),
    name=f'PDE配点 ({len(pde_pts)})',
    hoverinfo='skip',
    visible='legendonly',
))

well_step = max(1, len(well_pts) // 3000)
well_show = well_pts.iloc[::well_step]

fig.add_trace(go.Scatter3d(
    x=lx(well_show['x'].values), y=ly(well_show['y'].values),
    z=np.full(len(well_show), z_top_mean - 20),
    mode='markers',
    marker=dict(size=2, color='#FFB74D', opacity=0.5),
    name=f'井周加密 ({len(well_pts)})',
    hoverinfo='skip',
    visible='legendonly',
))

# ═══════════════════════════════════════════════════════════
# 7. 边界采样点
# ═══════════════════════════════════════════════════════════
fig.add_trace(go.Scatter3d(
    x=lx(bnd_pts['x'].values), y=ly(bnd_pts['y'].values),
    z=np.full(len(bnd_pts), z_top_mean - 10),
    mode='markers',
    marker=dict(size=2, color='#29B6F6', symbol='diamond', opacity=0.6),
    name=f'BC边界点 ({len(bnd_pts)})',
    hoverinfo='skip',
    visible='legendonly',
))

# ═══════════════════════════════════════════════════════════
# 8. 井点 (MK顶底 + 粗柱线 + 标签)
# ═══════════════════════════════════════════════════════════
for _, row in mk_pts.iterrows():
    wid = row['well_id']
    clr = WELL_CLR.get(wid, '#FFFFFF')
    xt, yt, zt = lx(row['x_top']), ly(row['y_top']), row['mk_top_z']
    xb, yb, zb = lx(row['x_bot']), ly(row['y_bot']), row['mk_bot_z']
    thk = row['mk_thickness']

    # MK段竖线（加粗到8，增强视觉）
    fig.add_trace(go.Scatter3d(
        x=[xt, xb], y=[yt, yb], z=[zt, zb],
        mode='lines+markers',
        line=dict(color=clr, width=8),
        marker=dict(
            size=[6, 5],
            color=clr,
            symbol='diamond',
            line=dict(color='white', width=1),
        ),
        name=wid,
        legendgroup=wid,
        showlegend=True,
        hovertemplate=(
            f'<b>{wid}</b><br>'
            f'MK顶: {zt:.1f}m<br>'
            f'MK底: {zb:.1f}m<br>'
            f'厚度: {thk:.1f}m'
            '<extra></extra>'
        ),
    ))

    # 井名标签（上移，加大字号）
    fig.add_trace(go.Scatter3d(
        x=[xt], y=[yt], z=[zt + 20],
        mode='text',
        text=[wid],
        textfont=dict(size=13, color=clr, family='Arial Black'),
        showlegend=False, legendgroup=wid,
        hoverinfo='skip',
    ))

# ═══════════════════════════════════════════════════════════
# 9. 统计注释（右下角信息卡片）
# ═══════════════════════════════════════════════════════════
thk_valid = thickness.dropna(subset=['z'])['z']
ann_text = (
    f"<b>MK组地质域统计</b><br>"
    f"井数: {len(mk_pts)} 口<br>"
    f"顶面标高: {valid_top['z'].min():.0f} ~ {valid_top['z'].max():.0f} m<br>"
    f"底面标高: {valid_bot['z'].min():.0f} ~ {valid_bot['z'].max():.0f} m<br>"
    f"厚度: {thk_valid.min():.0f} ~ {thk_valid.max():.0f} m (均值 {thk_valid.mean():.0f} m)<br>"
    f"GWC: {GWC_Z} m"
)

# ═══════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════
axis_common = dict(
    backgroundcolor=GRID_BG,
    gridcolor='#162030',
    showbackground=True,
    tickfont=dict(size=10, color='#90A4AE'),
)

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color='#CFD8DC', family='Microsoft YaHei, Arial, sans-serif'),
    title=dict(
        text=(
            '<b>碳酸盐岩气藏MK组三维地质模型</b><br>'
            '<span style="font-size:12px;color:#90A4AE">'
            'Kriging插值构造面 · 顶面(暖色) / 底面(冷色) · 7口井MK段柱 · GWC气水界面'
            '</span>'
        ),
        font=dict(size=17, color='#E0E0E0'),
        x=0.5,
        y=0.97,
    ),
    scene=dict(
        xaxis=dict(**axis_common, title='X (km)'),
        yaxis=dict(**axis_common, title='Y (km)'),
        zaxis=dict(
            **axis_common,
            title='标高 (m)',
            range=[GWC_Z - 50, valid_top['z'].max() + 30],
        ),
        aspectmode='manual',
        aspectratio=dict(x=2, y=2, z=0.7),
        camera=dict(
            eye=dict(x=1.6, y=-1.4, z=0.9),
            up=dict(x=0, y=0, z=1),
        ),
    ),
    legend=dict(
        bgcolor='rgba(10,14,23,0.90)',
        bordercolor='#37474F',
        borderwidth=1,
        font=dict(size=11, color='#CFD8DC'),
        itemsizing='constant',
        yanchor='top',
        y=0.98,
        xanchor='right',
        x=0.99,
        tracegroupgap=2,
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    autosize=True,
    annotations=[
        dict(
            text=ann_text,
            showarrow=False,
            xref='paper', yref='paper',
            x=0.01, y=0.01,
            xanchor='left', yanchor='bottom',
            font=dict(size=11, color='#B0BEC5', family='Microsoft YaHei'),
            bgcolor='rgba(10,14,23,0.85)',
            bordercolor='#37474F',
            borderwidth=1,
            borderpad=8,
            align='left',
        )
    ],
)

# ═══════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════
os.makedirs(OUT_HTML.parent, exist_ok=True)
fig.write_html(
    str(OUT_HTML),
    include_plotlyjs='cdn',
    full_html=True,
    default_width='100vw',
    default_height='100vh',
    config={
        'displaylogo': False,
        'toImageButtonOptions': {'format': 'png', 'width': 2400, 'height': 1600, 'scale': 2},
    }
)

size_mb = os.path.getsize(OUT_HTML) / 1024 / 1024
print(f"✅ 已保存: {OUT_HTML}")
print(f"   文件大小: {size_mb:.1f} MB")
print(f"   Trace 数: {len(fig.data)}")

# 自动打开
import webbrowser
webbrowser.open(str(OUT_HTML))
