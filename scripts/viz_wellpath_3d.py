"""
碳酸盐岩气藏三维地质模型  (CPEDC 2026 · 一等奖标准)
输出: outputs/mk_pinn_dt_v2/wellpath_3d_viz.html
设计理念: Petrel/RMS 专业风格, 图层分组切换, 简洁不花哨
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy.spatial import Delaunay

# ═══════════════════════════════════════════════════════════════
#  1. 数据加载
# ═══════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent
wellpath     = pd.read_csv(ROOT / 'data/clean/wellpath_stations.csv')
mk_pts       = pd.read_csv(ROOT / 'data/clean/mk_interval_points.csv')
wells        = pd.read_csv(ROOT / 'data/staged/wells_staged.csv')
mk_top_surf  = pd.read_csv(ROOT / 'geo/surfaces/mk_top_surface.csv')
mk_bot_surf  = pd.read_csv(ROOT / 'geo/surfaces/mk_bot_surface.csv')
boundary_pts = pd.read_csv(ROOT / 'geo/grids/boundary_points.csv')

x0 = wellpath['x'].mean()
y0 = wellpath['y'].mean()
kb_map = dict(zip(wells['well_id'], wells['kb_elev']))
GWC_Z = -4385.0

# ═══════════════════════════════════════════════════════════════
#  2. Petrel 专业配色（低饱和、高辨识）
# ═══════════════════════════════════════════════════════════════
WELL_COLORS = {
    'SY9':    '#4FC3F7',  # 淡蓝
    'SY13':   '#FFB74D',  # 琥珀
    'SY101':  '#81C784',  # 薄荷
    'SY102':  '#E57373',  # 玫红
    'SY116':  '#BA68C8',  # 淡紫
    'SY201':  '#FFF176',  # 柠檬
    'SYX211': '#F06292',  # 粉
}
# 流体类型图标: 气井=绿圆, 气水井=蓝方
FLUID_TYPE = {
    'SY9': '气井', 'SY13': '气井', 'SY101': '气井',
    'SY201': '气井', 'SY116': '气井',
    'SY102': '气水井', 'SYX211': '气水井',
}
BG      = '#101820'
GRID_C  = 'rgba(255,255,255,0.04)'
AXIS_C  = 'rgba(200,210,220,0.30)'
FONT    = 'Microsoft YaHei, sans-serif'

well_ids = sorted(wellpath['well_id'].unique())
fig = go.Figure()

# ═══════════════════════════════════════════════════════════════
#  trace 索引记录（用于图层切换 updatemenus）
# ═══════════════════════════════════════════════════════════════
trace_groups = {
    'surfaces': [],   # 构造面
    'wells':    [],   # 井轨迹+标记
    'mk':       [],   # MK段高亮
    'water':    [],   # 见水信息+GWC
    'ref':      [],   # 参考面+边界
}
_ti = 0  # trace index counter

def _track(group):
    global _ti
    trace_groups[group].append(_ti)
    _ti += 1


# ═══════════════════════════════════════════════════════════════
#  3. MK 构造面（Kriging — Delaunay 三角化）
# ═══════════════════════════════════════════════════════════════
def add_surface(df, name, cscale, opacity, show_cbar=False, group='surfaces'):
    valid = df.dropna(subset=['z']).copy()
    xu = np.sort(valid['x'].unique())
    yu = np.sort(valid['y'].unique())
    valid = valid[valid['x'].isin(set(xu[::2])) & valid['y'].isin(set(yu[::2]))]
    px = (valid['x'].values - x0).astype(np.float64)
    py = (valid['y'].values - y0).astype(np.float64)
    pz = valid['z'].values.astype(np.float64)
    tri = Delaunay(np.column_stack([px, py]))

    fig.add_trace(go.Mesh3d(
        x=px, y=py, z=pz,
        i=tri.simplices[:, 0], j=tri.simplices[:, 1], k=tri.simplices[:, 2],
        intensity=pz,
        colorscale=cscale,
        opacity=opacity,
        showscale=show_cbar,
        colorbar=dict(
            title=dict(text='海拔 (m)', font=dict(size=11, color='#90A4AE')),
            tickfont=dict(size=10, color='#78909C'),
            len=0.35, x=1.01, y=0.30,
            thickness=12,
            outlinewidth=0,
            bgcolor='rgba(16,24,32,0.85)',
        ) if show_cbar else None,
        name=name,
        showlegend=True,
        legendgroup=group,
        flatshading=False,
        lighting=dict(ambient=0.55, diffuse=0.65, specular=0.25,
                      roughness=0.6, fresnel=0.15),
        lightposition=dict(x=8000, y=-6000, z=15000),
        hovertemplate=f'<b>{name}</b><br>Z: %{{z:.1f}} m<extra></extra>',
    ))
    _track(group)

# 地球科学色阶: 深→浅 = 低→高
TOP_CSCALE = [
    [0, '#1A237E'], [0.25, '#1565C0'], [0.5, '#26A69A'],
    [0.75, '#66BB6A'], [1.0, '#AED581'],
]
BOT_CSCALE = [
    [0, '#4E342E'], [0.25, '#8D6E63'], [0.5, '#BCAAA4'],
    [0.75, '#D7CCC8'], [1.0, '#EFEBE9'],
]

add_surface(mk_top_surf, 'MK 顶界 (Kriging)', TOP_CSCALE, 0.35, show_cbar=True)
add_surface(mk_bot_surf, 'MK 底界 (Kriging)', BOT_CSCALE, 0.28)


# ═══════════════════════════════════════════════════════════════
#  4. 模型边界
# ═══════════════════════════════════════════════════════════════
bx = np.append(boundary_pts['x'].values, boundary_pts['x'].values[0]) - x0
by = np.append(boundary_pts['y'].values, boundary_pts['y'].values[0]) - y0
z_mk_avg = mk_pts['z_top_traj'].mean()

fig.add_trace(go.Scatter3d(
    x=bx, y=by, z=np.full_like(bx, z_mk_avg),
    mode='lines',
    line=dict(color='rgba(176,190,197,0.45)', width=2.5, dash='dash'),
    name='模型边界', showlegend=True, legendgroup='ref', hoverinfo='skip',
))
_track('ref')


# ═══════════════════════════════════════════════════════════════
#  辅助：沿轨迹生成高质量圆柱管道 Mesh3d
# ═══════════════════════════════════════════════════════════════
def make_tube(xs, ys, zs, radius=60, n_sides=16, color='#FFFFFF',
              opacity=0.40, name='', group='wells', legend=False):
    """沿 (xs,ys,zs) 轨迹点生成光滑圆柱管道 Mesh3d"""
    n = len(xs)
    if n < 2:
        return
    # 降采样：最多取 80 个截面
    step = max(1, n // 80)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    xs = np.array([xs[i] for i in idx], dtype=np.float64)
    ys = np.array([ys[i] for i in idx], dtype=np.float64)
    zs = np.array([zs[i] for i in idx], dtype=np.float64)
    n_rings = len(xs)

    # 每个截面的切线方向
    vx, vy, vz = [], [], []
    theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    for k in range(n_rings):
        # 切线
        if k == 0:
            tx, ty, tz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]
        elif k == n_rings - 1:
            tx, ty, tz = xs[-1]-xs[-2], ys[-1]-ys[-2], zs[-1]-zs[-2]
        else:
            tx, ty, tz = xs[k+1]-xs[k-1], ys[k+1]-ys[k-1], zs[k+1]-zs[k-1]
        tlen = np.sqrt(tx**2 + ty**2 + tz**2)
        if tlen < 1e-12:
            tx, ty, tz = 0, 0, 1
        else:
            tx, ty, tz = tx/tlen, ty/tlen, tz/tlen
        # 两个正交法向量
        if abs(tz) < 0.9:
            n1 = np.cross([tx, ty, tz], [0, 0, 1])
        else:
            n1 = np.cross([tx, ty, tz], [1, 0, 0])
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.cross([tx, ty, tz], n1)
        n2 = n2 / np.linalg.norm(n2)
        for t in theta:
            vx.append(xs[k] + radius * (np.cos(t)*n1[0] + np.sin(t)*n2[0]))
            vy.append(ys[k] + radius * (np.cos(t)*n1[1] + np.sin(t)*n2[1]))
            vz.append(zs[k] + radius * (np.cos(t)*n1[2] + np.sin(t)*n2[2]))

    # 三角面片
    ii, jj, kk = [], [], []
    for r in range(n_rings - 1):
        for s in range(n_sides):
            s1 = (s + 1) % n_sides
            a = r * n_sides + s
            b = r * n_sides + s1
            c = (r+1) * n_sides + s
            d = (r+1) * n_sides + s1
            ii += [a, b]; jj += [b, d]; kk += [c, c]

    fig.add_trace(go.Mesh3d(
        x=vx, y=vy, z=vz, i=ii, j=jj, k=kk,
        color=color, opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.35,
                      roughness=0.4, fresnel=0.2),
        lightposition=dict(x=8000, y=-6000, z=15000),
        name=name, showlegend=legend, legendgroup=group,
        hoverinfo='skip',
    ))
    _track(group)


# ═══════════════════════════════════════════════════════════════
#  5. 井眼轨迹 + 圆柱管 + MK段 + 标注
# ═══════════════════════════════════════════════════════════════
for wid in well_ids:
    c   = WELL_COLORS[wid]
    wdf = wellpath[wellpath['well_id'] == wid].sort_values('md_m')
    mk  = mk_pts[mk_pts['well_id'] == wid]
    zv  = wdf['z'].values
    wx  = wdf['x'].values - x0
    wy  = wdf['y'].values - y0
    inc = wdf['inc_deg'].values
    ft  = FLUID_TYPE[wid]
    sym = 'square' if ft == '气水井' else 'circle'

    # 5a-0. 井筒圆柱管道
    make_tube(wx, wy, zv, radius=70, n_sides=16, color=c,
              opacity=0.30, group='wells')

    # 5a. 井眼轨迹中心线 — 井斜渐变
    cdata = np.column_stack([wdf['md_m'].values, inc])
    fig.add_trace(go.Scatter3d(
        x=wx, y=wy, z=zv,
        mode='lines',
        line=dict(
            color=inc, width=4.5,
            colorscale=[[0, c], [0.2, c], [0.6, '#CFD8DC'], [1.0, '#EF5350']],
            cmin=0, cmax=60,
            showscale=(wid == 'SY9'),
            colorbar=dict(
                title=dict(text='井斜 (°)', font=dict(size=10, color='#90A4AE')),
                tickfont=dict(size=9, color='#78909C'),
                len=0.25, x=1.01, y=0.72, thickness=10,
                outlinewidth=0, bgcolor='rgba(16,24,32,0.85)',
            ) if wid == 'SY9' else None,
        ),
        name=wid, legendgroup='wells',
        customdata=cdata,
        hovertemplate=(
            f'<b style="color:{c}">{wid}</b> ({ft})<br>'
            'MD: <b>%{customdata[0]:.0f}</b> m<br>'
            '海拔: <b>%{z:.1f}</b> m<br>'
            '井斜: <b>%{customdata[1]:.1f}°</b><extra></extra>'
        ),
    ))
    _track('wells')

    # 5b. 井口标记
    fig.add_trace(go.Scatter3d(
        x=[wx[0]], y=[wy[0]], z=[zv[0]],
        mode='markers+text',
        marker=dict(size=8, color=c, symbol=sym,
                    line=dict(width=1.5, color='white')),
        text=[wid], textposition='top center',
        textfont=dict(size=12, color=c, family=FONT),
        showlegend=False, legendgroup='wells',
        hovertemplate=(
            f'<b>{wid} 井口</b><br>'
            f'地面海拔: {zv[0]:.1f} m<br>'
            f'完钻深度: {wdf["md_m"].iloc[-1]:.0f} m<br>'
            f'类型: {ft}<extra></extra>'
        ),
    ))
    _track('wells')

    if mk.empty:
        continue

    xt, yt, zt = mk['x_top'].values[0]-x0, mk['y_top'].values[0]-y0, mk['z_top_traj'].values[0]
    xb, yb, zb = mk['x_bot'].values[0]-x0, mk['y_bot'].values[0]-y0, mk['z_bot_traj'].values[0]
    h = mk['mk_thickness'].values[0]

    # 5c. MK储层段 — 金色粗圆柱
    mk_wdf = wdf[(wdf['z'] <= zt + 5) & (wdf['z'] >= zb - 5)]
    if len(mk_wdf) >= 2:
        mk_wx = mk_wdf['x'].values - x0
        mk_wy = mk_wdf['y'].values - y0
        mk_wz = mk_wdf['z'].values
    else:
        mk_wx = np.array([xt, xb])
        mk_wy = np.array([yt, yb])
        mk_wz = np.array([zt, zb])
    make_tube(mk_wx, mk_wy, mk_wz, radius=150, n_sides=16,
              color='#FDD835', opacity=0.50,
              name='MK 储层段' if wid == well_ids[0] else '',
              group='mk', legend=(wid == well_ids[0]))

    # 5d. MK顶 ◆
    fig.add_trace(go.Scatter3d(
        x=[xt], y=[yt], z=[zt],
        mode='markers',
        marker=dict(size=7, color='#FDD835', symbol='diamond',
                    line=dict(width=1.5, color='white')),
        showlegend=False, legendgroup='mk',
        hovertemplate=(
            f'<b>{wid} MK顶</b><br>'
            f'Z: {zt:.1f} m<br>'
            f'厚度: {h:.1f} m<extra></extra>'
        ),
    ))
    _track('mk')

    # 5e. MK底 ■
    fig.add_trace(go.Scatter3d(
        x=[xb], y=[yb], z=[zb],
        mode='markers',
        marker=dict(size=5, color='#FF8F00', symbol='square',
                    line=dict(width=1, color='white')),
        showlegend=False, legendgroup='mk',
        hovertemplate=f'<b>{wid} MK底</b><br>Z: {zb:.1f} m<extra></extra>',
    ))
    _track('mk')


# ═══════════════════════════════════════════════════════════════
#  6. 见水信息
# ═══════════════════════════════════════════════════════════════
# 附表8-测井解释成果表 — 精确值，全部7井全部层段
log_interp = {
    'SY9':    [dict(tvd_top=4546.77,  tvd_bot=4563.066, sw=13.80,   label='气层'),
               dict(tvd_top=4571.542, tvd_bot=4606.021, sw=15.5369, label='气层')],
    'SY13':   [dict(tvd_top=4574.421, tvd_bot=4579.671, sw=6.9091,  label='气层'),
               dict(tvd_top=4592.796, tvd_bot=4604.296, sw=12.6867, label='气层'),
               dict(tvd_top=4607.546, tvd_bot=4633.421, sw=12.385,  label='气层')],
    'SY101':  [dict(tvd_top=4588.792, tvd_bot=4635.042, sw=21.5,    label='气层'),
               dict(tvd_top=4638.292, tvd_bot=4642.042, sw=0.0,     label='气层')],
    'SY102':  [dict(tvd_top=4617.598, tvd_bot=4663.038, sw=16.8342, label='气层')],
    'SY116':  [dict(tvd_top=4630.479, tvd_bot=4635.978, sw=18.4429, label='气层'),
               dict(tvd_top=4637.853, tvd_bot=4679.347, sw=28.2212, label='气层')],
    'SY201':  [dict(tvd_top=4548.14,  tvd_bot=4555.39,  sw=14.3004, label='气层'),
               dict(tvd_top=4575.39,  tvd_bot=4608.64,  sw=11.7281, label='气层')],
    'SYX211': [dict(tvd_top=4661.439, tvd_bot=4673.416, sw=30.3,    label='气水同层'),
               dict(tvd_top=4673.883, tvd_bot=4681.188, sw=66.5775, label='水层')],
}

def sw_color(sw):
    if sw < 15:   return '#4CAF50'   # 深绿 — 低含水(纯气)
    elif sw < 25: return '#8BC34A'   # 浅绿 — 中低含水
    elif sw < 40: return '#FFA726'   # 橙   — 中高含水
    elif sw < 55: return '#EF5350'   # 红   — 气水同层
    else:         return '#42A5F5'   # 蓝   — 水层

first_water_legend = True
for wid, layers in log_interp.items():
    kb = kb_map.get(wid, 300.0)
    wdf_w = wellpath[wellpath['well_id'] == wid].sort_values('md_m')
    for layer in layers:
        if layer['sw'] <= 0:
            continue
        z_top = kb - layer['tvd_top']
        z_bot = kb - layer['tvd_bot']
        z_mid = (z_top + z_bot) / 2
        sc = sw_color(layer['sw'])
        x_m = np.interp(z_mid, wdf_w['z'].values[::-1], (wdf_w['x'].values - x0)[::-1])
        y_m = np.interp(z_mid, wdf_w['z'].values[::-1], (wdf_w['y'].values - y0)[::-1])

        # 含水段色标线
        fig.add_trace(go.Scatter3d(
            x=[x_m, x_m], y=[y_m, y_m], z=[z_top, z_bot],
            mode='lines', line=dict(color=sc, width=12),
            showlegend=False, legendgroup='water',
            hovertemplate=(
                f'<b style="color:{sc}">{wid} {layer["label"]}</b><br>'
                f'TVD: {layer["tvd_top"]:.1f}~{layer["tvd_bot"]:.1f} m<br>'
                f'Sw: <b>{layer["sw"]:.1f}%</b><br>'
                f'Sg: <b>{100 - layer["sw"]:.1f}%</b><extra></extra>'
            ),
        ))
        _track('water')

        # 文字标签：气水同层/水层 或 Sw≥20%
        if layer['label'] in ('气水同层', '水层') or layer['sw'] >= 20:
            fig.add_trace(go.Scatter3d(
                x=[x_m], y=[y_m], z=[z_mid],
                mode='markers+text',
                marker=dict(size=4, color=sc, line=dict(width=1, color='white')),
                text=[f"  {layer['label']} Sw={layer['sw']:.1f}%"],
                textposition='middle right',
                textfont=dict(size=10, color=sc, family=FONT),
                name='测井Sw色标' if first_water_legend else '',
                showlegend=first_water_legend, legendgroup='water',
                hoverinfo='skip',
            ))
            _track('water')
            first_water_legend = False

# 6b. GWC 平面 — 裁剪到模型边界（Delaunay）
bnd_x = boundary_pts['x'].values - x0
bnd_y = boundary_pts['y'].values - y0
# 在边界内生成均匀网格点
from matplotlib.path import Path as MplPath
bnd_path = MplPath(np.column_stack([bnd_x, bnd_y]))
gx = np.linspace(bnd_x.min(), bnd_x.max(), 60)
gy = np.linspace(bnd_y.min(), bnd_y.max(), 60)
gxx, gyy = np.meshgrid(gx, gy)
pts_flat = np.column_stack([gxx.ravel(), gyy.ravel()])
inside = bnd_path.contains_points(pts_flat)
gwc_x = pts_flat[inside, 0]
gwc_y = pts_flat[inside, 1]
gwc_z = np.full_like(gwc_x, GWC_Z)
gwc_tri = Delaunay(np.column_stack([gwc_x, gwc_y]))

fig.add_trace(go.Mesh3d(
    x=gwc_x, y=gwc_y, z=gwc_z,
    i=gwc_tri.simplices[:, 0], j=gwc_tri.simplices[:, 1], k=gwc_tri.simplices[:, 2],
    color='rgba(30,136,229,0.35)',
    flatshading=True,
    name=f'GWC 气水界面 ({GWC_Z:.0f} m)',
    showlegend=True, legendgroup='water',
    hovertemplate=f'<b>气水界面 (GWC)</b><br>Z = {GWC_Z:.0f} m<extra></extra>',
))
_track('water')

# 6c. GWC边界环线
fig.add_trace(go.Scatter3d(
    x=np.append(bnd_x, bnd_x[0]),
    y=np.append(bnd_y, bnd_y[0]),
    z=np.full(len(bnd_x)+1, GWC_Z),
    mode='lines',
    line=dict(color='rgba(30,136,229,0.6)', width=2),
    showlegend=False, legendgroup='water', hoverinfo='skip',
))
_track('water')


# ═══════════════════════════════════════════════════════════════
#  7. 地面参考面（极淡）
# ═══════════════════════════════════════════════════════════════
pad = 1500
xs_g = np.linspace(wellpath['x'].min()-x0-pad, wellpath['x'].max()-x0+pad, 4)
ys_g = np.linspace(wellpath['y'].min()-y0-pad, wellpath['y'].max()-y0+pad, 4)
xx, yy = np.meshgrid(xs_g, ys_g)
fig.add_trace(go.Surface(
    x=xx, y=yy, z=np.zeros_like(xx), opacity=0.04,
    colorscale=[[0, '#263238'], [1, '#263238']],
    showscale=False, name='地面 (z=0)', showlegend=True,
    legendgroup='ref', hoverinfo='skip',
))
_track('ref')

# ═══════════════════════════════════════════════════════════════
#  8. 比例尺参考线 (2000 m)
# ═══════════════════════════════════════════════════════════════
sx0 = wellpath['x'].min() - x0 - 500
sy0 = wellpath['y'].min() - y0 - 500
fig.add_trace(go.Scatter3d(
    x=[sx0, sx0+2000], y=[sy0, sy0], z=[0, 0],
    mode='lines+text',
    line=dict(color='rgba(200,200,200,0.6)', width=3),
    text=['', '2000 m'], textposition='top right',
    textfont=dict(size=10, color='rgba(200,200,200,0.6)', family=FONT),
    showlegend=False, hoverinfo='skip',
))
_track('ref')


# ═══════════════════════════════════════════════════════════════
#  9. 相机预设 & 图层切换
# ═══════════════════════════════════════════════════════════════
cam = dict(
    全景=dict(eye=dict(x=1.7, y=-1.4, z=0.55), up=dict(x=0, y=0, z=1)),
    俯视=dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)),
    南视=dict(eye=dict(x=0, y=-2.5, z=0.25), up=dict(x=0, y=0, z=1)),
    东视=dict(eye=dict(x=2.5, y=0, z=0.25), up=dict(x=0, y=0, z=1)),
    储层=dict(eye=dict(x=1.3, y=-0.9, z=0.10),
              center=dict(x=0, y=0, z=-0.6), up=dict(x=0, y=0, z=1)),
)

# 图层可见性切换按钮
n_traces = _ti
def make_vis(groups_on):
    vis = [False] * n_traces
    for g in groups_on:
        for idx in trace_groups[g]:
            vis[idx] = True
    return vis

layer_buttons = [
    dict(label='全部显示',
         method='update',
         args=[{'visible': [True]*n_traces}]),
    dict(label='仅构造面+井',
         method='update',
         args=[{'visible': make_vis(['surfaces','wells','mk','ref'])}]),
    dict(label='仅见水信息',
         method='update',
         args=[{'visible': make_vis(['wells','water','ref'])}]),
    dict(label='仅井轨迹',
         method='update',
         args=[{'visible': make_vis(['wells','ref'])}]),
]


# ═══════════════════════════════════════════════════════════════
#  10. Plotly 布局（全屏沉浸式，面板浮动叠加）
# ═══════════════════════════════════════════════════════════════
axis_common = dict(
    backgroundcolor=BG,
    gridcolor=GRID_C, gridwidth=1,
    color=AXIS_C, showspikes=False,
    zeroline=False,
)

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family=FONT),
    title=None,
    showlegend=False,
    scene=dict(
        xaxis=dict(title='E (m)', **axis_common),
        yaxis=dict(title='N (m)', **axis_common),
        zaxis=dict(title='海拔 (m)', **axis_common),
        aspectmode='manual',
        aspectratio=dict(x=1.5, y=1.0, z=1.3),
        camera=cam['全景'],
        bgcolor=BG,
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    autosize=True,
)


# ═══════════════════════════════════════════════════════════════
#  11. 构建数据供 HTML 面板
# ═══════════════════════════════════════════════════════════════
well_rows_html = ''
for wid in well_ids:
    wc = WELL_COLORS[wid]
    ft = FLUID_TYPE[wid]
    mk_row = mk_pts[mk_pts['well_id'] == wid]
    h_val = f'{mk_row["mk_thickness"].values[0]:.1f}' if not mk_row.empty else '—'
    zt_val = f'{mk_row["z_top_traj"].values[0]:.0f}' if not mk_row.empty else '—'
    td_val = f'{wellpath[wellpath["well_id"]==wid]["md_m"].max():.0f}'
    sw_layers = log_interp.get(wid, [])
    sw_vals = [l['sw'] for l in sw_layers if l['sw'] > 0]
    sw_avg = f'{np.mean(sw_vals):.1f}' if sw_vals else '—'
    sym_html = '■' if ft == '气水井' else '●'
    well_rows_html += (
        f'<tr class="wr">'
        f'<td><span style="color:{wc}">{sym_html}</span> {wid}</td>'
        f'<td>{td_val}</td><td>{h_val}</td><td>{zt_val}</td><td>{sw_avg}</td>'
        f'</tr>'
    )


# ═══════════════════════════════════════════════════════════════
#  12. 全屏沉浸式 HTML（Plotly 铺满 + 浮动面板叠加）
# ═══════════════════════════════════════════════════════════════
import json, re

plotly_config = {
    'displaylogo': False,
    'modeBarButtonsToRemove': ['resetCameraDefault3d'],
    'toImageButtonOptions': {
        'format': 'png', 'width': 1920, 'height': 1080,
        'filename': 'SY11_3D_GeoModel',
    },
    'responsive': True,
}

plotly_div = fig.to_html(
    full_html=False, include_plotlyjs='cdn',
    config=plotly_config,
    default_width='100vw', default_height='100vh',
)

div_id_match = re.search(r'id="([^"]+)"', plotly_div)
pid = div_id_match.group(1) if div_id_match else 'plotly-div'

cam_js = json.dumps(cam, ensure_ascii=False)
vis_js = json.dumps({
    'all':   [True]*n_traces,
    'surf':  make_vis(['surfaces','wells','mk','ref']),
    'water': make_vis(['wells','water','ref']),
    'wells': make_vis(['wells','ref']),
}, ensure_ascii=False)

n_surf_pts = len(mk_top_surf.dropna(subset=['z'])) + len(mk_bot_surf.dropna(subset=['z']))
n_wp = len(wellpath)

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SY11井区 · 三维地质综合模型</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0e17;color:#e0e4e8;font-family:'Noto Sans SC',sans-serif;overflow:hidden;height:100vh;width:100vw}}

/* Plotly 全屏底层 */
#plotly-wrap{{position:fixed;inset:0;z-index:1}}
#plotly-wrap .js-plotly-plot,.plotly-graph-div{{width:100vw!important;height:100vh!important}}

/* 所有浮动面板的公共样式 */
.fp{{position:fixed;z-index:10;
  background:rgba(10,14,23,0.82);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(255,255,255,0.07);
  border-radius:10px;
  color:#b0bec5;font-size:11px;
  transition:opacity .3s,transform .3s;
  pointer-events:auto}}
.fp.hide{{opacity:0;pointer-events:none;transform:translateY(8px)}}

/* 顶部标题条 */
#bar-top{{top:12px;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:14px;padding:8px 20px;
  white-space:nowrap}}
#bar-top h1{{font-size:15px;font-weight:500;color:#eceff1;letter-spacing:.3px}}
#bar-top h1 small{{font-weight:300;font-size:11px;color:#78909c;margin-left:8px}}
.badge{{background:rgba(79,195,247,.12);color:#4fc3f7;padding:3px 10px;
  border-radius:14px;font-size:10px;border:1px solid rgba(79,195,247,.2);font-weight:500}}

/* 左侧控制面板 */
#panel-left{{top:60px;left:12px;width:240px;padding:12px;
  display:flex;flex-direction:column;gap:10px;
  max-height:calc(100vh - 80px);overflow-y:auto}}
#panel-left::-webkit-scrollbar{{width:2px}}
#panel-left::-webkit-scrollbar-thumb{{background:rgba(255,255,255,.08);border-radius:2px}}

.sec-title{{font-size:10px;font-weight:500;color:#607d8b;
  text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;
  padding-left:8px;border-left:2px solid #4fc3f7}}

/* 按钮 */
.cb{{display:flex;align-items:center;gap:6px;
  background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);
  border-radius:6px;padding:6px 8px;cursor:pointer;
  color:#90a4ae;font-size:11px;font-family:inherit;
  transition:all .15s;text-align:left;width:100%}}
.cb:hover{{background:rgba(79,195,247,.08);border-color:rgba(79,195,247,.2);color:#e0e4e8}}
.cb.on{{background:rgba(79,195,247,.14);border-color:#4fc3f7;color:#4fc3f7}}

/* 右上角信息卡 */
#card-info{{top:60px;right:12px;width:200px;padding:12px}}
.info-row{{display:flex;justify-content:space-between;padding:3px 0;
  border-bottom:1px solid rgba(255,255,255,.03)}}
.info-row .v{{color:#e0e4e8;font-family:'JetBrains Mono',monospace;font-size:10.5px}}

/* 右下角井表 */
#card-wells{{bottom:56px;right:12px;width:340px;padding:12px}}
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;color:#546e7a;font-weight:400;font-size:9px;
  text-transform:uppercase;letter-spacing:.5px;padding:3px 4px;
  border-bottom:1px solid rgba(255,255,255,.06)}}
td{{padding:4px;font-family:'JetBrains Mono',monospace;font-size:10px;
  color:#90a4ae;border-bottom:1px solid rgba(255,255,255,.02)}}
.wr:hover td{{color:#e0e4e8;background:rgba(79,195,247,.04)}}

/* 底部状态栏 */
#bar-bot{{bottom:12px;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:20px;padding:6px 18px;font-size:10px}}
.si{{display:flex;align-items:center;gap:4px}}
.si .d{{width:5px;height:5px;border-radius:50%;flex-shrink:0}}
.si .n{{color:#e0e4e8;font-family:'JetBrains Mono',monospace;font-weight:500}}

/* 左下角图例 */
#card-legend{{bottom:56px;left:12px;width:180px;padding:10px}}
.lg{{display:flex;align-items:center;gap:5px;margin:2px 0;font-size:9.5px;color:#607d8b}}
.ls{{width:16px;height:6px;border-radius:1px;flex-shrink:0}}

/* 隐藏/显示面板的总开关 */
#toggle-ui{{position:fixed;bottom:14px;right:14px;z-index:20;
  width:32px;height:32px;border-radius:50%;
  background:rgba(10,14,23,.8);border:1px solid rgba(255,255,255,.1);
  color:#78909c;font-size:14px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:all .2s}}
#toggle-ui:hover{{background:rgba(79,195,247,.15);color:#4fc3f7}}
</style>
</head>
<body>

<!-- Plotly 全屏底层 -->
<div id="plotly-wrap">
{plotly_div}
</div>

<!-- 顶部标题 -->
<div class="fp" id="bar-top">
  <h1>SY11井区碳酸盐岩气藏 · 三维地质综合模型<small>井眼轨迹 · MK Kriging构造面 · 气水界面</small></h1>
  <span class="badge">CPEDC 2026</span>
</div>

<!-- 左侧控制面板 -->
<div class="fp" id="panel-left">
  <div class="sec-title">视角</div>
  <div id="cg">
    <button class="cb on" onclick="camTo('全景',this)">🔭 全景</button>
    <button class="cb" onclick="camTo('俯视',this)">⬇ 俯视</button>
    <button class="cb" onclick="camTo('南视',this)">🧭 南视</button>
    <button class="cb" onclick="camTo('东视',this)">➡ 东视</button>
    <button class="cb" onclick="camTo('储层',this)">🎯 储层聚焦</button>
  </div>
  <div class="sec-title" style="margin-top:6px">图层</div>
  <div id="lg">
    <button class="cb on" onclick="layTo('all',this)">◉ 全部</button>
    <button class="cb" onclick="layTo('surf',this)">◧ 构造面+井</button>
    <button class="cb" onclick="layTo('water',this)">💧 见水</button>
    <button class="cb" onclick="layTo('wells',this)">│ 仅井</button>
  </div>
</div>

<!-- 右上角信息 -->
<div class="fp" id="card-info">
  <div class="sec-title">项目概要</div>
  <div class="info-row"><span>研究区</span><span class="v">SY11井区</span></div>
  <div class="info-row"><span>目的层</span><span class="v">茅口组(MK)</span></div>
  <div class="info-row"><span>井数</span><span class="v">7 (5气+2气水)</span></div>
  <div class="info-row"><span>GWC</span><span class="v" style="color:#4fc3f7">{GWC_Z:.0f} m</span></div>
  <div class="info-row"><span>坐标系</span><span class="v">CGCS2000</span></div>
</div>

<!-- 右下角井表 -->
<div class="fp" id="card-wells">
  <div class="sec-title">井信息一览 (附表8)</div>
  <table>
    <tr><th>井号</th><th>完钻(m)</th><th>MK厚(m)</th><th>MK顶(m)</th><th>Sw(%)</th></tr>
    {well_rows_html}
  </table>
</div>

<!-- 左下角图例 -->
<div class="fp" id="card-legend">
  <div class="sec-title">Sw 色标</div>
  <div class="lg"><span class="ls" style="background:#4CAF50"></span>&lt;15% 纯气</div>
  <div class="lg"><span class="ls" style="background:#8BC34A"></span>15~25%</div>
  <div class="lg"><span class="ls" style="background:#FFA726"></span>25~40%</div>
  <div class="lg"><span class="ls" style="background:#EF5350"></span>40~55% 气水</div>
  <div class="lg"><span class="ls" style="background:#42A5F5"></span>≥55% 水层</div>
  <div style="margin-top:5px;border-top:1px solid rgba(255,255,255,.05);padding-top:5px">
    <div class="lg"><span class="ls" style="background:#FDD835"></span>MK 储层段</div>
    <div class="lg"><span class="ls" style="background:rgba(30,136,229,.5)"></span>GWC 界面</div>
    <div class="lg"><span class="ls" style="background:rgba(176,190,197,.4)"></span>模型边界</div>
  </div>
</div>

<!-- 底部状态栏 -->
<div class="fp" id="bar-bot">
  <div class="si"><span class="d" style="background:#4fc3f7"></span><span>数据</span><span class="n">M1+M2</span></div>
  <div class="si"><span class="d" style="background:#fdd835"></span><span>构造面</span><span class="n">{n_surf_pts}</span></div>
  <div class="si"><span class="d" style="background:#26a69a"></span><span>测点</span><span class="n">{n_wp}</span></div>
  <div class="si"><span class="d" style="background:#ef5350"></span><span>Trace</span><span class="n">{n_traces}</span></div>
  <span style="color:#37474f">│</span>
  <span style="color:#546e7a">■方块=气水井 ●圆=纯气</span>
  <span style="color:#37474f">│</span>
  <span style="color:#546e7a">第十六届中国石油工程设计大赛</span>
</div>

<!-- UI 显隐开关 -->
<button id="toggle-ui" onclick="toggleUI()" title="显示/隐藏面板">◐</button>

<script>
const P='{pid}';
const C={cam_js};
const V={vis_js};
function camTo(n,el){{Plotly.relayout(P,{{'scene.camera':C[n]}});
  document.querySelectorAll('#cg .cb').forEach(b=>b.classList.remove('on'));el.classList.add('on')}}
function layTo(n,el){{Plotly.restyle(P,{{'visible':V[n]}});
  document.querySelectorAll('#lg .cb').forEach(b=>b.classList.remove('on'));el.classList.add('on')}}
let uiOn=true;
function toggleUI(){{uiOn=!uiOn;
  document.querySelectorAll('.fp').forEach(p=>p.classList.toggle('hide',!uiOn));
  document.getElementById('toggle-ui').textContent=uiOn?'◐':'◑'}}
window.addEventListener('resize',()=>Plotly.Plots.resize(document.getElementById(P)));
setTimeout(()=>Plotly.Plots.resize(document.getElementById(P)),300);
</script>
</body>
</html>'''

out = ROOT / 'outputs' / 'mk_pinn_dt_v2' / 'wellpath_3d_viz.html'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding='utf-8')

print(f'✅ 已保存: {out}')
print(f'   文件大小: {out.stat().st_size / 1024 / 1024:.1f} MB')
print(f'   Trace 数: {n_traces}')

import webbrowser
webbrowser.open(f'file:///{out.as_posix()}')
