"""
Variogram模型对比: spherical / exponential / gaussian
对MK顶面和底面分别做LOO交叉验证，输出MAE/RMSE对比表。
"""
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pykrige.ok import OrdinaryKriging

mk = pd.read_csv(PROJECT_ROOT / 'data' / 'clean' / 'mk_interval_points.csv')

MODELS = ['spherical', 'exponential', 'gaussian']

def loo_cv(x, y, z, variogram_model):
    """Leave-one-out交叉验证，返回每个点的绝对误差"""
    errors = []
    d_ext = max(x.max() - x.min(), y.max() - y.min())
    
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        tx, ty, tz = x[mask], y[mask], z[mask]
        
        v_sill = float(np.var(tz)) if np.var(tz) > 0 else 1.0
        v_range = d_ext * 0.8
        v_nugget = v_sill * 0.05
        
        try:
            OK = OrdinaryKriging(
                tx, ty, tz,
                variogram_model=variogram_model,
                variogram_parameters={'sill': v_sill, 'range': v_range, 'nugget': v_nugget},
                verbose=False, enable_plotting=False
            )
            z_pred, _ = OK.execute('points', np.array([x[i]]), np.array([y[i]]))
            errors.append(abs(z_pred[0] - z[i]))
        except Exception as e:
            print(f"  ⚠ {variogram_model} LOO#{i} failed: {e}")
            errors.append(np.nan)
    
    return np.array(errors)


surfaces = {
    'MK顶面': ('x_top', 'y_top', 'mk_top_z'),
    'MK底面': ('x_bot', 'y_bot', 'mk_bot_z'),
}

results = []

for surf_name, (xc, yc, zc) in surfaces.items():
    x = mk[xc].values
    y = mk[yc].values
    z = mk[zc].values
    
    print(f"\n{'='*60}")
    print(f"  {surf_name} ({zc}): {len(x)} 个控制点")
    print(f"  数据范围: [{z.min():.1f}, {z.max():.1f}] m, std={z.std():.1f} m")
    print(f"{'='*60}")
    
    for model in MODELS:
        errs = loo_cv(x, y, z, model)
        valid = errs[~np.isnan(errs)]
        mae = np.mean(valid)
        rmse = np.sqrt(np.mean(valid**2))
        max_e = np.max(valid)
        
        results.append({
            'surface': surf_name,
            'variogram': model,
            'MAE': mae,
            'RMSE': rmse,
            'MAX': max_e,
            'n_valid': len(valid),
        })
        
        flag = ''
        print(f"  {model:12s}  MAE={mae:7.2f} m  RMSE={rmse:7.2f} m  MAX={max_e:7.2f} m")

# 汇总
print(f"\n{'='*60}")
print("  汇总对比")
print(f"{'='*60}")

df = pd.DataFrame(results)

for surf_name in surfaces:
    sub = df[df['surface'] == surf_name].copy()
    best_idx = sub['MAE'].idxmin()
    best_model = sub.loc[best_idx, 'variogram']
    print(f"\n  {surf_name} 最优模型: {best_model} (MAE={sub.loc[best_idx, 'MAE']:.2f} m)")
    for _, row in sub.iterrows():
        tag = ' ★' if row['variogram'] == best_model else ''
        print(f"    {row['variogram']:12s}  MAE={row['MAE']:7.2f}  RMSE={row['RMSE']:7.2f}{tag}")

# 保存CSV
out_path = PROJECT_ROOT / 'outputs' / 'mk_pinn_dt_v2' / 'variogram_model_comparison.csv'
df.to_csv(out_path, index=False)
print(f"\n✅ 结果已保存: {out_path}")
