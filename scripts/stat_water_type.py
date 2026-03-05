"""统计附表6各井水型分布和Na/Cl指纹"""
import pandas as pd
import numpy as np

df = pd.read_csv(
    r'c:\Users\16281\Desktop\cpedc_project\data\raw\附表6-流体性质统计表__水分析.csv',
    skiprows=2, header=None, encoding='utf-8-sig'
)

well_col = 1
wtype_col = 42
tds_col = 32
na_col = 16
cl_col = 24

wells = ['SY9', 'SY13', 'SY101', 'SY102', 'SY116', 'SY201', 'SYX211']

print("=" * 85)
print("附表6 各井水型统计")
print("=" * 85)

for w in wells:
    mask = df[well_col].astype(str).str.strip() == w
    sub = df[mask]
    n = len(sub)

    wt = sub[wtype_col].astype(str).str.strip()
    c2 = wt.str.contains('CaCl2|氯化钙', case=False, na=False).sum()
    mg = wt.str.contains('MgCl2|氯化镁', case=False, na=False).sum()
    nh = wt.str.contains('NaHCO3|碳酸氢钠', case=False, na=False).sum()
    ns = wt.str.contains('Na2SO|硫酸钠', case=False, na=False).sum()
    ot = n - c2 - mg - nh - ns
    pct = c2 / n * 100 if n > 0 else 0

    tds = pd.to_numeric(sub[tds_col], errors='coerce').dropna()
    med = tds.median() if len(tds) > 0 else 0
    p90 = tds.quantile(0.9) if len(tds) > 5 else 0

    print(f"{w:8s}: n={n:4d} | CaCl2={c2:4d}({pct:5.1f}%) MgCl2={mg:3d} NaHCO3={nh:3d} Na2SO4={ns:3d} other={ot:3d} | TDS_med={med:.0f} P90={p90:.0f}")

print()
print("=" * 85)
print("高TDS(>1000 mg/L)样本 各井水型")
print("=" * 85)

nacl_means = {}
for w in wells:
    mask = df[well_col].astype(str).str.strip() == w
    sub = df[mask]
    tds = pd.to_numeric(sub[tds_col], errors='coerce')
    hi_mask = tds > 1000
    sub_hi = sub[hi_mask]
    nh_count = len(sub_hi)

    if nh_count > 0:
        wt = sub_hi[wtype_col].astype(str).str.strip()
        c2 = wt.str.contains('CaCl2|氯化钙', case=False, na=False).sum()
        pct = c2 / nh_count * 100

        # Na/Cl摩尔比
        na_v = pd.to_numeric(sub_hi[na_col], errors='coerce')
        cl_v = pd.to_numeric(sub_hi[cl_col], errors='coerce')
        valid = na_v.notna() & cl_v.notna() & (cl_v > 0)
        if valid.sum() > 2:
            ratio = (na_v[valid] / 23.0) / (cl_v[valid] / 35.45)
            ratio = ratio[(ratio > 0) & (ratio < 5)]
            rm = ratio.mean()
            rs = ratio.std()
            nacl_means[w] = rm
            nacl_str = f"Na/Cl={rm:.3f}+/-{rs:.3f} (n={len(ratio)})"
        else:
            nacl_str = "Na/Cl=N/A"
        print(f"{w:8s}: {nh_count:4d} hi-TDS, CaCl2={c2}({pct:.1f}%) {nacl_str}")
    else:
        print(f"{w:8s}: no hi-TDS samples")

print()
if len(nacl_means) >= 2:
    vals = list(nacl_means.values())
    print("Na/Cl指纹一致性:")
    for k, v in nacl_means.items():
        print(f"  {k}: {v:.4f}")
    print(f"  全场均值={np.mean(vals):.4f}, std={np.std(vals):.4f}, CV={np.std(vals)/np.mean(vals)*100:.1f}%")
