#!/usr/bin/env python3
"""
Corey 指数拟合脚本
==================
用附表7 (SY13 MK组) 的 21 个气水相渗实验点，
最小二乘拟合 Corey-Brooks 模型的 ng 和 nw。

拟合模型:
    krg(Sw) = krg_max · Se_g^ng
    krw(Sw) = krw_max · Se_w^nw

    其中:
        Se_g = (1 - Sw - Sgr) / (1 - Swc - Sgr)
        Se_w = (Sw - Swc) / (1 - Swc - Sgr)

结果:
    ng = 1.0846  (R² = 0.9945, RMSE = 0.0162)
    nw = 4.4071  (R² = 0.9823, RMSE = 0.0169)

用途:
    - src/pinn/torch_physics.py  TorchRelPerm 类
    - src/physics/relperm_validate.py  M3 出图

运行:
    python scripts/fit_corey_exponents.py

作者: CPEDC 2026 创新组
日期: 2026-02-22
"""

import numpy as np
from scipy.optimize import curve_fit


def main():
    # ========== 1. 附表7 原始数据 ==========
    # SY13井, MK组, 孔隙度 2.16%, 绝对渗透率 3.20×10⁻³ μm²
    sw_pct = np.array([
        26, 29.4, 32.8, 36.2, 39.6, 43, 46.4, 49.7,
        53.1, 56.5, 59.9, 63.3, 66.7, 70.1, 73.5,
        76.9, 80.3, 83.7, 87.1, 90.4, 93.8
    ])
    krg_data = np.array([
        0.675, 0.655, 0.628, 0.587, 0.547, 0.522, 0.485, 0.445,
        0.400, 0.359, 0.315, 0.273, 0.238, 0.204, 0.169,
        0.134, 0.102, 0.068, 0.043, 0.022, 0.009
    ])
    krw_data = np.array([
        0.000, 0.008, 0.010, 0.013, 0.014, 0.018, 0.023, 0.028,
        0.034, 0.039, 0.046, 0.053, 0.072, 0.086, 0.105,
        0.131, 0.165, 0.215, 0.280, 0.369, 0.480
    ])

    sw = sw_pct / 100.0
    n_pts = len(sw)

    # ========== 2. 端点参数 (直接从数据提取) ==========
    Swc = sw[0]            # 0.260  束缚水饱和度
    Sgr = 1.0 - sw[-1]    # 0.062  残余气饱和度
    krg_max = krg_data[0]  # 0.675  最大气相相渗
    krw_max = krw_data[-1] # 0.480  最大水相相渗
    denom = 1.0 - Swc - Sgr  # 0.678

    print("=" * 60)
    print("Corey 指数拟合 (附表7 SY13 MK组)")
    print("=" * 60)
    print(f"\n数据点数: {n_pts}")
    print(f"端点参数:")
    print(f"  Swc (束缚水饱和度)   = {Swc:.4f}")
    print(f"  Sgr (残余气饱和度)   = {Sgr:.4f}")
    print(f"  krg_max              = {krg_max:.4f}")
    print(f"  krw_max              = {krw_max:.4f}")
    print(f"  归一化分母 (1-Swc-Sgr) = {denom:.4f}")

    # ========== 3. 归一化饱和度 ==========
    Se_g = np.clip((1.0 - sw - Sgr) / denom, 1e-8, 1.0)
    Se_w = np.clip((sw - Swc) / denom, 1e-8, 1.0)

    # ========== 4. 拟合 ng (气相 Corey 指数) ==========
    mask_g = (Se_g > 0.01) & (Se_g < 0.99)

    def corey_g(Se, n):
        return krg_max * Se ** n

    popt_g, pcov_g = curve_fit(corey_g, Se_g[mask_g], krg_data[mask_g], p0=[2.0])
    ng_fit = popt_g[0]
    ng_std = np.sqrt(pcov_g[0, 0])

    krg_fit = krg_max * Se_g ** ng_fit
    rmse_g = np.sqrt(np.mean((krg_data - krg_fit) ** 2))
    ss_res_g = np.sum((krg_data - krg_fit) ** 2)
    ss_tot_g = np.sum((krg_data - np.mean(krg_data)) ** 2)
    r2_g = 1.0 - ss_res_g / ss_tot_g

    print(f"\n--- 气相 Corey 指数拟合 ---")
    print(f"  ng = {ng_fit:.4f} ± {ng_std:.4f}")
    print(f"  R² = {r2_g:.6f}")
    print(f"  RMSE = {rmse_g:.6f}")
    print(f"  拟合使用 {mask_g.sum()}/{n_pts} 个点 (排除端点)")

    # ========== 5. 拟合 nw (水相 Corey 指数) ==========
    mask_w = Se_w > 0.01

    def corey_w(Se, n):
        return krw_max * Se ** n

    popt_w, pcov_w = curve_fit(corey_w, Se_w[mask_w], krw_data[mask_w], p0=[3.0])
    nw_fit = popt_w[0]
    nw_std = np.sqrt(pcov_w[0, 0])

    krw_fit = krw_max * Se_w ** nw_fit
    rmse_w = np.sqrt(np.mean((krw_data - krw_fit) ** 2))
    ss_res_w = np.sum((krw_data - krw_fit) ** 2)
    ss_tot_w = np.sum((krw_data - np.mean(krw_data)) ** 2)
    r2_w = 1.0 - ss_res_w / ss_tot_w

    print(f"\n--- 水相 Corey 指数拟合 ---")
    print(f"  nw = {nw_fit:.4f} ± {nw_std:.4f}")
    print(f"  R² = {r2_w:.6f}")
    print(f"  RMSE = {rmse_w:.6f}")
    print(f"  拟合使用 {mask_w.sum()}/{n_pts} 个点 (排除 Sw=Swc)")

    # ========== 6. 汇总 ==========
    print(f"\n{'=' * 60}")
    print(f"拟合结果汇总:")
    print(f"  ng = {ng_fit:.4f}  (旧值 2.0, 差异 {abs(ng_fit - 2.0):.2f})")
    print(f"  nw = {nw_fit:.4f}  (旧值 3.0, 差异 {abs(nw_fit - 3.0):.2f})")
    print(f"{'=' * 60}")
    print(f"\n已更新到:")
    print(f"  src/pinn/torch_physics.py  → TorchRelPerm.ng / .nw")
    print(f"  src/physics/relperm_validate.py → plot_curves() ng / nw")


if __name__ == '__main__':
    main()
