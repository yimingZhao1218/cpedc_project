"""
Z 因子最小二乘拟合过程（可复现）
====================================
从附表5-2 油藏流体恒质膨胀数据表读取 (p, Z) 实测数据，
对每个温度列做**三次**多项式最小二乘拟合:
  Z(p) = a0 + a1·p + a2·p² + a3·p³
输出各温度的 a0, a1, a2, a3 与 RMSE。最终 torch_physics.py 采用此三次拟合。

数据来源: data/raw/附表5-PVT数据__2油藏流体恒质膨胀数据表.csv
拟合方法: numpy.polyfit(p, Z, deg=3) — 最小二乘
参考: M1_M4_Code_Audit_Report.md / M1_M4_Audit_Fix_Report.md

运行: 在项目根目录执行
  python scripts/fit_z_factor_least_squares.py
  或
  python scripts/fit_z_factor_least_squares.py data/raw/附表5-PVT数据__2油藏流体恒质膨胀数据表.csv
"""

import os
import re
import sys
import numpy as np

# 默认路径（项目根 = 上两级）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV = os.path.join(
    PROJECT_ROOT,
    "data", "raw", "附表5-PVT数据__2油藏流体恒质膨胀数据表.csv"
)


def parse_cce_csv(filepath: str):
    """
    解析附表5-2 恒质膨胀数据表。
    返回: temperatures (list[float]), pressures (array), data_by_T (dict T -> Z array)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 第2行: 温度标签 ,16.5℃,46.5℃,78.0℃,109.0℃,140.32℃
    temp_line = lines[1]
    temp_matches = re.findall(r"([\d.]+)℃", temp_line)
    temperatures = [float(t) for t in temp_matches]
    n_temps = len(temperatures)

    pressures = []
    data_by_temp = {T: [] for T in temperatures}

    for line in lines[2:]:
        line = line.strip()
        if not line or line.startswith("（") or line.startswith("("):
            continue
        parts = line.split(",")
        p_str = parts[0].strip().lstrip("*")
        try:
            p = float(p_str)
        except ValueError:
            continue
        values = []
        for i in range(1, n_temps + 1):
            if i < len(parts):
                try:
                    values.append(float(parts[i].strip()))
                except ValueError:
                    values.append(np.nan)
            else:
                values.append(np.nan)
        if len(values) == n_temps and not any(np.isnan(values)):
            pressures.append(p)
            for j, T in enumerate(temperatures):
                data_by_temp[T].append(values[j])

    pressures = np.array(pressures, dtype=np.float64)
    data_by_temp = {T: np.array(v, dtype=np.float64) for T, v in data_by_temp.items()}
    return temperatures, pressures, data_by_temp


def fit_z_third_order(p: np.ndarray, z: np.ndarray):
    """
    三次最小二乘拟合 Z(p) = a0 + a1*p + a2*p^2 + a3*p^3。
    polyfit 返回系数从高次到低次: [a3, a2, a1, a0]。
    """
    coefs = np.polyfit(p, z, deg=3)  # [a3, a2, a1, a0]
    a3, a2, a1, a0 = coefs[0], coefs[1], coefs[2], coefs[3]
    z_fit = a0 + a1 * p + a2 * (p ** 2) + a3 * (p ** 3)
    rmse = np.sqrt(np.mean((z - z_fit) ** 2))
    return a0, a1, a2, a3, rmse


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.isfile(csv_path):
        print(f"错误: 文件不存在 {csv_path}")
        sys.exit(1)

    print("=" * 70)
    print("Z factor least-squares fit (Appendix 5-2 CCE) -- CUBIC")
    print("=" * 70)
    print(f"Data file: {csv_path}")
    print("Model: Z(p) = a0 + a1*p + a2*p^2 + a3*p^3")
    print("Method: numpy.polyfit(p, Z, deg=3)")
    print()

    temperatures, pressures, data_by_temp = parse_cce_csv(csv_path)
    print(f"Pressure: {pressures.min():.1f} ~ {pressures.max():.1f} MPa, n={len(pressures)}")
    print(f"Temperatures: {temperatures}")
    print()

    print("Fit results (cubic):")
    print("-" * 90)
    print(f"{'T(C)':<8} {'a3':<14} {'a2':<14} {'a1':<14} {'a0':<10} {'RMSE':<10}")
    print("-" * 90)

    results = []
    for T in temperatures:
        z = data_by_temp[T]
        a0, a1, a2, a3, rmse = fit_z_third_order(pressures, z)
        results.append((T, a0, a1, a2, a3, rmse))
        print(f"{T:<8.2f} {a3:<14.4e} {a2:<14.4e} {a1:<14.4e} {a0:<10.4f} {rmse:<10.5f}")

    print("-" * 90)
    print()

    # T=140.32C coefficients -> torch_physics.py (used in code)
    T_res = 140.32
    for r in results:
        if abs(r[0] - T_res) < 0.01:
            _, a0, a1, a2, a3, rmse = r
            print("T=140.32C (reservoir) -- copy to torch_physics.py:")
            print(f"  self.z_a0 = {a0}")
            print(f"  self.z_a1 = {a1}")
            print(f"  self.z_a2 = {a2}")
            print(f"  self.z_a3 = {a3}")
            print(f"  # RMSE = {rmse:.6f}")
            break

    print()
    print("Full fitting process: outputs/mk_pinn_dt_v2/reports/M1_M4_Audit_Fix_Report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
