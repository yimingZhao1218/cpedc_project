"""
SY9 井口压力观测缺失自检脚本
读取 production_SY9.csv，输出非空数、缺失率、缺失段清单。
用于 CI/本地验收，与 PINNSampler 中缺失审计逻辑一致。
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

OBS_PRESSURE_COL = 'tubing_p_avg'


def compute_missing_runs(valid_mask: np.ndarray, df: pd.DataFrame, t_days: np.ndarray) -> list:
    """连续缺失区间 (start_idx, end_idx, start_date, end_date, length_days)."""
    missing_runs = []
    in_run = False
    start_idx = None
    dates = df['date'] if 'date' in df.columns else None
    n = len(valid_mask)
    for i in range(n):
        if not valid_mask[i]:
            if not in_run:
                in_run = True
                start_idx = i
        else:
            if in_run:
                end_idx = i - 1
                start_date = str(dates.iloc[start_idx]) if dates is not None else str(start_idx)
                end_date = str(dates.iloc[end_idx]) if dates is not None else str(end_idx)
                length_days = int(round(t_days[end_idx] - t_days[start_idx])) + 1 if end_idx >= start_idx else 1
                missing_runs.append((start_idx, end_idx, start_date, end_date, length_days))
                in_run = False
    if in_run:
        end_idx = n - 1
        start_date = str(dates.iloc[start_idx]) if dates is not None else str(start_idx)
        end_date = str(dates.iloc[end_idx]) if dates is not None else str(end_idx)
        length_days = int(round(t_days[end_idx] - t_days[start_idx])) + 1 if end_idx >= start_idx else 1
        missing_runs.append((start_idx, end_idx, start_date, end_date, length_days))
    return missing_runs


def main(csv_path: str) -> int:
    if not os.path.isfile(csv_path):
        print(f"错误: 文件不存在 {csv_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(csv_path)
    if OBS_PRESSURE_COL not in df.columns:
        print(f"错误: 未找到列 '{OBS_PRESSURE_COL}'", file=sys.stderr)
        return 1
    t_days = df['t_day'].values.astype(np.float64)
    p = np.asarray(df[OBS_PRESSURE_COL].values, dtype=np.float64)
    valid_mask = np.isfinite(p) & ~np.isnan(p)
    total = len(p)
    valid_count = int(np.sum(valid_mask))
    missing_count = total - valid_count
    nan_ratio = missing_count / total if total else 0.0
    missing_runs = compute_missing_runs(valid_mask, df, t_days)

    print("SY9 井口压力观测缺失自检 (tubing_p_avg)")
    print("=" * 50)
    print(f"文件: {csv_path}")
    print(f"总点数: {total}")
    print(f"非空数: {valid_count}")
    print(f"缺失数: {missing_count}")
    print(f"缺失率: {nan_ratio:.2%}")
    print()
    if missing_runs:
        print("缺失段清单 (起止日期, 长度/天):")
        for run in missing_runs:
            start_idx, end_idx, start_date, end_date, length_days = run
            print(f"  [{start_date} ~ {end_date}] 长度 {length_days} 天 (idx {start_idx}~{end_idx})")
        longest = max(missing_runs, key=lambda r: r[4])
        print(f"\n最长缺失段: {longest[2]} ~ {longest[3]}, {longest[4]} 天")
    else:
        print("无连续缺失段。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SY9 压力观测缺失自检")
    default_path = Path(__file__).resolve().parent.parent / "data" / "clean" / "production_SY9.csv"
    parser.add_argument("csv", nargs="?", default=str(default_path), help="production_SY9.csv 路径")
    args = parser.parse_args()
    sys.exit(main(args.csv))
