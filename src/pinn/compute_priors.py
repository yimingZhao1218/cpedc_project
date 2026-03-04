"""
数据驱动先验值自动计算模块
===========================
从附表3测井数据 + 附表4分层数据自动计算渗透率先验，
替代人工指定，实现全流程数据驱动闭环。

方法：MK层段厚度加权几何均值 + 试油反推裂缝因子

数据文件对应关系（经逐一核实项目实际文件）：
    附表4分层数据.csv             → 7口井 MK 顶/底界 MD (测量深度)
    附表3测井数据__SY9.csv        → Depth + PERM 列, 有效
    附表3测井数据__SY13.csv       → Depth + PERM 列, 有效
    附表3测井数据__SY201.csv      → Depth + PERM 列, 有效
    附表3测井数据__SY101.csv      → Depth + PERM 列, 含 -9999 哨兵值
    附表3测井数据__SY102.csv      → Depth + PERM 列, 含 -9999 哨兵值
    附表3测井数据__SY116.csv      → Depth + PERM 列, 含 -9999, 无 TVD 列
    附表3测井数据__SYX211.csv     → PERM 全为 -9999 或 0, 自动跳过

注意:
    1. 附表4 的 MK顶界钻井深度/MK底界钻井深度 是 MD (测量深度)
    2. 附表3 的 Depth 列也是 MD → 二者直接匹配
    3. 不需要 TVD 转换
"""


import os
import math
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger('compute_priors')

# ================================================================
# 项目中附表3的精确文件名映射（与 data/raw 一致，带短横线 附表3-）
# 文件名格式: 附表3-测井数据__<井号>.csv (双下划线)
# ================================================================
WELL_LOG_FILES = {
    'SY9':    '附表3-测井数据__SY9.csv',
    'SY13':   '附表3-测井数据__SY13.csv',
    'SY201':  '附表3-测井数据__SY201.csv',
    'SY101':  '附表3-测井数据__SY101.csv',
    'SY102':  '附表3-测井数据__SY102.csv',
    'SY116':  '附表3-测井数据__SY116.csv',
    'SYX211': '附表3-测井数据__SYX211.csv',
}

# 各文件的 PERM 数据质量备注
WELL_LOG_NOTES = {
    'SY9':    '750/1601 有效, max=22.02 mD',
    'SY13':   '700/1601 有效, max=2.02 mD, 无 -9999',
    'SY201':  '648/1385 有效, max=12.63 mD, 含 -9999',
    'SY101':  '1106/1713 有效, max=14.58 mD, 含 -9999',
    'SY102':  '1102/1601 有效, max=6.51 mD, 含 -9999',
    'SY116':  '882/2001 有效, max=2.55 mD, 含 -9999, 无TVD列',
    'SYX211': 'PERM 全无效 (-9999 或 0), 自动跳过',
}

# PERM 哨兵值阈值：小于此值视为无效（附表3 中 -9999 为缺失标记）
PERM_SENTINEL_THRESHOLD = -999.0


def compute_permeability_prior(
    layer_csv: str = '附表4-分层数据.csv',
    data_dir: str = '.',
) -> Tuple[float, Dict[str, float], List[float]]:
    """
    从附表3测井数据 + 附表4分层数据自动计算渗透率先验。

    方法：MK层段厚度加权几何均值

    处理细节：
        1. 附表4 的 MK顶界钻井深度(m) / MK底界钻井深度(m) 是 MD
        2. 附表3 的 Depth 列也是 MD → 直接匹配，无需 TVD 转换
        3. PERM 列中 -9999 为哨兵值（SY101/SY102/SY116/SYX211），需过滤
        4. SYX211 的 PERM 全为 -9999 或 0，自动跳过
        5. PERM ≤ 0 的点也跳过（几何均值需要正值）

    附表4 各井 MK 层段 (MD, m):
        SY9:    4543.0 – 4635.2   (h=92.2 m)
        SY13:   4569.0 – 4663.5   (h=94.5 m)
        SY201:  4541.0 – 4632.5   (h=91.5 m)
        SY101:  4592.5 – 4674.5   (h=82.0 m)
        SY102:  4612.0 – 4709.7   (h=97.7 m)
        SY116:  4622.4 – 4708.6   (h=86.2 m)
        SYX211: 4925.9 – 5099.2   (h=173.4 m, MD >> 其他井因为大斜度)

    Args:
        layer_csv: 附表4分层数据文件名
        data_dir:  数据文件所在目录

    Returns:
        k_geo_weighted: 厚度加权几何均值 (mD)
        k_per_well: {井号: 几何均值(mD)} 字典
        k_bounds: [P10, P90] 区间
    """
    layer_path = os.path.join(data_dir, layer_csv)
    layers = pd.read_csv(layer_path, encoding='utf-8-sig')

    all_h = []       # 各井 MK 厚度
    all_logk = []    # 各井 ln(k_geo)
    k_per_well = {}  # 各井几何均值
    all_perm_values = []  # 所有有效 PERM 值（用于整体统计）

    for _, row in layers.iterrows():
        well = str(row['井号']).strip()
        md_top = float(row['MK顶界钻井深度（m）'])
        md_bot = float(row['MK底界钻井深度（m）'])
        h_mk = md_bot - md_top

        # 查找对应的附表3文件
        if well not in WELL_LOG_FILES:
            logger.warning(f"  {well}: 未在 WELL_LOG_FILES 映射中，跳过")
            continue

        log_path = os.path.join(data_dir, WELL_LOG_FILES[well])
        if not os.path.exists(log_path):
            logger.warning(f"  {well}: 文件不存在 {log_path}，跳过")
            continue

        # 读取测井数据
        df = pd.read_csv(log_path, encoding='utf-8-sig')

        # 查找 Depth 列（兼容 BOM 和前导空格）
        depth_col = None
        for col in df.columns:
            if col.strip().lower() == 'depth':
                depth_col = col
                break
        if depth_col is None:
            logger.warning(f"  {well}: 无 Depth 列，跳过")
            continue

        # 查找 PERM 列
        perm_col = None
        for col in df.columns:
            if col.strip().upper() == 'PERM':
                perm_col = col
                break
        if perm_col is None:
            logger.warning(f"  {well}: 无 PERM 列，跳过")
            continue

        # 筛选 MK 层段 (附表4 MD 与 附表3 Depth 直接匹配)
        mask_depth = (df[depth_col] >= md_top) & (df[depth_col] <= md_bot)
        perm_raw = df.loc[mask_depth, perm_col].copy()

        # 过滤无效值:
        # - NaN
        # - -9999 哨兵值 (SY101/SY102/SY116/SYX211)
        # - ≤0 (几何均值需正值)
        perm_valid = perm_raw.dropna()
        perm_valid = perm_valid[perm_valid > PERM_SENTINEL_THRESHOLD]
        perm_valid = perm_valid[perm_valid > 0]

        n_total = mask_depth.sum()
        n_valid = len(perm_valid)

        if n_valid == 0:
            logger.warning(
                f"  {well}: MK[{md_top:.1f}-{md_bot:.1f}m] "
                f"共 {n_total} 个深度点, 有效 PERM = 0, 跳过 "
                f"(备注: {WELL_LOG_NOTES.get(well, '')})"
            )
            continue

        # 几何均值 = exp(mean(ln(k)))
        log_perm = np.log(perm_valid.values)
        k_geo = np.exp(log_perm.mean())
        k_arith = perm_valid.mean()
        k_p50 = perm_valid.median()
        k_p10 = perm_valid.quantile(0.1)
        k_p90 = perm_valid.quantile(0.9)

        k_per_well[well] = k_geo
        all_h.append(h_mk)
        all_logk.append(np.log(k_geo))
        all_perm_values.extend(perm_valid.values.tolist())

        logger.info(
            f"  {well}: MK[{md_top:.1f}-{md_bot:.1f}m], h={h_mk:.1f}m, "
            f"N={n_valid}/{n_total}, "
            f"k_geo={k_geo:.4f}, k_arith={k_arith:.4f}, "
            f"P10={k_p10:.4f}, P50={k_p50:.4f}, P90={k_p90:.4f} mD"
        )

    if len(all_h) == 0:
        logger.error("  无任何有效井数据! 返回默认值 0.1 mD")
        return 0.1, {}, [0.01, 1.0]

    # 厚度加权几何均值: k_geo_w = exp(Σ(h_i · ln(k_geo,i)) / Σh_i)
    all_h = np.array(all_h)
    all_logk = np.array(all_logk)
    k_geo_weighted = np.exp(np.sum(all_h * all_logk) / np.sum(all_h))

    # 搜索范围（基于各井几何均值的分位数）
    k_values = np.array(list(k_per_well.values()))
    k_bounds = [float(np.percentile(k_values, 10)),
                float(np.percentile(k_values, 90))]

    # 整体统计（所有有效点）
    all_perm = np.array(all_perm_values)
    overall_geo = np.exp(np.log(all_perm).mean())
    overall_p90 = np.percentile(all_perm, 90)
    overall_p95 = np.percentile(all_perm, 95)

    logger.info("=" * 60)
    logger.info(f"  厚度加权几何均值: {k_geo_weighted:.4f} mD")
    logger.info(f"  参与统计井数: {len(k_per_well)} / 7")
    logger.info(f"  各井 P10={k_bounds[0]:.4f}, P90={k_bounds[1]:.4f} mD")
    logger.info(f"  全部点几何均值: {overall_geo:.4f} mD (共 {len(all_perm)} 点)")
    logger.info(f"  全部点 P90={overall_p90:.4f}, P95={overall_p95:.4f} mD")
    logger.info("=" * 60)

    return k_geo_weighted, k_per_well, k_bounds


def compute_frac_factor_from_DST(
    k_m_mD: float,
    q_surface_m3d: float = 568663.0,     # 附表9: SY9 日产气量 (m³/d)
    pe_MPa: float = 74.875,               # 附表9: SY9 井底静压 (MPa)
    pwf_MPa: float = 71.226,              # 附表9: SY9 井底流压 (MPa)
    h_m: float = 48.4,                    # 附表8: SY9 有效厚度 16.3+32.1 m
    mu_mPa_s: float = 0.035,              # Lee-Gonzalez-Eakin 估算
    bg: float = 0.002577,                 # 附表5-4: Bg(75.7MPa, 140.32°C)
    r_e_m: float = 500.0,
    r_w_m: float = 0.1,
    skin: float = 0.0,
) -> Tuple[float, float]:
    """
    从 SY9 试油数据反推裂缝导流增强因子。

    Darcy 稳态径向流:
        q_res = 2π k_eff h Δp / (μ (ln(r_e/r_w) + s))
        k_eff = q_res · μ · (ln(r_e/r_w) + s) / (2π h Δp)
        f = k_eff / k_m

    注意:
        Bg = 0.002577 (附表5-4 实测), 不是 0.004!
        附表9 中 pe-pwf = 3.649 MPa 与表中"压差"2.770 MPa 不一致，
        差异可能来自测压深度 4400m vs 射孔中部 4549m 的气柱修正。
        两个压差都算，给出 k_eff 区间。

    Returns:
        k_eff_mD: 等效渗透率 (mD)
        f_frac: 裂缝增效因子 = k_eff / k_m
    """
    dp_Pa = (pe_MPa - pwf_MPa) * 1e6       # Pa
    mu_Pa_s = mu_mPa_s * 1e-3              # Pa·s
    q_res_m3s = (q_surface_m3d / 86400.0) * bg  # m³/s (地层条件)

    ln_ratio = math.log(r_e_m / r_w_m) + skin

    # Darcy 反推: k = q·μ·ln(re/rw) / (2π·h·Δp)
    k_eff_SI = q_res_m3s * mu_Pa_s * ln_ratio / (2 * math.pi * h_m * dp_Pa)
    k_eff_mD = k_eff_SI / 9.869233e-16     # m² → mD

    f_frac = k_eff_mD / k_m_mD if k_m_mD > 0 else 1.0

    logger.info(
        f"  试油反推: Δp={pe_MPa-pwf_MPa:.3f}MPa, Bg={bg:.6f}, "
        f"q_res={q_res_m3s:.6e} m³/s, "
        f"k_eff={k_eff_mD:.2f} mD, f={f_frac:.1f}"
    )

    return k_eff_mD, f_frac


def compute_all_priors(data_dir: str = '.') -> dict:
    """
    一键计算全部先验值。

    答辩演示用：展示"全流程数据驱动"闭环。
    
    执行流程:
        1. 从附表3+4 计算 MK 基质渗透率 k_m
        2. 尺度效应修正 ×3 (实验室→油藏)
        3. 从附表9 试油数据反推 k_eff → f = k_eff / k_m
        4. 用两个压差值给出 f 的区间

    Returns:
        dict: 包含 k_m, f_frac, 推荐 config 更新等
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("=" * 60)
    logger.info("  数据驱动先验值自动计算")
    logger.info("  数据来源: 附表3(7个文件) + 附表4 + 附表5-4 + 附表9")
    logger.info("=" * 60)

    # Step 1: 测井渗透率先验
    logger.info("\n[Step 1] 从附表3+4 计算 MK 基质渗透率...")
    k_geo, k_per_well, k_bounds = compute_permeability_prior(
        layer_csv='附表4-分层数据.csv',
        data_dir=data_dir,
    )

    # Step 2: 尺度效应修正
    scale_factor = 3.0
    k_m = k_geo * scale_factor
    logger.info(
        f"\n[Step 2] 尺度效应修正:"
        f"\n  k_geo = {k_geo:.4f} mD"
        f"\n  × {scale_factor} (实验室→油藏, 含微裂缝) = k_m = {k_m:.4f} mD"
    )

    # Step 3: 试油反推裂缝因子（两个压差）
    logger.info("\n[Step 3] 从 SY9 试油数据反推 f_frac...")

    logger.info("  场景A: 用 pe-pwf = 3.649 MPa (直接相减)")
    k_eff_A, f_A = compute_frac_factor_from_DST(
        k_m_mD=k_m, pe_MPa=74.875, pwf_MPa=71.226
    )

    logger.info("  场景B: 用表中压差 = 2.770 MPa (可能含深度修正)")
    k_eff_B, f_B = compute_frac_factor_from_DST(
        k_m_mD=k_m, pe_MPa=74.875, pwf_MPa=74.875-2.7697
    )

    f_mid = (f_A + f_B) / 2.0

    result = {
        'k_geo_weighted_mD': round(k_geo, 4),
        'scale_factor': scale_factor,
        'k_m_mD': round(k_m, 3),
        'k_eff_SY9_A_mD': round(k_eff_A, 2),
        'k_eff_SY9_B_mD': round(k_eff_B, 2),
        'f_frac_A': round(f_A, 1),
        'f_frac_B': round(f_B, 1),
        'f_frac_mid': round(f_mid, 0),
        'k_per_well': {w: round(v, 4) for w, v in k_per_well.items()},
        'k_bounds': [round(b, 4) for b in k_bounds],
        'wells_used': list(k_per_well.keys()),
        'wells_skipped': [w for w in WELL_LOG_FILES if w not in k_per_well],
        'recommended_config': {
            'k_eff_mD': round(k_m, 3),
            'k_eff_bounds': [0.001, 100.0],
            'frac_conductivity_factor': round(f_mid, 0),
            'frac_bounds': [1.0, 1000.0],
        },
        'data_sources': {
            'layer_data': '附表4-分层数据.csv (7口井 MK 顶/底界 MD)',
            'well_logs': list(WELL_LOG_FILES.values()),
            'bg': '附表5-4: Bg(75.7MPa, 140.32°C) = 0.002577 m³/sm³',
            'dst': '附表9: SY9 q=568663 m³/d, pe=74.875 MPa, pwf=71.226 MPa',
            'viscosity': 'Lee-Gonzalez-Eakin: μ=0.035 mPa·s',
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info(f"  推荐 config 修改:")
    logger.info(f"    k_eff_mD:                  {k_m:.3f}")
    logger.info(f"    frac_conductivity_factor:   {f_mid:.0f}")
    logger.info(f"    验证: {k_m:.3f} × {f_mid:.0f} = {k_m*f_mid:.2f} mD (Peaceman WI)")
    logger.info(f"    试油反推范围:               {k_eff_A:.2f} ~ {k_eff_B:.2f} mD")
    logger.info(f"    使用井数: {len(k_per_well)}/7, 跳过: {result['wells_skipped']}")
    logger.info("=" * 60)

    return result


# ================================================================
# 命令行入口
# ================================================================
if __name__ == '__main__':
    import sys
    import json
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    result = compute_all_priors(data_dir=data_dir)

    # 输出 JSON
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False, default=str))
