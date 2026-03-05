"""
M7-A: NSGA-II 多目标优化引擎 (PINN-as-Surrogate)
===================================================
v4.7 2026-03-04

架构: Phase 1 PINN推理(1次~2s) -> cache -> Phase 2 NSGA-II(3000次, 每次~0.1ms)
决策变量(4维): dp_stage1[0,5] dp_stage2[0,10] t_switch[0.3,0.7] ramp_days[30,180]
目标(3个minimize): -Gp, Sw_end, -NPV
Sw来源: TDS标定BL递推, dp_ratio驱动
"""

import os, time, math, logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _corey_fw_static(Sw, Swc, Sgr, nw, ng, krw_max, krg_max, mu_w, mu_g):
    Se = np.clip((Sw - Swc) / (1.0 - Swc - Sgr), 0, 1)
    krw = krw_max * np.power(Se, nw)
    krg = krg_max * np.power(1.0 - Se, ng)
    mob_w = krw / mu_w
    mob_g = krg / mu_g
    return mob_w / (mob_w + mob_g + 1e-15)


def _corey_dfw_static(Sw, Swc, Sgr, nw, ng, krw_max, krg_max, mu_w, mu_g):
    eps = 1e-4
    fh = _corey_fw_static(Sw + eps, Swc, Sgr, nw, ng, krw_max, krg_max, mu_w, mu_g)
    fl = _corey_fw_static(Sw - eps, Swc, Sgr, nw, ng, krw_max, krg_max, mu_w, mu_g)
    return (fh - fl) / (2 * eps)


def compute_sw_bl_static(sw_base, dp_mod, dp_base, data_end_idx,
                          Sw_0, lambda_BL, dt_step,
                          Swc, Sgr, nw, ng, krw_max, krg_max, mu_w, mu_g):
    """TDS标定BL递推Sw (对应_compute_sw_nonlinear模式A)"""
    DFW_FLOOR = 0.05
    n = len(sw_base)
    sw = sw_base.copy()
    if Sw_0 is not None and lambda_BL is not None and data_end_idx < n:
        for i in range(data_end_idx + 1, n):
            dp_r = float(dp_mod[i]) / (float(dp_base[i]) + 1e-10)
            dfw = max(_corey_dfw_static(
                np.array([sw[i-1]]), Swc, Sgr, nw, ng,
                krw_max, krg_max, mu_w, mu_g).item(), DFW_FLOOR)
            sw[i] = sw[i-1] + max(lambda_BL * dt_step * dp_r * dfw, 0.0)
    return np.clip(sw, Swc, 1.0 - Sgr)


def build_pwf_schedule(x, cache):
    """决策变量[dp1,dp2,t_sw_frac,ramp_d] -> pwf_mod(n_time)"""
    dp1, dp2, tsf, rd = x
    pwf = cache['pwf_base'].copy()
    i0 = cache['data_end_idx']
    tfc = cache['t_days'][i0:] - cache['t_days'][i0]
    if len(tfc) == 0:
        return pwf
    tsw = tsf * tfc[-1]
    sig = 1.0 / (1.0 + np.exp(-(tfc - tsw) / max(float(rd), 1.0)))
    pwf[i0:] += float(dp1) * (1.0 - sig) + float(dp2) * sig
    return pwf


def nsga2_evaluate(x, cache):
    """单次评估(~0.1ms) -> [f1=-Gp/1e6, f2=Sw_end, f3=-NPV/1e6]"""
    pwf = build_pwf_schedule(x, cache)
    dp = np.maximum(cache['p_cell'] - pwf, 0.1)
    qg = cache['qg_base'] * (dp / cache['dp_base'])
    qg[cache['shutin']] = 0.0
    dt = cache['dt_days']
    Gp = np.sum(qg * dt)

    sw = compute_sw_bl_static(
        cache['sw_base'], dp, cache['dp_base'], cache['data_end_idx'],
        cache['Sw_0'], cache['lambda_BL'], cache['dt_step'],
        cache['Swc'], cache['Sgr'], cache['nw'], cache['ng'],
        cache['krw_max'], cache['krg_max'], cache['mu_w'], cache['mu_g'])

    fw = _corey_fw_static(sw, cache['Swc'], cache['Sgr'],
                           cache['nw'], cache['ng'],
                           cache['krw_max'], cache['krg_max'],
                           cache['mu_w'], cache['mu_g'])
    qw = np.clip(qg * fw / (1.0 - fw + 1e-10), 0, qg * 2)  # v4.7: 5→2 与water_invasion.py统一
    qw_e = qw.copy()
    qw_e[:cache['data_end_idx']] = 0.0
    disc = cache['discount_factors']
    npv = np.sum(qg * dt * cache['gas_price'] * disc) - np.sum(qw_e * dt * cache['water_cost'] * disc)
    return [-Gp / 1e6, float(sw[-1]), -npv / 1e6]


def build_evaluation_cache(analyzer, well_id='SY9', n_time=500, bg_ref=0.002577):
    """Phase 1: 一次PINN推理 -> cache dict"""
    logger.info(f"NSGA-II Phase1: cache (well={well_id}, n={n_time})")
    m = analyzer.model
    s = analyzer.sampler
    dev = analyzer.device
    m.eval()

    tn = np.linspace(0, 1, n_time).astype(np.float32)
    td = tn * s.t_max
    dtd = np.diff(td, prepend=0)
    dts = s.t_max / max(n_time - 1, 1)
    hw = analyzer.well_h.get(well_id, 90.0)
    dei = int(n_time * analyzer.train_frac)

    wm = s.well_ids == well_id
    if not np.any(wm):
        raise ValueError(f"井 {well_id} 不在sampler中")
    wx, wy = s.well_xy[wm][0]
    xn, yn = s.normalize_xy(np.array([wx]), np.array([wy]))
    xyt = np.stack([np.full(n_time, xn[0]), np.full(n_time, yn[0]), tn], -1).astype(np.float32)
    xyt_t = torch.from_numpy(xyt).to(dev)

    with torch.no_grad():
        res = m.evaluate_at_well(well_id, xyt_t, h_well=hw, bg_val=bg_ref)

    qgb = res['qg'].cpu().numpy().flatten()
    pwfb = res['p_wf'].cpu().numpy().flatten()
    pc = res['p_cell'].cpu().numpy().flatten()

    shut = np.zeros(n_time, dtype=bool)
    wd = s.sample_well_data(well_id)
    if wd is not None:
        to = wd['t_days']
        qo = wd['qg_obs']
        if hasattr(to, 'cpu'): to = to.cpu().numpy().flatten()
        if hasattr(qo, 'cpu'): qo = qo.cpu().numpy().flatten()
        shut = np.interp(td, to, (np.abs(qo) <= 1.0).astype(float)) > 0.5
        qgb[shut] = 0.0

    swtds = analyzer._compute_tds_sw_timeseries(well_id, td)
    swb = swtds if swtds is not None else np.clip(
        res['sw_cell'].cpu().numpy().flatten(), analyzer.Swc, 1.0 - analyzer.Sgr)
    Sw0, lBL = analyzer._calibrate_sw_from_tds(well_id)
    dpb = np.maximum(pc - pwfb, 0.1)
    yrs = td / 365.25
    disc = 1.0 / (1.08 ** yrs)

    cache = dict(
        qg_base=qgb, pwf_base=pwfb, p_cell=pc, dp_base=dpb,
        sw_base=swb, shutin=shut, t_days=td, dt_days=dtd, dt_step=dts,
        data_end_idx=dei, forecast_len=n_time - dei, n_time=n_time,
        Sw_0=Sw0, lambda_BL=lBL,
        Swc=analyzer.Swc, Sgr=analyzer.Sgr,
        nw=analyzer.nw, ng=analyzer.ng,
        krw_max=analyzer.krw_max, krg_max=analyzer.krg_max,
        mu_w=analyzer.mu_w, mu_g=analyzer.mu_g,
        Sw_mobile_range=analyzer.Sw_mobile_range,
        gas_price=2.50, water_cost=50.0, discount_factors=disc,
        well_id=well_id, h_well=hw, bg_ref=bg_ref, sampler_t_max=s.t_max,
    )
    logger.info(f"  cache ok: qg=[{qgb.min():.0f},{qgb.max():.0f}], Sw0={Sw0:.4f}, lBL={lBL:.2e}")
    return cache


def _pareto_filter(F):
    n = len(F)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]: continue
        for j in range(n):
            if i == j or not mask[j]: continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                mask[i] = False; break
    return mask


def _make_top(label, x, gp, sw, npv):
    return dict(label=label, dp1=x[0], dp2=x[1], t_switch=x[2],
                ramp_days=x[3], Gp_M=gp, Sw_end=sw, NPV_M=npv)


def _extract_top3(Xd, Gp, Sw, NPV):
    if len(Gp) == 0:
        return []
    top3 = []
    used = set()

    idx1 = int(np.argmax(Gp))
    top3.append(_make_top('最大产气', Xd[idx1], Gp[idx1], Sw[idx1], NPV[idx1]))
    used.add(idx1)

    gn = (Gp - Gp.min()) / (Gp.ptp() + 1e-10)
    sn = (Sw - Sw.min()) / (Sw.ptp() + 1e-10)
    nn = (NPV - NPV.min()) / (NPV.ptp() + 1e-10)
    score = gn - sn + nn
    order2 = np.argsort(-score)
    idx2 = int(next((j for j in order2 if j not in used), order2[0]))
    top3.append(_make_top('最佳平衡', Xd[idx2], Gp[idx2], Sw[idx2], NPV[idx2]))
    used.add(idx2)

    order3 = np.argsort(Sw)
    idx3 = int(next((j for j in order3 if j not in used), order3[0]))
    top3.append(_make_top('最保守', Xd[idx3], Gp[idx3], Sw[idx3], NPV[idx3]))
    return top3


def _eval_existing(cache):
    strats = {}
    f = nsga2_evaluate([0, 0, 0.5, 90], cache)
    strats['稳产方案'] = dict(Gp_M=-f[0], Sw_end=f[1], NPV_M=-f[2], color='#E74C3C')
    f = nsga2_evaluate([1.5, 3.0, 0.5, 30], cache)
    strats['阶梯降产'] = dict(Gp_M=-f[0], Sw_end=f[1], NPV_M=-f[2], color='#F39C12')
    fd = cache['t_days'][-1] - cache['t_days'][cache['data_end_idx']]
    f = nsga2_evaluate([0, 6.0, 0.3, max(fd * 0.8, 30)], cache)
    strats['控压方案'] = dict(Gp_M=-f[0], Sw_end=f[1], NPV_M=-f[2], color='#27AE60')
    return strats


def run_nsga2_optimization(cache, pop_size=100, n_gen=30, seed=20260304):
    """Phase 2: NSGA-II优化主流程"""
    logger.info(f"NSGA-II Phase2: pop={pop_size}, gen={n_gen}")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PP
        from pymoo.optimize import minimize as pm
        from pymoo.termination import get_termination
        HAS = True
    except ImportError:
        HAS = False
        logger.warning("pymoo未安装, 回退随机采样")

    t0 = time.time()
    if HAS:
        class _P(PP):
            def __init__(s2, cr):
                super().__init__(n_var=4, n_obj=3, n_ieq_constr=0,
                                 xl=np.array([0.,0.,0.3,30.]),
                                 xu=np.array([5.,7.,0.7,180.]))  # v4.7: dp2上限10→7 避免关井平台
                s2.cr = cr; s2.ne = 0
            def _evaluate(s2, X, out, *a, **kw):
                F = np.zeros((len(X), 3))
                for i, x in enumerate(X):
                    F[i] = nsga2_evaluate(x, s2.cr); s2.ne += 1
                out["F"] = F
        prob = _P(cache)
        res = pm(prob, NSGA2(pop_size=pop_size),
                 get_termination("n_gen", n_gen), seed=seed, verbose=False)
        Fp, Xd, ne = res.F, res.X, prob.ne
    else:
        ns = pop_size * n_gen
        rng = np.random.RandomState(seed)
        Xd = np.column_stack([rng.uniform(0,5,ns), rng.uniform(0,7,ns),  # v4.7: dp2上限10→7
                               rng.uniform(0.3,0.7,ns), rng.uniform(30,180,ns)])
        Fp = np.array([nsga2_evaluate(Xd[i], cache) for i in range(ns)])
        mk = _pareto_filter(Fp); Fp = Fp[mk]; Xd = Xd[mk]; ne = ns

    el = time.time() - t0
    Gp, Sw, NPV = -Fp[:,0], Fp[:,1], -Fp[:,2]
    top3 = _extract_top3(Xd, Gp, Sw, NPV)
    ex = _eval_existing(cache)

    logger.info(f"  done: {ne}evals {el:.1f}s pareto={len(Gp)}")
    for i, t in enumerate(top3):
        logger.info(f"  TOP{i+1}[{t['label']}]: dp=({t['dp1']:.1f},{t['dp2']:.1f}) "
                     f"Gp={t['Gp_M']:.1f}M Sw={t['Sw_end']:.3f} NPV={t['NPV_M']:.1f}M")
    return dict(X_dec=Xd, F=Fp, Gp_M=Gp, Sw_end=Sw, NPV_M=NPV,
                elapsed=el, n_eval=ne, top3=top3, existing_strategies=ex)
