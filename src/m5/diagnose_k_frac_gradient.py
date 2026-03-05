#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_frac 梯度诊断脚本
===================
验证 k_frac 是否真的梯度断裂 (Qg Loss → Peaceman WI → k_frac 路径)。

用法 (在项目根目录):
    python src/diagnose_k_frac_gradient.py
    python src/diagnose_k_frac_gradient.py --ckpt best   # 指定 best 或 final
    python src/diagnose_k_frac_gradient.py --cpu         # 强制 CPU（避免 GPU 驱动/架构不兼容）
"""

import sys
import os
from pathlib import Path

# 脚本位于 src/m5/，需将 src 加入 path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
project_root = Path(__file__).resolve().parent.parent.parent

import torch
import numpy as np
from pinn.m5_model import M5PINNNet
from utils import load_config


def main():
    # 加载配置 (与 run_m5_single_well 一致)
    config = load_config(str(project_root / 'config.yaml'))
    for key, value in config['paths'].items():
        config['paths'][key] = str(project_root / value)

    use_cpu = '--cpu' in sys.argv
    device = 'cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    if use_cpu:
        print("[设备] 使用 CPU (--cpu)")
    model = M5PINNNet(config, well_ids=['SY9']).to(device)

    # 尝试加载最新 checkpoint (不更新参数，仅用于梯度诊断)
    ckpt_dir = config['paths'].get('checkpoints', os.path.join(config['paths']['outputs'], 'ckpt'))
    ckpt_path = None
    if '--ckpt' in sys.argv:
        idx = sys.argv.index('--ckpt')
        tag = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'best'
        ckpt_path = os.path.join(ckpt_dir, f'm5_pinn_{tag}.pt')
    else:
        for tag in ('best', 'final'):
            p = os.path.join(ckpt_dir, f'm5_pinn_{tag}.pt')
            if os.path.exists(p):
                ckpt_path = p
                break

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)  # v4.8: 兼容新增 _r_e_raw
        # 确保加载后仍为训练模式且参数可导 (不冻结)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        print(f"[加载] 已加载 checkpoint: {ckpt_path}")
    else:
        print(f"[加载] 未找到 checkpoint，使用随机初始化模型 (ckpt_dir={ckpt_dir})")

    # 构造测试数据 (SY9 井位, t=500 天)
    # 归一化坐标: x_norm, y_norm ∈ [-1,1], t_norm ∈ [0,1]
    t_norm = 500.0 / 1331.0
    x_well_norm = torch.tensor([[0.0, 0.0, t_norm]], device=device, dtype=torch.float32)
    qg_obs = torch.tensor([[15000.0]], device=device, dtype=torch.float32)  # 观测产气量 m³/d

    # 前向计算
    model.train()
    model.zero_grad()

    result = model.evaluate_at_well(
        well_id='SY9',
        well_xyt_norm=x_well_norm,
        h_well=90.0,
        bg_val=0.002577,
        krg_val=None,
    )

    qg_pred = result['qg']
    qg_loss = torch.mean((qg_pred - qg_obs) ** 2)

    # ========== 中间变量审计 (Forward Pass Audit) — 在 backward 之前 ==========
    print("\n" + "=" * 60)
    print("中间变量审计 (Forward Pass Audit)")
    print("=" * 60)

    p_cell = result['p_cell']
    p_wf = result['p_wf']
    dp_draw = p_cell - p_wf
    WI = result['WI']
    lambda_g = result['lambda_g']

    print(f"p_cell = {p_cell.item():.2f} MPa")
    print(f"sw_cell = {result['sw_cell'].item():.4f}")
    print(f"p_wf = {p_wf.item():.2f} MPa")
    print(f"Δp (p_cell - p_wf) = {dp_draw.item():.2f} MPa")
    print(f"WI = {WI.item():.6e} (SI: m³)")
    print(f"λ_g = {lambda_g.item():.6e} (1/(Pa·s))")

    print(f"\nqg 计算分解:")
    print(f"  qg_reservoir = WI × λ_g × Δp(Pa)  [m³/s]")
    dp_Pa = dp_draw.item() * 1e6
    print(f"     = {WI.item():.3e} × {lambda_g.item():.3e} × {dp_Pa:.2e} Pa")
    qg_reservoir_val = WI.item() * lambda_g.item() * (p_cell.item() - p_wf.item()) * 1e6
    bg_val_t = model.well_model.pvt.bg(p_cell).item()
    print(f"     → qg_reservoir = {qg_reservoir_val:.6e} m³/s")
    print(f"  qg_surface = qg_reservoir / Bg,  Bg = {bg_val_t:.6e}")
    print(f"  qg_m3d = qg_surface × 86400 = {qg_pred.item():.1f} m³/d")

    print(f"\n梯度追踪检查:")
    print(f"  p_cell.requires_grad = {p_cell.requires_grad}")
    print(f"  p_wf.requires_grad = {p_wf.requires_grad}")
    print(f"  qg.requires_grad = {qg_pred.requires_grad}")
    print(f"  qg.grad_fn = {qg_pred.grad_fn}")

    print(f"\n参数状态检查:")
    # k_frac_mD 是 property 返回 tensor，需用 .requires_grad 看其依赖的 _k_frac_raw
    print(f"  _k_frac_raw.requires_grad = {model.well_model.peaceman._k_frac_raw.requires_grad}")
    print(f"  _dp_wellbore_raw.requires_grad = {model._dp_wellbore_raw.requires_grad}")
    if model.k_net is not None:
        first_linear = next(m for m in model.k_net.net if isinstance(m, torch.nn.Linear))
        print(f"  k_net.net[0].weight.requires_grad = {first_linear.weight.requires_grad}")

    # 反向传播 (不 step，仅诊断梯度)
    qg_loss.backward()

    # ========== 梯度审计 ==========
    print("\n" + "=" * 60)
    print("梯度诊断报告 (Qg Loss Gradient Audit)")
    print("=" * 60)

    # 1. k_frac 梯度 (梯度在 _k_frac_raw 上，k_frac_mD 是 property 无 .grad)
    k_raw = model.well_model.peaceman._k_frac_raw
    k_frac_grad = k_raw.grad
    if k_frac_grad is not None:
        k_frac_grad_val = k_frac_grad.item()
        print(f"\n[核心诊断] k_frac (_k_frac_raw) 梯度:")
        print(f"  值: {k_frac_grad_val:.6e}")
        print(f"  量级: {'正常 (>1e-6)' if abs(k_frac_grad_val) > 1e-6 else '异常 (<1e-6)'}")
    else:
        k_frac_grad_val = 0.0
        print(f"\n[核心诊断] k_frac 梯度: None (梯度断裂)")

    # 2. dp_wellbore 梯度
    dp_raw = model._dp_wellbore_raw
    dp_grad = dp_raw.grad
    if dp_grad is not None:
        dp_grad_val = dp_grad.item()
        print(f"\n[核心诊断] dp_wellbore 梯度:")
        print(f"  值: {dp_grad_val:.6e}")
        print(f"  量级: {'正常 (>1e-6)' if abs(dp_grad_val) > 1e-6 else '异常 (<1e-6)'}")
    else:
        print(f"\n[核心诊断] dp_wellbore 梯度: None (本脚本仅 Qg 路径，p_wf 来自 pwf_net 非 WHP+dp，故预期为 None)")

    # 3. k_net 梯度 (如果启用) — PermeabilityNet 首层为 .net[0]
    if model.k_net is not None:
        first_linear = next(m for m in model.k_net.net if isinstance(m, torch.nn.Linear))
        k_net_grad = first_linear.weight.grad
        if k_net_grad is not None:
            grad_norm = k_net_grad.norm().item()
            print(f"\n[对比诊断] k_net 首层权重梯度范数:")
            print(f"  值: {grad_norm:.6e}")
            print(f"  量级: {'正常 (>1e-4)' if grad_norm > 1e-4 else '异常 (<1e-4)'}")
        else:
            grad_norm = 0.0
            print(f"\n[对比诊断] k_net 首层权重梯度: None (本脚本 WI 用 k_frac 未用 k_net，故预期为 None)")
    else:
        grad_norm = 0.0
        print(f"\n[对比诊断] k_net: 未启用")

    # 4. 场网络梯度 (PINNNet 首层为 input_proj)
    field_first = model.field_net.input_proj.weight.grad
    if field_first is not None:
        field_grad_norm = field_first.norm().item()
        print(f"\n[基准对比] field_net 首层 (input_proj) 权重梯度范数:")
        print(f"  值: {field_grad_norm:.6e}")
    else:
        field_grad_norm = 0.0
        print(f"\n[基准对比] field_net 首层梯度: None")

    # 5. 梯度比值分析
    print("\n" + "=" * 60)
    print("梯度比值分析 (Gradient Ratio Analysis)")
    print("=" * 60)

    if model.k_net is not None and grad_norm > 0:
        ratio_k_frac_to_k_net = abs(k_frac_grad_val) / max(grad_norm, 1e-20)
        print(f"k_frac梯度 / k_net梯度 = {ratio_k_frac_to_k_net:.6e}")
        if ratio_k_frac_to_k_net < 1e-10:
            print("  ❌ 严重失衡! k_frac 几乎无梯度，k_net 主导更新")
            print("  → 验证: k_net 完全替代了 k_frac 的梯度路径 (evaluate_at_well 中 k_local_mD 覆盖)")
        elif ratio_k_frac_to_k_net < 1e-3:
            print("  ⚠️  轻度失衡，k_frac 梯度过小")
        else:
            print("  ✅ 梯度分配合理")

    if field_grad_norm > 0:
        ratio_k_frac_to_field = abs(k_frac_grad_val) / max(field_grad_norm, 1e-20)
        print(f"k_frac梯度 / field_net梯度 = {ratio_k_frac_to_field:.6e}")
        if ratio_k_frac_to_field < 1e-6:
            print("  ⚠️  k_frac 梯度远小于场网络，可能需要提高 k_frac_lr_factor 或 inversion_lr_factor")

    # 6. 当前参数值
    print("\n" + "=" * 60)
    print("当前参数值")
    print("=" * 60)
    print(f"k_frac = {model.well_model.peaceman.k_frac_mD.item():.4f} mD")
    print(f"dp_wellbore = {model.dp_wellbore.item():.2f} MPa")
    print(f"qg_pred = {qg_pred.item():.1f} m³/d")
    print(f"qg_obs = {qg_obs.item():.1f} m³/d")
    print(f"qg_loss = {qg_loss.item():.6e}")

    print("\n诊断完成!")


if __name__ == "__main__":
    main()
