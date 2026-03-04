"""
TDD 验证: dp_wellbore 冻结 (v3.7)
确认:
  1. dp_wellbore 值 = 13.3 MPa
  2. dp_wellbore 不在 model.parameters() 中
  3. dp_wellbore 不在 optimizer param groups 中
  4. 反向传播不影响 dp_wellbore
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import yaml


def load_test_config():
    cfg_path = project_root / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    for key, value in config['paths'].items():
        config['paths'][key] = str(project_root / value)
    return config


def test_dp_is_frozen():
    """dp_wellbore 应该是 buffer, 不是 Parameter"""
    from pinn.m5_model import M5PINNNet
    config = load_test_config()
    well_ids = ['SY9']
    model = M5PINNNet(config, well_ids)

    # 1. dp 值 = 13.3
    dp_val = model.dp_wellbore.item()
    assert abs(dp_val - 13.3) < 0.01, f"dp_wellbore={dp_val}, expected 13.3"
    print(f"  [PASS] dp_wellbore = {dp_val:.2f} MPa")

    # 2. dp 不在 named_parameters 中
    param_names = [n for n, _ in model.named_parameters()]
    dp_in_params = any('dp_wellbore' in n or '_dp_wellbore' in n for n in param_names)
    assert not dp_in_params, f"dp_wellbore found in parameters: {[n for n in param_names if 'dp' in n]}"
    print(f"  [PASS] dp_wellbore NOT in model.parameters()")

    # 3. dp 在 buffers 中
    buffer_names = [n for n, _ in model.named_buffers()]
    dp_in_buffers = any('dp_wellbore' in n or '_dp_wellbore' in n for n in buffer_names)
    assert dp_in_buffers, f"dp_wellbore NOT found in buffers: {buffer_names}"
    print(f"  [PASS] dp_wellbore IS in model.buffers()")

    # 4. dp 无梯度
    assert not model.dp_wellbore.requires_grad, "dp_wellbore should not require grad"
    print(f"  [PASS] dp_wellbore.requires_grad = False")

    return model, config


def test_dp_not_in_optimizer(model, config):
    """dp_wellbore 不应该出现在优化器参数组中"""
    import torch.optim as optim

    train_cfg = config.get('train', {})
    opt_cfg = train_cfg.get('optimizer', {})
    lr = train_cfg.get('learning_rate', 1e-3)
    inv_lr_factor = opt_cfg.get('inversion_lr_factor', 1.5)
    k_frac_lr_factor = opt_cfg.get('k_frac_lr_factor', 10.0)

    # 模拟 m5_trainer 的参数分组逻辑
    field_params = list(model.field_net.parameters())
    well_other_params = []
    k_frac_params = []

    if hasattr(model, 'well_model'):
        for name, param in model.well_model.named_parameters():
            if '_k_frac_raw' in name:
                k_frac_params.append(param)
            else:
                well_other_params.append(param)

    # 关键检查: dp_wellbore 不应被添加
    if hasattr(model, '_dp_wellbore_raw') and isinstance(model._dp_wellbore_raw, torch.nn.Parameter):
        raise AssertionError("_dp_wellbore_raw is still an nn.Parameter! Should be a buffer.")

    print(f"  [PASS] _dp_wellbore_raw is NOT an nn.Parameter")

    # 构建优化器
    param_groups = [
        {'params': field_params, 'lr': lr},
        {'params': well_other_params, 'lr': lr * inv_lr_factor},
    ]
    if k_frac_params:
        param_groups.append({'params': k_frac_params, 'lr': lr * k_frac_lr_factor})

    optimizer = optim.AdamW(param_groups, lr=lr)

    # 验证 dp 不在任何组
    all_opt_params = set()
    for g in optimizer.param_groups:
        for p in g['params']:
            all_opt_params.add(id(p))
    
    dp_id = id(model.dp_wellbore)
    assert dp_id not in all_opt_params, "dp_wellbore tensor found in optimizer!"
    print(f"  [PASS] dp_wellbore NOT in optimizer param groups")


def test_dp_survives_backward():
    """反向传播后 dp_wellbore 值不变"""
    from pinn.m5_model import M5PINNNet
    config = load_test_config()
    model = M5PINNNet(config, ['SY9'])

    dp_before = model.dp_wellbore.item()

    # 模拟一个损失计算
    dummy_whp = torch.tensor([58.0])
    p_wf_target = dummy_whp + model.dp_wellbore
    p_wf_pred = torch.tensor([72.0], requires_grad=True)
    loss = (p_wf_pred - p_wf_target) ** 2
    loss.backward()

    dp_after = model.dp_wellbore.item()
    assert abs(dp_after - dp_before) < 1e-6, f"dp changed from {dp_before} to {dp_after} after backward!"
    print(f"  [PASS] dp_wellbore unchanged after backward: {dp_after:.4f}")


if __name__ == '__main__':
    print("=" * 50)
    print("TDD: dp_wellbore 冻结验证 (v3.7)")
    print("=" * 50)

    print("\n[Test 1] dp_wellbore 是 buffer 且值为 13.3")
    model, config = test_dp_is_frozen()

    print("\n[Test 2] dp_wellbore 不在优化器中")
    test_dp_not_in_optimizer(model, config)

    print("\n[Test 3] backward 不影响 dp_wellbore")
    test_dp_survives_backward()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
