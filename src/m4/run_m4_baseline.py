#!/usr/bin/env python3
"""
M4 PINN 基线训练脚本
===================
仅执行第四步：PINN 基线训练（SY9 压力趋势拟合 + 分阶段课程学习）

使用方法:
    python run_m4_baseline.py
    python run_m4_baseline.py --steps 50000
    python run_m4_baseline.py --device cuda
    python run_m4_baseline.py --skip-validation
    python run_m4_baseline.py --debug-nan   # 开启 NaN 调试：isfinite 断言 + autograd 异常检测 + dump
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 脚本位于 src/m4/，需将 src 加入 path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
project_root = Path(__file__).resolve().parent.parent.parent

from utils import setup_chinese_support, setup_logger, load_config, ensure_dir

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    parser = argparse.ArgumentParser(description='M4 PINN 基线训练')
    parser.add_argument('--steps', type=int, default=None, help='训练步数 (覆盖 config)')
    parser.add_argument('--device', default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--skip-validation', action='store_true', help='训练后不执行 M4 验收')
    parser.add_argument('--debug-nan', action='store_true',
                        help='开启 NaN 调试：子损失 isfinite 断言、backward 时 autograd 异常检测、非有限时 dump 到 outputs/debug_nan/step_{step}.pt')
    args = parser.parse_args()

    setup_chinese_support()
    logger = setup_logger('M4_Run')

    # P0: 强制 cwd=project_root，解决 PVT 相对路径 data/raw/... 找不到（由 cwd 引起）
    os.chdir(project_root)
    logger.info(f"cwd set to: {os.getcwd()}")

    # 加载配置
    config = load_config(str(project_root / 'config.yaml'))
    for key, value in config['paths'].items():
        config['paths'][key] = str(project_root / value)

    if args.steps:
        config['train']['max_steps'] = args.steps
    # debug_nan：可由命令行覆盖 config（默认关闭）
    if args.debug_nan:
        config['debug_nan'] = True

    try:
        import torch
    except ImportError:
        logger.error("PyTorch 未安装! 请运行: pip install torch")
        sys.exit(1)

    # 设备选择：若未指定则先试 CUDA，不兼容时回退 CPU（与 validate_m4 一致）
    if args.device:
        device = args.device.strip().lower()
        device = device if device in ('cuda', 'cpu') else 'cpu'
    elif torch.cuda.is_available():
        try:
            t = torch.zeros(1, device='cuda')
            _ = t + 1
            device = 'cuda'
        except RuntimeError:
            logger.warning("CUDA 可用但当前 PyTorch 无法在本机 GPU 上运行，已回退到 CPU")
            device = 'cpu'
    else:
        device = 'cpu'

    logger.info("=" * 60)
    logger.info("M4 PINN 基线训练")
    logger.info("=" * 60)
    logger.info(f"  设备: {device}")
    logger.info(f"  训练步数: {config['train']['max_steps']}")
    logger.info(f"  debug_nan: {config.get('debug_nan', False)}")
    if device == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 前置检查：需要 M2 配点网格
    geo_grid = Path(config['paths']['geo_data']) / 'grids' / 'collocation_grid.csv'
    if not geo_grid.exists():
        logger.error(f"需要 M2 输出: {geo_grid}")
        logger.error("请先执行: python main.py --stage m2")
        sys.exit(1)

    # 初始化组件
    logger.info("初始化采样器...")
    from pinn.sampler import PINNSampler
    sampler = PINNSampler(config=config)

    logger.info("初始化 M4 网络...")
    from pinn.model import PINNNet
    model = PINNNet(config)
    logger.info(f"  参数量: {model.count_parameters():,}")

    logger.info("初始化损失函数...")
    from pinn.losses import PINNLoss
    loss_fn = PINNLoss(config, device=device)

    logger.info("初始化 M4 训练器...")
    from pinn.trainer import PINNTrainer
    trainer = PINNTrainer(config, model, loss_fn, sampler, device=device)

    # 训练
    logger.info("开始 M4 分阶段训练...")
    start_time = time.time()
    history = trainer.train()
    elapsed = time.time() - start_time

    logger.info(f"M4 训练完成! 耗时: {elapsed:.1f}s")

    # 出图
    logger.info("生成训练曲线与压力对比图...")
    trainer.plot_training_history()
    trainer.plot_pressure_comparison()

    # 可选：M4 验收（与训练使用相同 device，避免 CUDA 与显卡架构不兼容时崩溃）
    if not args.skip_validation:
        logger.info("执行 M4 验收...")
        from m4.validate_m4 import M4Validator
        validator = M4Validator()
        success = validator.run(device=device)
        if not success:
            logger.warning("M4 验收未完全通过（可接受，后续迭代改进）")
        else:
            logger.info("M4 验收通过 ✅")

    # 输出摘要
    out_dir = config['paths']['outputs']
    fig_dir = config['paths'].get('figures', os.path.join(out_dir, 'figs'))
    rpt_dir = config['paths'].get('reports', os.path.join(out_dir, 'reports'))
    ckpt_dir = config['paths'].get('checkpoints', os.path.join(out_dir, 'ckpt'))

    logger.info("=" * 60)
    logger.info("M4 训练结果摘要")
    logger.info("=" * 60)
    logger.info(f"  最终总损失: {history['total'][-1]:.6e}")
    logger.info(f"  最终 IC 损失: {history['ic'][-1]:.6e}")
    logger.info(f"  最终 BC 损失: {history['bc'][-1]:.6e}")
    logger.info(f"  最终 PDE 损失: {history['pde'][-1]:.6e}")
    logger.info(f"  最终 Data 损失: {history['data'][-1]:.6e}")
    logger.info("=" * 60)

    logger.info("\n输出文件:")
    logger.info(f"  - {ckpt_dir}/pinn_best.pt")
    logger.info(f"  - {ckpt_dir}/pinn_final.pt")
    logger.info(f"  - {fig_dir}/M4_training_history.png")
    logger.info(f"  - {fig_dir}/M4_pressure_comparison.png")
    logger.info(f"  - {rpt_dir}/M4_validation_report.md")


if __name__ == '__main__':
    main()
