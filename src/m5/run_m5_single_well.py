#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M5 单井同化训练脚本
===================
执行 SY9 单井的井—藏耦合同化训练 + 反演 + 验收输出

使用方法:
    python -m src.m5.run_m5_single_well
    python -m src.m5.run_m5_single_well --well SY9 --steps 50000
    或从项目根目录: python src/m5/run_m5_single_well.py --well SY9 --steps 50000

    python src/m5/run_m5_single_well.py --well SY9 --steps 2000 --device cuda   # 短步诊断
    python src/m5/run_m5_single_well.py --well SY9 --steps 10000                 # 建议：达标验收至少 1 万步
    python src/m5/run_m5_single_well.py --no-rar          # 关闭 RAR
    python src/m5/run_m5_single_well.py --no-relobralo     # 关闭 ReLoBRaLo
"""

# ===== 在所有 import 之前强制设置 UTF-8，避免 Windows 控制台中文乱码 =====
import sys
import os

if sys.platform == 'win32':
    # 设置控制台代码页为 UTF-8
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except:
        pass
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 强制 stdout/stderr 使用 UTF-8
for stream_name in ['stdout', 'stderr']:
    stream = getattr(sys, stream_name)
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass
    elif hasattr(stream, 'buffer'):
        try:
            import io
            new_stream = io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='replace', line_buffering=True)
            setattr(sys, stream_name, new_stream)
        except:
            pass
# ===== UTF-8 设置结束 =====

import argparse
import time
from pathlib import Path

# 脚本位于 src/m5/，需将 src 加入 path 以便 import utils / pinn
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
project_root = Path(__file__).resolve().parent.parent.parent

from utils import setup_chinese_support, setup_logger, load_config, ensure_dir

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 启用 CuBLAS 确定性，消除 deterministic 模式下的 UserWarning（需在 import torch 之前设置）
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main():
    parser = argparse.ArgumentParser(description='M5 单井同化训练')
    parser.add_argument('--well', default='SY9', help='主井号 (默认 SY9)')
    parser.add_argument('--steps', type=int, default=None, help='训练步数 (覆盖 config)')
    parser.add_argument('--no-rar', action='store_true', help='关闭 RAR')
    parser.add_argument('--no-relobralo', action='store_true', help='关闭 ReLoBRaLo')
    parser.add_argument('--device', default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--resume', action='store_true', help='从best checkpoint微调 (跳过A/B阶段冷启动)')
    args = parser.parse_args()
    
    setup_chinese_support()
    logger = setup_logger('M5_Run')
    
    # 加载配置
    config = load_config(str(project_root / 'config.yaml'))
    for key, value in config['paths'].items():
        config['paths'][key] = str(project_root / value)
    
    # 命令行覆盖
    config['data']['mode'] = 'single_well'
    config['data']['primary_well'] = args.well
    
    if args.steps:
        config['train']['max_steps'] = args.steps
    if args.no_rar:
        config.setdefault('m5_config', {}).setdefault('rar', {})['enable'] = False
    if args.no_relobralo:
        config.setdefault('m5_config', {}).setdefault('relobralo', {})['enable'] = False
    
    # 设备 (PyTorch 需 cuda 而非 gpu)
    import torch
    if args.device:
        device = args.device.lower()
        if device == 'gpu':
            device = 'cuda'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("=" * 60)
    logger.info("M5 单井同化训练")
    logger.info("=" * 60)
    logger.info(f"  主井: {args.well}")
    logger.info(f"  设备: {device}")
    logger.info(f"  训练步数: {config['train']['max_steps']}")
    
    # 前置检查
    geo_grid = Path(config['paths']['geo_data']) / 'grids' / 'collocation_grid.csv'
    if not geo_grid.exists():
        logger.error(f"需要 M2 输出: {geo_grid}")
        sys.exit(1)
    
    prod_file = Path(config['paths']['clean_data']) / f'production_{args.well}.csv'
    if not prod_file.exists():
        logger.error(f"需要生产数据: {prod_file}")
        sys.exit(1)
    
    # 初始化组件
    logger.info("初始化采样器...")
    from pinn.sampler import PINNSampler
    sampler = PINNSampler(config=config)
    
    logger.info("初始化 M5 网络...")
    from pinn.m5_model import M5PINNNet
    model = M5PINNNet(config, well_ids=[args.well])
    
    params_breakdown = model.count_parameters_breakdown()
    logger.info(f"  参数量: {params_breakdown}")
    
    logger.info("初始化 M5 训练器...")
    from pinn.m5_trainer import M5Trainer
    trainer = M5Trainer(config, model, sampler, device=device)
    
    # 训练
    if args.resume:
        logger.info("[resume] 加载 best checkpoint 作为微调起点...")
        trainer.load_checkpoint('best')
        logger.info("[resume] 权重已恢复, 开始微调")

    logger.info("开始 M5 训练...")
    start_time = time.time()
    history = trainer.train()
    elapsed = time.time() - start_time
    
    logger.info(f"M5 训练完成! 耗时: {elapsed:.1f}s")
    
    # 训练结束后: 在 best/final 中自动择优再出报告（auto=按 qg 综合分数选更优）
    logger.info("生成 M5 验收报告 (auto: best vs final 择优)...")
    trainer.generate_report(report_ckpt='auto')
    
    # 输出摘要
    inv = model.get_inversion_params()
    logger.info("=" * 60)
    logger.info("M5 训练结果摘要")
    logger.info("=" * 60)
    _k = inv.get('k_eff_mD', inv.get('k_frac_mD'))
    _f = inv.get('f_frac')
    _dp = inv.get('dp_wellbore_MPa')
    logger.info(f"  反演 k_eff: {_k:.4f} mD" if isinstance(_k, (int, float)) else f"  反演 k_eff: {_k} mD")
    logger.info(f"  反演 f_frac: {_f:.2f}" if isinstance(_f, (int, float)) else f"  反演 f_frac: {_f if _f is not None else 'N/A'}")
    logger.info(f"  反演 dp_wellbore: {_dp:.2f} MPa" if isinstance(_dp, (int, float)) else f"  反演 dp_wellbore: {_dp} MPa")
    logger.info(f"  最终总损失: {history['total'][-1]:.6e}")
    logger.info(f"  最终 PDE 损失: {history['pde'][-1]:.6e}")
    logger.info(f"  最终 qg 损失: {history['qg'][-1]:.6e}")
    logger.info("=" * 60)
    
    logger.info("\n输出文件:")
    out_dir = config['paths']['outputs']
    fig_dir = config['paths'].get('figures', os.path.join(out_dir, 'figs'))
    rpt_dir = config['paths'].get('reports', os.path.join(out_dir, 'reports'))
    logger.info(f"  - {fig_dir}/M5_qg_comparison_*.png")
    logger.info(f"  - {fig_dir}/M5_pwf_inversion_*.png")
    logger.info(f"  - {fig_dir}/M5_training_history.png")
    logger.info(f"  - {fig_dir}/M5_pde_residual_map.png")
    logger.info(f"  - {rpt_dir}/M5_validation_report.md")
    logger.info(f"  - {rpt_dir}/M5_inversion_params.json")


if __name__ == '__main__':
    main()
