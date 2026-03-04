#!/usr/bin/env python3
"""
M6 UQ Ensemble 入口脚本
========================
运行多随机种子 ensemble 不确定性量化。

使用方法:
    python src/m6/run_uq_ensemble.py
    python src/m6/run_uq_ensemble.py --n 10                # 10 个 ensemble 成员
    python src/m6/run_uq_ensemble.py --n 5 --seed 42       # 指定基础种子
"""

import os
import sys
import argparse
import time
from pathlib import Path

_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
project_root = Path(__file__).resolve().parent.parent.parent

from utils import setup_chinese_support, setup_logger, load_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    parser = argparse.ArgumentParser(description='M6 UQ Ensemble')
    parser.add_argument('--n', type=int, default=5, help='Ensemble 成员数 (默认 5)')
    parser.add_argument('--seed', type=int, default=20260209, help='基础种子')
    parser.add_argument('--device', default=None, help='设备')
    args = parser.parse_args()
    
    setup_chinese_support()
    logger = setup_logger('UQ_Ensemble')
    
    config = load_config(str(project_root / 'config.yaml'))
    for key, value in config['paths'].items():
        config['paths'][key] = str(project_root / value)
    
    config['data']['mode'] = 'single_well'
    
    logger.info("=" * 60)
    logger.info(f"M6 UQ Ensemble 运行 (N={args.n}, seed={args.seed})")
    logger.info("=" * 60)
    
    from pinn.uq_runner import UQRunner
    
    runner = UQRunner(config, n_ensemble=args.n, base_seed=args.seed)
    
    start = time.time()
    results = runner.run()
    elapsed = time.time() - start
    
    logger.info(f"\nUQ Ensemble 完成! 总耗时: {elapsed:.1f}s")
    
    # 生成报告
    report_path = runner.generate_report(results)
    logger.info(f"UQ 报告: {report_path}")
    
    # 摘要
    import numpy as np
    if results['inversion_params']:
        k_vals = [p.get('k_frac_mD', p.get('k_eff_mD', 0)) for p in results['inversion_params']]
        
        logger.info("\n反演参数 UQ 摘要:")
        logger.info(f"  k_frac: P10={np.percentile(k_vals,10):.3f}, "
                     f"P50={np.percentile(k_vals,50):.3f}, "
                     f"P90={np.percentile(k_vals,90):.3f} mD")
    
    logger.info("\n输出文件:")
    out = config['paths']['outputs']
    fig_dir = config['paths'].get('figures', os.path.join(out, 'figs'))
    logger.info(f"  - {fig_dir}/M6_uq_qg_p10p50p90.png")
    logger.info(f"  - {fig_dir}/M6_uq_pwf_p10p50p90.png")
    logger.info(f"  - {fig_dir}/M6_uq_param_distribution.png")
    logger.info(f"  - {report_path}")


if __name__ == '__main__':
    main()
