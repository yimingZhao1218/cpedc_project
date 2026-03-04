#!/usr/bin/env python3
"""
合并消融实验结果: 日志中的前4组指标 + pkl缓存的后2组完整数据
生成完整的6组对比图件和报告
"""
import os, sys, pickle
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
from utils import setup_chinese_support, setup_logger, load_config, ensure_dir

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
setup_chinese_support()

# 从日志中提取的前4组指标 (Ablation_20260301_101916.log)
LOG_RESULTS = [
    {'name': 'pure_ml',      'rmse': 89006, 'mape': 25.0, 'mape_test': 1.8,
     'elapsed': 2757.2, 'inference_ms': 2.5, 'final_pde': 0.0, 'final_total': 0, 'final_qg': 0,
     'rmse_train': 0, 'rmse_test': 0},
    {'name': 'pinn_const_k', 'rmse': 82595, 'mape': 23.0, 'mape_test': 1.5,
     'elapsed': 4501.2, 'inference_ms': 2.4, 'final_pde': 0.0, 'final_total': 0, 'final_qg': 0,
     'rmse_train': 0, 'rmse_test': 0},
    {'name': 'pinn_base',    'rmse': 93021, 'mape': 27.1, 'mape_test': 2.7,
     'elapsed': 4351.2, 'inference_ms': 1.8, 'final_pde': 0.0, 'final_total': 0, 'final_qg': 0,
     'rmse_train': 0, 'rmse_test': 0},
    {'name': 'pinn_full',    'rmse': 79971, 'mape': 20.8, 'mape_test': 1.6,
     'elapsed': 5950.1, 'inference_ms': 2.4, 'final_pde': 0.0, 'final_total': 0, 'final_qg': 0,
     'rmse_train': 0, 'rmse_test': 0},
]

def main():
    logger = setup_logger('MergeAblation')
    config = load_config(str(project_root / 'config.yaml'))
    for key, value in config['paths'].items():
        config['paths'][key] = str(project_root / value)

    base_out = config['paths']['outputs']
    results = list(LOG_RESULTS)

    # 加载 pkl 缓存的后2组
    for name in ['pinn_no_fourier', 'pinn_no_rar']:
        pkl_path = os.path.join(base_out, 'M6_ablation', name, 'ablation_result.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                res = pickle.load(f)
            logger.info(f"加载 {name}: RMSE={res.get('rmse',0):.0f}")
            results.append(res)
        else:
            logger.warning(f"未找到 {name} 的缓存!")

    logger.info(f"合并完成: {len(results)} 组实验")

    # 导入绘图和报告函数
    sys.path.insert(0, str(project_root / 'src' / 'm6'))
    from run_ablation_suite import generate_comparison_plots, generate_text_report

    generate_comparison_plots(results, base_out, logger)
    generate_text_report(results, base_out, logger)

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("完整6组消融实验摘要")
    logger.info("=" * 60)
    for res in sorted(results, key=lambda r: r.get('rmse', float('inf'))):
        logger.info(
            f"  {res['name']:15s}: RMSE={res.get('rmse',0):>8.0f}, "
            f"MAPE={res.get('mape',0):>6.1f}%, "
            f"Test_MAPE={res.get('mape_test',0):>6.1f}%"
        )

if __name__ == '__main__':
    main()
