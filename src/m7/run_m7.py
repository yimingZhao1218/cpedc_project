#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M7 水侵预警与制度优化 — 一键运行脚本"""
import sys, os, time

if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except: pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

for _s in ['stdout', 'stderr']:
    _st = getattr(sys, _s)
    if hasattr(_st, 'reconfigure'):
        try: _st.reconfigure(encoding='utf-8', errors='replace')
        except: pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent

from utils import setup_chinese_support, setup_logger, load_config, ensure_dir
setup_chinese_support()
logger = setup_logger('M7_Run')

import torch

# ── 加载配置 ──
config = load_config(str(project_root / 'config.yaml'))
for k, v in config['paths'].items():
    config['paths'][k] = str(project_root / v)

output_dir = config['paths']['outputs']
ensure_dir(output_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info("=" * 60)
logger.info("M7 水侵预警与制度优化")
logger.info("=" * 60)
logger.info(f"  设备: {device}")
logger.info(f"  输出目录: {output_dir}")

# ── 初始化 M5 组件 ──
logger.info("加载 M5 PINN checkpoint...")
from pinn.sampler import PINNSampler
from pinn.m5_model import M5PINNNet
from pinn.m5_trainer import M5Trainer

sampler = PINNSampler(config=config)
model = M5PINNNet(config, well_ids=['SY9'])
trainer = M5Trainer(config, model, sampler, device=device)
trainer.load_checkpoint('best')
logger.info("M5 checkpoint 加载完成")

# ── 初始化 M7 分析器 ──
from pinn.water_invasion import WaterInvasionAnalyzer
analyzer = WaterInvasionAnalyzer(model, sampler, config)

# ── 一键生成全部输出 ──
t0 = time.time()
analyzer.generate_all(output_dir=output_dir, well_id='SY9')
elapsed = time.time() - t0

logger.info("=" * 60)
logger.info(f"M7 全部输出生成完毕，耗时 {elapsed:.1f}s")
logger.info("=" * 60)

fig_dir = os.path.join(output_dir, 'figs')
rpt_dir = os.path.join(output_dir, 'reports')
logger.info("输出文件:")
for fn in [
    'M7_water_invasion_dashboard.png',
    'M7_strategy_comparison.png',
    'M7_sw_vs_tds_validation.png',
    'M7_multiwell_tds_dashboard.png',
    'M7_tds_vs_wiri_crossvalidation.png',
    'M7_water_type_timeline.png',
    'M7_pareto_frontier.png',
    'M7_sensitivity_tornado.png',
]:
    fp = os.path.join(fig_dir, fn)
    status = '✓' if os.path.exists(fp) else '✗ 未生成'
    logger.info(f"  [{status}] figs/{fn}")
logger.info(f"  [报告] reports/M7_water_invasion_report.md")
