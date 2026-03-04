"""
v3.13: 补丁 checkpoint 的 smooth_qg 历史 + 重新生成 M5 报告图件
用法: python scripts/regen_m5_report.py
"""
import re
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# ── Step 1: 从日志解析 SQg ──
LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'M5Trainer_20260222_010005.log')
pattern = re.compile(r'\[Step\s+(\d+)/\d+\].*SQg=([\d.eE+\-]+)')

sqg_by_step = {}
with open(LOG_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern.search(line)
        if m:
            sqg_by_step[int(m.group(1))] = float(m.group(2))

print(f"从日志解析到 {len(sqg_by_step)} 个 SQg 数据点")

# ── Step 2: 注入 checkpoint ──
CKPT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'mk_pinn_dt_v2', 'ckpt')

for tag in ['final', 'best']:
    ckpt_path = os.path.join(CKPT_DIR, f'm5_pinn_{tag}.pt')
    if not os.path.exists(ckpt_path):
        print(f"  跳过 {tag} (不存在)")
        continue
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    history = ckpt.get('history', {})
    steps = history.get('step', [])
    if not steps:
        print(f"  跳过 {tag} (无 history)")
        continue

    # 构建 smooth_qg 数组，对齐 history steps
    smooth_qg = []
    last_val = 0.0
    for s in steps:
        if s in sqg_by_step:
            last_val = sqg_by_step[s]
        smooth_qg.append(last_val)

    history['smooth_qg'] = smooth_qg
    ckpt['history'] = history
    torch.save(ckpt, ckpt_path)
    print(f"  ✅ {tag} checkpoint 已注入 {len(smooth_qg)} 个 smooth_qg 值")

# ── Step 3: 重新生成报告和图件 ──
print("\n重新生成 M5 报告...")

from utils import load_config
from pinn.m5_model import M5PINNNet
from pinn.pinn_sampler import PINNSampler
from pinn.m5_trainer import M5Trainer

config = load_config(os.path.join(PROJECT_ROOT, 'config.yaml'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sampler = PINNSampler(config)
model = M5PINNNet(config).to(device)
trainer = M5Trainer(config, model, sampler, device=device)

trainer.generate_report(report_ckpt='auto')
print("\n✅ 报告和图件重新生成完毕!")
