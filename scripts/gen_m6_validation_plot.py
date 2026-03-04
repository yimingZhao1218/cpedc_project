"""
单独生成 M6 连通性验证图 (不重跑全部M6)
用法: python scripts/gen_m6_validation_plot.py
"""
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from utils import load_config
from pinn.m5_model import M5PINNNet
from pinn.sampler import PINNSampler
from m6.connectivity import ConnectivityAnalyzer

# ── 加载配置和模型 ──
config = load_config(os.path.join(PROJECT_ROOT, 'config.yaml'))
device = 'cpu'

sampler = PINNSampler(config)
model = M5PINNNet(config).to(device)

# ── 加载 final checkpoint (v3.17) ──
ckpt_dir = os.path.join(PROJECT_ROOT, 'outputs', 'mk_pinn_dt_v2', 'ckpt')
ckpt_path = os.path.join(ckpt_dir, 'm5_pinn_final.pt')
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(ckpt_dir, 'm5_pinn_best.pt')

print(f"加载 checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

k_frac = model.well_model.peaceman.k_frac_mD.item()
print(f"k_frac = {k_frac:.4f} mD")

# ── 连通性分析 + 验证图 ──
conn = ConnectivityAnalyzer(model, sampler, config)
conn.compute_connectivity_matrix()

fig_dir = os.path.join(PROJECT_ROOT, 'outputs', 'mk_pinn_dt_v2', 'figs')
save_path = os.path.join(fig_dir, 'M6_connectivity_validation.png')
conn.plot_connectivity_validation(save_path)

print(f"\n✅ 连通性验证图已生成: {save_path}")
