"""
v3.13: 单独重跑 M6 连通性分析 (数据融合版)
用法: python scripts/regen_m6_connectivity.py

修改内容:
    - 用附表3各井实测PERM + IDW插值替代k_net外推
    - SY9叠加PINN反演k_frac, SYX211用附表8补充
    - 报告增加数据源说明和SY9连通中心性分析
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
device = 'cpu'  # RTX 5060 CUDA sm_120 与当前 PyTorch 不兼容

sampler = PINNSampler(config)
model = M5PINNNet(config).to(device)

# ── 加载 best checkpoint ──
ckpt_dir = os.path.join(PROJECT_ROOT, 'outputs', 'mk_pinn_dt_v2', 'ckpt')
ckpt_path = os.path.join(ckpt_dir, 'm5_pinn_best.pt')
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(ckpt_dir, 'm5_pinn_final.pt')

print(f"加载 checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

k_frac = model.well_model.peaceman.k_frac_mD.item()
print(f"k_frac = {k_frac:.4f} mD")

# ── 运行 M6 连通性分析 ──
output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'mk_pinn_dt_v2')
conn = ConnectivityAnalyzer(model, sampler, config)
conn.generate_all(output_dir)

print("\n✅ M6 连通性分析重新生成完毕!")
print(f"   图件: {output_dir}/figs/M6_*.png")
print(f"   报告: {output_dir}/reports/M6_connectivity_report.md")
print(f"   矩阵: {output_dir}/reports/M6_connectivity_matrix.csv")
