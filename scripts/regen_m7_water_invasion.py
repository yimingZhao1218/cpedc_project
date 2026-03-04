"""
v3.17: 单独重跑 M7 水侵预警分析 (PINN正演决策支持版)
用法: python scripts/regen_m7_water_invasion.py

核心改进:
    - 全程+分界线: 历史实线+外推虚线, 数据截止线
    - 仪表盘: SY9状态 + 三策略Sw对比 + 全场风险 + 决策表
    - 策略对比: 2×2叠加面板, 外推区ΔSw差异放大
    - 报告: PINN正演替代器叙事, 突出秒级策略筛选价值
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
from pinn.water_invasion import WaterInvasionAnalyzer

# ── 加载配置和模型 ──
config = load_config(os.path.join(PROJECT_ROOT, 'config.yaml'))
device = 'cpu'

sampler = PINNSampler(config)
model = M5PINNNet(config).to(device)

# ── 加载 best checkpoint ──
# 注: final.pt 与 best.pt 均为同一次最新训练 (k_frac=14.28mD)
# M5_validation_report.md 中 9.013mD 是旧训练过期报告, 不反映当前模型
# best.pt 的压力场 dp_base 更大, 策略扰动分化更清晰
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

output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'mk_pinn_dt_v2')

# ── 先跑 M6 连通性 (需要C矩阵和Sw数据) ──
print("\n── M6 连通性分析 ──")
conn = ConnectivityAnalyzer(model, sampler, config)
conn.compute_connectivity_matrix()
conn.compute_water_risk_index()

# ── 跑 M7 水侵预警 (传入M6连通性) ──
print("\n── M7 水侵预警 ──")
wi = WaterInvasionAnalyzer(model, sampler, config, connectivity_analyzer=conn)
wi.generate_all(output_dir)

print("\n✅ M7 水侵预警分析重新生成完毕!")
print(f"   图件: {output_dir}/figs/M7_*.png")
print(f"   报告: {output_dir}/reports/M7_water_invasion_report.md")
