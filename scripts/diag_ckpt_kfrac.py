"""诊断: final.pt vs best.pt 的 k_frac 实际值"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config
from pinn.m5_model import M5PINNNet

config = load_config('config.yaml')
ckpt_dir = 'outputs/mk_pinn_dt_v2/ckpt'

for tag, fname in [('m5_final', 'm5_pinn_final.pt'), ('m5_best', 'm5_pinn_best.pt'),
                   ('pinn_final', 'pinn_final.pt'), ('pinn_best', 'pinn_best.pt')]:
    path = os.path.join(ckpt_dir, fname)
    if not os.path.exists(path):
        print(f'[{tag}] 文件不存在: {path}')
        continue
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = M5PINNNet(config).to('cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    k_frac = model.well_model.peaceman.k_frac_mD.item()
    inv = model.get_inversion_params()
    step = ckpt.get('step', 'N/A')
    best_loss = ckpt.get('best_loss', 'N/A')
    total_loss = ckpt.get('loss', 'N/A')

    print(f'\n=== {tag}.pt ===')
    print(f'  peaceman.k_frac_mD = {k_frac:.4f} mD')
    print(f'  get_inversion_params: {inv}')
    print(f'  checkpoint step={step}, loss={total_loss}, best_loss={best_loss}')
    print(f'  state_dict keys (k_frac相关): '
          + str([k for k in ckpt.get('model_state_dict', {}).keys() if 'k_frac' in k or 'peaceman' in k]))
    # 直接读state_dict里的k_frac值
    sd = ckpt.get('model_state_dict', {})
    for k, v in sd.items():
        if 'k_frac' in k or ('peaceman' in k and 'k' in k):
            print(f'  state_dict[{k}] = {v}')
