"""
CPEDC PINN 模块 (v2)
M4 基线 + M5 井—藏耦合 + M6 消融/域分解/UQ

v2 新增:
    - FourierFeatureEncoding: Fourier 特征编码 (解决频谱偏置)
    - PermeabilityNet: k(x,y) 空间渗透率子网络
    - ResidualBlock: 残差块
"""

# M4 基线
from .model import PINNNet, FourierFeatureEncoding, PermeabilityNet, ResidualBlock
from .sampler import PINNSampler
from .losses import PINNLoss
from .trainer import PINNTrainer

# M5 井—藏耦合
from .well_model import WellModel, PeacemanWI, PwfHiddenVariable, GaussianSourceTerm
from .m5_model import M5PINNNet
from .assimilation_losses import AssimilationLoss
from .relobralo import ReLoBRaLo, ManualLossBalancer
from .rar_sampler import RARSampler
from .m5_trainer import M5Trainer

# 可微分物性
from .torch_physics import TorchPVT, TorchRelPerm

# M6 域分解 + UQ + 连通性
from .xpinn import XPINNModel, APINNModel, create_domain_decomposition
from .uq_runner import UQRunner
from .connectivity import ConnectivityAnalyzer
from .water_invasion import WaterInvasionAnalyzer

__all__ = [
    # M4
    'PINNNet', 'FourierFeatureEncoding', 'PermeabilityNet', 'ResidualBlock',
    'PINNSampler', 'PINNLoss', 'PINNTrainer',
    # M5
    'WellModel', 'PeacemanWI', 'PwfHiddenVariable', 'GaussianSourceTerm',
    'M5PINNNet', 'AssimilationLoss', 'ReLoBRaLo', 'ManualLossBalancer',
    'RARSampler', 'M5Trainer',
    # 物性
    'TorchPVT', 'TorchRelPerm',
    # M6
    'XPINNModel', 'APINNModel', 'create_domain_decomposition', 'UQRunner',
    'ConnectivityAnalyzer',
    # M7
    'WaterInvasionAnalyzer',
]
