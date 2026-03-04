"""
配置加载器
==========
从 outputs 目录加载训练结果、图表、报告
"""

import os
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class OutputLoader:
    """加载 outputs 目录下的训练结果"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.outputs_dir = project_root / 'outputs'
        
    def load_config(self) -> Dict:
        """加载项目配置文件"""
        config_path = self.project_root / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_training_history(self, exp_name: str = 'mk_pinn_dt_v2') -> Optional[Dict]:
        """加载训练历史 JSON"""
        history_path = self.outputs_dir / exp_name / 'reports' / 'M5_training_history.json'
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def load_inversion_params(self, exp_name: str = 'mk_pinn_dt_v2') -> Optional[Dict]:
        """加载反演参数"""
        params_path = self.outputs_dir / exp_name / 'reports' / 'M5_inversion_params.json'
        if params_path.exists():
            with open(params_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def load_connectivity_matrix(self, exp_name: str = 'mk_pinn_dt_v2') -> Optional[pd.DataFrame]:
        """加载连通性矩阵"""
        matrix_path = self.outputs_dir / exp_name / 'reports' / 'M6_connectivity_matrix.csv'
        if matrix_path.exists():
            df = pd.read_csv(matrix_path)
            df.index = df.columns
            return df
        return None
    
    def get_figure_path(self, exp_name: str, fig_name: str) -> Optional[Path]:
        """获取图片路径"""
        fig_path = self.outputs_dir / exp_name / 'figs' / fig_name
        if fig_path.exists():
            return fig_path
        return None
    
    def list_experiments(self) -> List[str]:
        """列出所有可用的实验"""
        if not self.outputs_dir.exists():
            return []
        return [d.name for d in self.outputs_dir.iterdir() if d.is_dir()]
    
    def load_well_data(self, well_id: str = 'SY9') -> Optional[pd.DataFrame]:
        """加载井数据（如果有导出的话）"""
        data_path = self.project_root / 'data' / 'clean' / f'{well_id}_production.csv'
        if data_path.exists():
            return pd.read_csv(data_path)
        return None


def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve()
    # 从 app/components/config_loader.py 向上 2 层
    return current.parent.parent.parent


# 全局单例
_loader = None

def get_loader() -> OutputLoader:
    """获取全局 OutputLoader 实例"""
    global _loader
    if _loader is None:
        _loader = OutputLoader(get_project_root())
    return _loader
