"""
UQ Runner: 不确定性量化 (Uncertainty Quantification)
=====================================================
通过多随机种子/多初始化 ensemble 实现低成本 UQ。

功能:
    1. 用不同随机种子训练 N 个模型
    2. 汇聚预测输出 P10/P50/P90 区间
    3. 反演参数给出区间/方差 (后验离散度)
    4. 输出 UQ 图表与摘要

使用方法:
    runner = UQRunner(config, n_ensemble=5)
    results = runner.run()
    runner.generate_report(results)
"""

import os
import sys
import time
import copy
import json
import math
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    raise ImportError("uq_runner 需要 PyTorch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logger, setup_chinese_support, ensure_dir

setup_chinese_support()
import matplotlib.pyplot as plt


class UQRunner:
    """
    不确定性量化 ensemble 运行器
    
    Args:
        config: 全局配置
        n_ensemble: ensemble 成员数 (>= 5)
        base_seed: 基础随机种子
    """
    
    def __init__(self, config: dict, n_ensemble: int = 5, base_seed: int = 20260209):
        self.config = config
        self.n_ensemble = max(n_ensemble, 3)
        self.base_seed = base_seed
        self.logger = setup_logger('UQRunner')
        
        output_dir = config['paths']['outputs']
        self.output_dir = output_dir
        self.fig_dir = config['paths'].get('figures', os.path.join(output_dir, 'figs'))
        self.report_dir = config['paths'].get('reports', os.path.join(output_dir, 'reports'))
        ensure_dir(self.fig_dir)
        ensure_dir(self.report_dir)
        
        self.logger.info(
            f"UQRunner 初始化: n_ensemble={self.n_ensemble}, base_seed={self.base_seed}"
        )
    
    def run(self) -> Dict[str, object]:
        """
        运行 ensemble 训练
        
        Returns:
            {
                'qg_predictions': list of arrays,
                'pwf_predictions': list of arrays,
                'inversion_params': list of dicts,
                'histories': list of dicts,
                't_days': array,
            }
        """
        from pinn.sampler import PINNSampler
        from pinn.m5_model import M5PINNNet
        from pinn.m5_trainer import M5Trainer
        
        results = {
            'qg_predictions': [],
            'pwf_predictions': [],
            'inversion_params': [],
            'histories': [],
            't_days': None,
        }
        
        # 初始化采样器 (所有 ensemble 共享)
        sampler = PINNSampler(config=self.config)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for i in range(self.n_ensemble):
            seed_i = self.base_seed + i * 1000
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Ensemble {i+1}/{self.n_ensemble}, seed={seed_i}")
            self.logger.info(f"{'='*50}")
            
            # 修改种子
            config_i = copy.deepcopy(self.config)
            config_i['reproducibility']['seed'] = seed_i
            
            # UQ 步数: 固定30k步(消融级别), 不使用全局80k避免训练时间膨胀
            config_i['train']['max_steps'] = 30000
            
            # --- v3: 直接扰动 k_frac_init, 避免 k_eff×f_frac 乘积爆炸 ---
            rng = np.random.RandomState(seed_i)
            
            # 1. 扰动 k_frac 初始值: 围绕最新收敛值 ~9.7 mD 做对数正态扰动
            #    v4.7: σ从0.3→0.5 扩大覆盖率(44%→目标>70%)
            #    k_frac ~ LogNormal(log(9.7), 0.5) → P5≈3.9, P50=9.7, P95≈24
            k_frac_center = 10.13  # v4.8: M5 80k全训收敛值 (MAPE=4.6%)
            k_frac_perturbed = max(rng.lognormal(math.log(k_frac_center), 0.5), 1.0)
            k_frac_perturbed = min(k_frac_perturbed, 40.0)  # v4.7: 30→40 适配更宽扰动
            # 分解为 k_eff × f_frac 写入 config (PeacemanWI 读取这两个先验)
            f_frac_fixed = 16.0  # 与 config 默认一致
            k_eff_for_config = k_frac_perturbed / f_frac_fixed
            priors = config_i.get('physics', {}).get('priors', {})
            if isinstance(priors.get('k_eff_mD'), dict):
                priors['k_eff_mD']['value'] = k_eff_for_config
            if isinstance(priors.get('frac_conductivity_factor'), dict):
                priors['frac_conductivity_factor']['value'] = f_frac_fixed
            
            # 2. 扰动边界条件: p_boundary ±1.5 MPa
            #    v4.7: ±0.5→±1.5 MPa, 覆盖测点分布范围(75.74~76.09)
            #    及深度换算+温度效应不确定性
            p_init = config_i.get('mk_formation', {}).get('avg_pressure_MPa', {})
            if isinstance(p_init, dict):
                p_base = p_init.get('value', 76.0)
            else:
                p_base = float(p_init)
            p_perturbed = p_base + rng.uniform(-1.5, 1.5)
            if isinstance(config_i['mk_formation']['avg_pressure_MPa'], dict):
                config_i['mk_formation']['avg_pressure_MPa']['value'] = p_perturbed
            
            self.logger.info(
                f"  扰动: k_frac_init={k_frac_perturbed:.2f} mD, "
                f"p_init={p_perturbed:.1f} MPa"
            )
            
            # ★ 在模型构造前显式设置种子 ★
            # M5Trainer.__init__ 中也会设置种子, 但那在 model 之后.
            # Fourier Feature 的随机矩阵 B 在 PINNNet.__init__ 中通过
            # torch.randn 生成, 必须在此之前设置种子才能保证:
            # (a) 不同 ensemble 成员的 B 矩阵不同 (因 seed 不同)
            # (b) 同一 seed 可复现相同 B 矩阵
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_i)
            
            try:
                # 创建模型
                well_ids = [config_i['data'].get('primary_well', 'SY9')]
                model = M5PINNNet(config_i, well_ids=well_ids)
                
                # 训练
                trainer = M5Trainer(config_i, model, sampler, device=device)
                history = trainer.train()
                
                # 收集预测
                model.eval()
                primary_well = well_ids[0]
                
                if primary_well in trainer.well_data:
                    wdata = trainer.well_data[primary_well]
                    
                    with torch.no_grad():
                        well_result = model.evaluate_at_well(
                            primary_well, wdata['xyt'],
                            h_well=trainer.well_h.get(primary_well, 90.0),
                            bg_val=trainer.bg_ref,
                        )
                    
                    qg_pred = well_result['qg'].cpu().numpy().flatten()
                    pwf_pred = well_result['p_wf'].cpu().numpy().flatten()
                    
                    results['qg_predictions'].append(qg_pred)
                    results['pwf_predictions'].append(pwf_pred)
                    results['t_days'] = wdata['t_days']
                
                results['inversion_params'].append(model.get_inversion_params())
                results['histories'].append({
                    'total': history['total'][-1] if history['total'] else float('inf'),
                    'qg': history['qg'][-1] if history['qg'] else float('inf'),
                })
                
                inv = results['inversion_params'][-1]
                k_frac_val = inv.get('k_frac_mD', inv.get('k_eff_mD', 'N/A'))
                k_str = f"{k_frac_val:.3f}" if isinstance(k_frac_val, (int, float)) else str(k_frac_val)
                self.logger.info(
                    f"  Ensemble {i+1} 完成, k_frac={k_str} mD"
                )
            except Exception as e:
                self.logger.error(
                    f"  Ensemble {i+1} 失败: {e}\n{traceback.format_exc()}"
                )
                self.logger.info("  跳过该成员, 继续下一个...")
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """生成 UQ 报告 (图表 + 文字)"""
        self._plot_qg_uq(results)
        self._plot_pwf_uq(results)
        self._plot_param_distribution(results)
        report_path = self._write_text_report(results)
        return report_path
    
    def _plot_qg_uq(self, results: Dict):
        """绘制 qg P10/P50/P90"""
        if not results['qg_predictions']:
            return
        
        qg_all = np.array(results['qg_predictions'])  # (N_ensemble, N_time)
        t_days = results['t_days']
        
        p10 = np.percentile(qg_all, 10, axis=0)
        p50 = np.percentile(qg_all, 50, axis=0)
        p90 = np.percentile(qg_all, 90, axis=0)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(t_days, p10, p90, alpha=0.2, color='blue', label='P10-P90')
        ax.plot(t_days, p50, 'b-', linewidth=1.5, label='P50')
        
        # 观测值
        from pinn.sampler import PINNSampler
        sampler = PINNSampler(config=self.config)
        wdata = sampler.sample_well_data(self.config['data'].get('primary_well', 'SY9'))
        if wdata:
            ax.plot(wdata['t_days'], wdata['qg_obs'], 'r.', markersize=3,
                    alpha=0.5, label='观测值')
        
        ax.set_xlabel('时间 (天)')
        ax.set_ylabel('产气量 (m³/d)')
        ax.set_title(f'UQ: 产气量 P10/P50/P90 (N={self.n_ensemble})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fp = os.path.join(self.fig_dir, 'M6_uq_qg_p10p50p90.png')
        fig.savefig(fp, dpi=200, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"UQ qg 图已保存: {fp}")
    
    def _plot_pwf_uq(self, results: Dict):
        """绘制 p_wf P10/P50/P90"""
        if not results['pwf_predictions']:
            return
        
        pwf_all = np.array(results['pwf_predictions'])
        t_days = results['t_days']
        
        p10 = np.percentile(pwf_all, 10, axis=0)
        p50 = np.percentile(pwf_all, 50, axis=0)
        p90 = np.percentile(pwf_all, 90, axis=0)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(t_days, p10, p90, alpha=0.2, color='red', label='P10-P90')
        ax.plot(t_days, p50, 'r-', linewidth=1.5, label='P50')
        ax.set_xlabel('时间 (天)')
        ax.set_ylabel('p_wf (MPa)')
        ax.set_title(f'UQ: 井底流压 P10/P50/P90 (N={self.n_ensemble})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fp = os.path.join(self.fig_dir, 'M6_uq_pwf_p10p50p90.png')
        fig.savefig(fp, dpi=200, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"UQ pwf 图已保存: {fp}")
    
    def _plot_param_distribution(self, results: Dict):
        """绘制反演参数分布"""
        if not results['inversion_params']:
            return
        
        k_vals = [p.get('k_frac_mD', p.get('k_eff_mD', 0)) for p in results['inversion_params']]
        
        # 左: k_frac 直方图 + strip plot
        # 右: 各成员 qg 最终损失
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        ax1.hist(k_vals, bins=max(3, self.n_ensemble // 2), edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(k_vals), color='red', linestyle='--', label=f'mean={np.mean(k_vals):.3f}')
        # strip plot: 显示每个成员的具体值
        for j, kv in enumerate(k_vals):
            ax1.plot(kv, 0.05, 'ko', markersize=8, alpha=0.6)
        ax1.set_xlabel('k_frac (mD)')
        ax1.set_title('k_frac 后验分布')
        ax1.legend()
        
        ax2 = axes[1]
        if results['histories']:
            qg_losses = [h.get('qg', 0) for h in results['histories']]
            member_ids = list(range(1, len(qg_losses) + 1))
            ax2.bar(member_ids, qg_losses, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Ensemble 成员')
            ax2.set_ylabel('Final Qg Loss')
            ax2.set_title('各成员最终 Qg 损失')
            ax2.set_xticks(member_ids)
        
        plt.tight_layout()
        fp = os.path.join(self.fig_dir, 'M6_uq_param_distribution.png')
        fig.savefig(fp, dpi=200, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"参数分布图已保存: {fp}")
    
    def _write_text_report(self, results: Dict) -> str:
        """生成文字摘要报告"""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = [
            "# M6 不确定性量化 (UQ) 报告\n",
            f"> 生成时间: {ts}\n",
            f"## 配置",
            f"- Ensemble 数量: {self.n_ensemble}",
            f"- 基础种子: {self.base_seed}",
            f"- 每成员训练步数: {self.config.get('train', {}).get('max_steps', 30000)}\n",
        ]
        
        # 反演参数统计
        if results['inversion_params']:
            k_vals = [p.get('k_frac_mD', p.get('k_eff_mD', 0)) for p in results['inversion_params']]
            
            lines.append("## 反演参数统计")
            lines.append(f"### k_frac (裂缝增强渗透率, mD)")
            lines.append(f"- P10: {np.percentile(k_vals, 10):.4f}")
            lines.append(f"- P50: {np.percentile(k_vals, 50):.4f}")
            lines.append(f"- P90: {np.percentile(k_vals, 90):.4f}")
            lines.append(f"- Mean: {np.mean(k_vals):.4f}")
            lines.append(f"- Std: {np.std(k_vals):.4f}")
            lines.append(f"- CV: {np.std(k_vals)/max(np.mean(k_vals), 1e-8)*100:.1f}%")
            lines.append(f"- 各成员值: {', '.join(f'{v:.3f}' for v in k_vals)}\n")
        
        # 预测区间
        if results['qg_predictions']:
            qg_all = np.array(results['qg_predictions'])
            cum_gas = np.cumsum(qg_all, axis=1)[:, -1]  # 累计产气
            lines.append("## 累计产气量 (m³)")
            lines.append(f"- P10: {np.percentile(cum_gas, 10):.0f}")
            lines.append(f"- P50: {np.percentile(cum_gas, 50):.0f}")
            lines.append(f"- P90: {np.percentile(cum_gas, 90):.0f}\n")
            
            # v4.1: P10-P90 覆盖率 (Coverage Probability)
            # 理想值: 80% (P10-P90区间应覆盖80%的观测值)
            try:
                from pinn.sampler import PINNSampler
                sampler_cp = PINNSampler(config=self.config)
                wdata_cp = sampler_cp.sample_well_data(
                    self.config['data'].get('primary_well', 'SY9'))
                if wdata_cp and 'qg_obs' in wdata_cp:
                    t_obs = wdata_cp['t_days']
                    qg_obs = wdata_cp['qg_obs']
                    if hasattr(qg_obs, 'cpu'):
                        qg_obs = qg_obs.cpu().numpy().flatten()
                    t_pred = results['t_days']
                    p10 = np.percentile(qg_all, 10, axis=0)
                    p90 = np.percentile(qg_all, 90, axis=0)
                    # 将P10/P90插值到观测时间点
                    p10_at_obs = np.interp(t_obs, t_pred, p10)
                    p90_at_obs = np.interp(t_obs, t_pred, p90)
                    valid = qg_obs > 0
                    if valid.any():
                        covered = (qg_obs[valid] >= p10_at_obs[valid]) & \
                                  (qg_obs[valid] <= p90_at_obs[valid])
                        coverage = float(covered.sum()) / float(valid.sum()) * 100
                        lines.append("## P10-P90 覆盖率 (Coverage Probability)")
                        lines.append(f"- 覆盖率: {coverage:.1f}% (理想值: 80%)")
                        lines.append(f"- 覆盖点数: {covered.sum()}/{valid.sum()}")
                        if coverage >= 70:
                            lines.append("- 评价: ✓ 区间校准良好, UQ可信度高\n")
                        elif coverage >= 50:
                            lines.append("- 评价: △ 区间偏窄, 建议增大扰动幅度\n")
                        else:
                            lines.append("- 评价: ✗ 区间严重偏窄, 需检查扰动策略\n")
            except Exception:
                pass
        
        lines.append("## 工程含义")
        lines.append("- P10/P50/P90 区间反映了模型参数不确定性对预测的影响")
        lines.append("- CV (变异系数) 越大说明对该参数的约束越弱，数据信息量不足")
        lines.append("- 决策建议: 以 P50 为基准方案，P10 为保守方案，P90 为乐观方案")
        
        report_path = os.path.join(self.report_dir, 'M6_uq_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"UQ 报告已保存: {report_path}")
        return report_path
