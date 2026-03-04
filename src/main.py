#!/usr/bin/env python3
"""
CPEDC 创新组 B层标准 - 主执行脚本
执行 M1 ~ M6 完整流程并进行验收

使用方法:
    python main.py --stage m1        # 仅执行M1
    python main.py --stage m2        # 仅执行M2（需要M1完成）
    python main.py --stage m3        # 仅执行M3（PVT+相渗）
    python main.py --stage m4        # 仅执行M4（PINN基线训练）
    python main.py --stage m5        # 仅执行M5（井—藏耦合同化）
    python main.py --stage m6        # 仅执行M6（消融+UQ）
    python main.py --stage all       # 执行M1~M6完整流程（默认）
    python main.py --validate-only   # 仅验收，不执行
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 启用 CuBLAS 确定性，消除 deterministic 模式下的 UserWarning（需在 import torch 之前设置）
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from utils import setup_chinese_support, setup_logger, load_config, ensure_dir
from m1.m1_data_processor import M1_DataProcessor
from m2.m2_geo_builder import M2_GeoDomainBuilder
from m1.validate_m1 import M1Validator
from m2.validate_m2 import M2Validator
from m3.validate_m3 import M3Validator


def print_banner(text: str):
    """打印标题横幅"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def run_m3(config: dict, logger) -> bool:
    """
    执行 M3: 物性模块（PVT + 相渗）
    
    不需要"训练"，只需加载数据、构建插值器、验收
    """
    from physics.pvt import GasPVT
    from physics.relperm import RelPermGW
    
    logger.info("构建 PVT 插值器...")
    gas_pvt = GasPVT(config=config)
    
    logger.info("构建相渗插值器...")
    relperm = RelPermGW(config=config)
    
    # 快速功能测试
    logger.info("功能测试: 查询 p=50 MPa, T=120 ℃")
    props = gas_pvt.query_all(50.0, 120.0)
    for name, val in props.items():
        logger.info(f"  {name} = {val.item():.6g}")
    
    logger.info("功能测试: 查询 Sw=0.5")
    krw_val = relperm.krw(0.5).item()
    krg_val = relperm.krg(0.5).item()
    logger.info(f"  krw(0.5) = {krw_val:.6f}, krg(0.5) = {krg_val:.6f}")
    
    # 端点
    Swr, Sgr, krw_max, krg_max = relperm.endpoints()
    logger.info(f"  端点: Swr={Swr:.4f}, Sgr={Sgr:.4f}, "
                f"krw_max={krw_max:.4f}, krg_max={krg_max:.4f}")
    
    return True


def run_m4(config: dict, logger) -> bool:
    """
    执行 M4: PINN 基线训练
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch 未安装! 请运行: pip install torch")
        logger.info("M4 需要 PyTorch，跳过训练")
        return False
    
    from pinn.model import PINNNet
    from pinn.sampler import PINNSampler
    from pinn.losses import PINNLoss
    from pinn.trainer import PINNTrainer
    
    # 设备选择：先试 CUDA，不兼容时回退 CPU（与 validate_m4 一致）
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device='cuda')
            _ = t + 1
            device = 'cuda'
        except RuntimeError:
            logger.warning("CUDA 可用但当前 PyTorch 无法在本机 GPU 上运行，已回退到 CPU")
            device = 'cpu'
    else:
        device = 'cpu'
    logger.info(f"M4 PINN 训练设备: {device}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
        logger.info(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"\n>>> M4 使用 GPU 训练: {gpu_name} ({gpu_mem:.1f} GB)\n")
    else:
        logger.info("  提示: 未检测到 CUDA GPU，使用 CPU 训练（速度较慢）")
        logger.info("  如需 GPU 加速，请安装 CUDA 版 PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("\n>>> M4 使用 CPU 训练（未检测到 CUDA，速度较慢）\n")
    
    # 初始化组件
    logger.info("初始化采样器...")
    sampler = PINNSampler(config=config)
    
    logger.info("初始化网络...")
    model = PINNNet(config)
    logger.info(f"  模型参数量: {model.count_parameters():,}")
    
    logger.info("初始化损失函数...")
    loss_fn = PINNLoss(config, device=device)
    
    logger.info("初始化训练器...")
    trainer = PINNTrainer(config, model, loss_fn, sampler, device=device)
    
    # 训练
    logger.info("开始分阶段训练...")
    history = trainer.train()
    
    # 出图
    trainer.plot_training_history()
    trainer.plot_pressure_comparison()
    
    logger.info("M4 PINN 基线训练完成!")
    return True


def run_m5(config: dict, logger) -> bool:
    """
    执行 M5: 井—藏耦合同化 (SY9 单井)
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch 未安装!")
        return False
    
    from pinn.sampler import PINNSampler
    from pinn.m5_model import M5PINNNet
    from pinn.m5_trainer import M5Trainer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"M5 训练设备: {device}")
    
    primary_well = config['data'].get('primary_well', 'SY9')
    logger.info(f"M5 主井: {primary_well}")
    
    sampler = PINNSampler(config=config)
    model = M5PINNNet(config, well_ids=[primary_well])
    logger.info(f"  M5 参数量: {model.count_parameters_breakdown()}")
    
    trainer = M5Trainer(config, model, sampler, device=device)
    history = trainer.train()
    
    # 生成报告
    trainer.generate_report()
    
    # 反演参数
    inv = model.get_inversion_params()
    logger.info(f"  反演 k_eff: {inv.get('k_eff_mD', 'N/A'):.4f} mD")
    logger.info(f"  反演 f_frac: {inv.get('f_frac', 'N/A'):.2f}")
    
    return True


def run_m6(config: dict, logger) -> bool:
    """
    执行 M6: 连通性分析 + 消融实验 + UQ
    
    子功能:
        1. 连通性分析: 加载M5 best checkpoint, 生成k场/连通矩阵/报告
        2. 消融实验: 多组对比训练 (pure_ml / pinn_full 等)
        3. UQ: 多种子ensemble (可选, 完整版请用独立脚本)
    
    独立脚本:
        python src/m6/run_ablation_suite.py
        python src/m6/run_uq_ensemble.py
        python scripts/regen_m6_connectivity.py
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch 未安装!")
        return False
    
    from pinn.sampler import PINNSampler
    from pinn.m5_model import M5PINNNet
    import copy
    
    device = config.get('device')
    if device is None or device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA 不可用，退回到 CPU")
    logger.info(f"M6 设备: {device}")
    primary_well = config['data'].get('primary_well', 'SY9')
    base_out = config['paths']['outputs']
    
    # ── 1. 连通性分析 (加载 M5 best checkpoint) ──
    logger.info("M6-1: 连通性分析...")
    try:
        from m6.connectivity import ConnectivityAnalyzer
        
        ckpt_dir = config['paths'].get('checkpoints',
                                        os.path.join(base_out, 'ckpt'))
        ckpt_path = os.path.join(ckpt_dir, 'm5_pinn_best.pt')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, 'm5_pinn_final.pt')
        
        if os.path.exists(ckpt_path):
            sampler = PINNSampler(config=config)
            model = M5PINNNet(config, well_ids=[primary_well]).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            logger.info(f"  加载 checkpoint: {ckpt_path}")
            
            conn = ConnectivityAnalyzer(model, sampler, config)
            conn.generate_all(base_out)
            logger.info("  连通性分析完成")
        else:
            logger.warning("  未找到 M5 checkpoint, 跳过连通性分析")
    except Exception as e:
        logger.warning(f"  连通性分析跳过: {e}")
    
    # ── 2. 消融实验 ──
    logger.info("M6-2: 消融实验...")
    from pinn.m5_trainer import M5Trainer
    
    ablation_cfg = config.get('ablation', {})
    experiments = ablation_cfg.get('experiments', [])
    
    if not experiments:
        experiments = [
            {'name': 'pure_ml', 'overrides': {
                'physics.enable': False, 'loss.physics.enable': False
            }},
            {'name': 'pinn_full', 'overrides': {
                'physics.enable': True
            }},
        ]
    
    results_summary = []
    for exp in experiments:
        exp_name = exp['name']
        logger.info(f"\n  消融: {exp_name}")
        
        exp_config = copy.deepcopy(config)
        m6_exp_out = os.path.join(base_out, 'M6_ablation', exp_name)
        ensure_dir(m6_exp_out)
        exp_config['paths']['outputs'] = m6_exp_out
        exp_config['paths']['checkpoints'] = os.path.join(m6_exp_out, 'ckpt')
        exp_config['paths']['reports'] = os.path.join(m6_exp_out, 'reports')
        exp_config['paths']['figures'] = os.path.join(m6_exp_out, 'figs')
        exp_config['train']['max_steps'] = exp_config['train'].get(
            'max_steps', 50000
        )
        
        for key_path, value in exp.get('overrides', {}).items():
            keys = key_path.split('.')
            d = exp_config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        
        try:
            sampler = PINNSampler(config=exp_config)
            model = M5PINNNet(exp_config, well_ids=[primary_well])
            trainer = M5Trainer(exp_config, model, sampler, device=device)
            history = trainer.train()
            try:
                trainer.generate_report(report_ckpt='best')
            except Exception as report_err:
                logger.warning(f"  消融 {exp_name} 报告生成跳过: {report_err}")
            final_qg = history['qg'][-1] if history['qg'] else float('inf')
            results_summary.append(f"  {exp_name}: qg_loss={final_qg:.4e}")
        except Exception as e:
            logger.warning(f"  消融 {exp_name} 失败: {e}")
            results_summary.append(f"  {exp_name}: FAILED")
    
    logger.info("\n消融结果摘要:")
    for line in results_summary:
        logger.info(line)
    
    return True


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='CPEDC 创新组 PINN数字孪生系统 - M1~M4实施'
    )
    parser.add_argument(
        '--stage',
        choices=['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'all'],
        default='all',
        help='执行阶段: m1, m2, m3, m4, m5, m6, 或 all (默认: all)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅执行验收，不运行主流程'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='跳过验收检查'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='覆盖训练步数 (如 5000 用于快速测试 M5/M6)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default=None,
        help='强制使用 cuda 或 cpu (默认跟随 config)'
    )
    
    args = parser.parse_args()
    
    # 设置中文支持 & 加载配置（路径绝对化）
    setup_chinese_support()
    logger = setup_logger('Main')
    
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(str(project_root / 'config.yaml'))
    for key, value in config['paths'].items():
        config['paths'][key] = str(project_root / value)
    
    print_banner("CPEDC 创新组 B层标准项目")
    print("项目: 边水缝洞强非均质气藏PINN数字孪生系统")
    print("模块: M1(数据) + M2(地质域) + M3(物性) + M4(PINN基线) + M5(井藏耦合) + M6(消融/UQ)")
    print()
    
    try:
        # ===== 验收模式 =====
        if args.validate_only:
            print_banner("验收模式")
            
            if args.stage in ['m1', 'all']:
                print("\n>>> 执行 M1 验收...")
                m1_validator = M1Validator()
                m1_success = m1_validator.run()
                if not m1_success:
                    logger.error("M1验收未通过，请查看报告")
                    return False
            
            if args.stage in ['m2', 'all']:
                print("\n>>> 执行 M2 验收...")
                m2_validator = M2Validator()
                m2_success = m2_validator.run()
                if not m2_success:
                    logger.error("M2验收未通过，请查看报告")
                    return False
            
            if args.stage in ['m3', 'all']:
                print("\n>>> 执行 M3 验收...")
                m3_validator = M3Validator()
                m3_success = m3_validator.run()
                if not m3_success:
                    logger.error("M3验收未通过，请查看报告")
                    return False
            
            if args.stage in ['m4', 'all']:
                print("\n>>> 执行 M4 验收...")
                from m4.validate_m4 import M4Validator
                m4_validator = M4Validator()
                m4_success = m4_validator.run()
                if not m4_success:
                    logger.error("M4验收未通过，请查看报告")
                    return False
            
            if args.stage in ['m6', 'all']:
                print("\n>>> 执行 M6 验收/签收...")
                from m6.validate_m6 import M6Validator
                m6_validator = M6Validator()
                m6_success = m6_validator.run()
                if not m6_success:
                    logger.error("M6签收未通过，请查看 reports/M6_validation_report.md")
                    return False
            
            print_banner("所有验收通过!")
            return True
        
        # ===== 正常执行模式 =====
        
        # ---------- M1 ----------
        if args.stage in ['m1', 'all']:
            print_banner("第一步: M1 数据层与坐标统一")
            processor = M1_DataProcessor()
            m1_results = processor.run()
            
            if not args.skip_validation:
                print("\n>>> 执行 M1 自动验收...")
                m1_validator = M1Validator()
                m1_success = m1_validator.run()
                if not m1_success:
                    logger.error("M1验收未通过，请修复问题后重试")
                    return False
            
            print_banner("M1 完成!")
        
        # ---------- M2 ----------
        if args.stage in ['m2', 'all']:
            clean_data_path = Path(config['paths']['clean_data']) / 'mk_interval_points.csv'
            if not clean_data_path.exists():
                logger.error("M2需要M1的输出，请先执行M1")
                return False
            
            print_banner("第二步: M2 弱空间地质域构建")
            builder = M2_GeoDomainBuilder()
            m2_results = builder.run()
            
            if not args.skip_validation:
                print("\n>>> 执行 M2 自动验收...")
                m2_validator = M2Validator()
                m2_success = m2_validator.run()
                if not m2_success:
                    logger.error("M2验收未通过，请修复问题后重试")
                    return False
            
            print_banner("M2 完成!")
        
        # ---------- M3 ----------
        if args.stage in ['m3', 'all']:
            print_banner("第三步: M3 物性模块（PVT + 相渗）")
            m3_ok = run_m3(config, logger)
            
            if not args.skip_validation:
                print("\n>>> 执行 M3 自动验收...")
                m3_validator = M3Validator()
                m3_success = m3_validator.run()
                if not m3_success:
                    logger.error("M3验收未通过，请修复问题后重试")
                    return False
            
            print_banner("M3 完成!")
        
        # ---------- M4 ----------
        if args.stage in ['m4', 'all']:
            # 检查 M2/M3 前置条件
            geo_grid = Path(config['paths']['geo_data']) / 'grids' / 'collocation_grid.csv'
            if not geo_grid.exists():
                logger.error("M4 需要 M2 的输出（配点网格），请先执行 M2")
                return False
            
            print_banner("第四步: M4 PINN 基线（最小闭环）")
            m4_ok = run_m4(config, logger)
            
            if not args.skip_validation and m4_ok:
                print("\n>>> 执行 M4 自动验收...")
                from m4.validate_m4 import M4Validator
                m4_validator = M4Validator()
                m4_success = m4_validator.run()
                if not m4_success:
                    logger.warning("M4验收未完全通过（基线可接受，后续迭代改进）")
            
            print_banner("M4 完成!")
        
        # ---------- M5 ----------
        if args.stage in ['m5', 'all']:
            geo_grid = Path(config['paths']['geo_data']) / 'grids' / 'collocation_grid.csv'
            if not geo_grid.exists():
                logger.error("M5 需要 M2 的输出（配点网格），请先执行 M2")
                return False
            
            print_banner("第五步: M5 井—藏耦合同化")
            m5_ok = run_m5(config, logger)
            
            if m5_ok:
                print_banner("M5 完成!")
            else:
                logger.warning("M5 执行异常（可接受，后续迭代改进）")
        
        # ---------- M6 ----------
        if args.stage in ['m6', 'all']:
            geo_grid = Path(config['paths']['geo_data']) / 'grids' / 'collocation_grid.csv'
            if not geo_grid.exists():
                logger.error("M6 需要 M2 的输出（配点网格），请先执行 M2")
                return False
            print_banner("第六步: M6 消融 + UQ")
            if args.max_steps is not None:
                config['train']['max_steps'] = args.max_steps
                logger.info(f"使用 --max-steps={args.max_steps}")
            if args.device is not None:
                config['device'] = args.device
                logger.info(f"使用 --device={args.device}")
            m6_ok = run_m6(config, logger)
            
            if m6_ok:
                print_banner("M6 完成!")
            else:
                logger.warning("M6 执行异常")
        
        # ===== 最终总结 =====
        if args.stage == 'all':
            clean = config['paths']['clean_data']
            geo = config['paths']['geo_data']
            out = config['paths']['outputs']
            rpt = config['paths']['reports']
            fig = config['paths'].get('figures', os.path.join(out, 'figs'))
            
            print_banner("M1 ~ M6 完整流程执行完成!")
            print("\n输出文件清单:")
            
            print("\n【M1 输出】")
            print(f"  - {clean}/wellpath_stations.csv  (井眼3D轨迹)")
            print(f"  - {clean}/mk_interval_points.csv (MK段代表点)")
            print(f"  - {clean}/production_SY9.csv     (SY9生产数据)")
            
            print("\n【M2 输出】")
            print(f"  - {geo}/surfaces/mk_top_surface.csv   (MK顶面)")
            print(f"  - {geo}/surfaces/mk_bot_surface.csv   (MK底面)")
            print(f"  - {geo}/surfaces/mk_thickness.csv     (厚度场)")
            print(f"  - {geo}/grids/collocation_grid.csv    (PINN配点网格)")
            
            print("\n【M3 输出】")
            print(f"  - {out}/M3_pvt_curves.png         (PVT曲线)")
            print(f"  - {out}/M3_relperm_curves.png     (相渗曲线)")
            print(f"  - {rpt}/M3_pvt_report.md          (PVT验收报告)")
            print(f"  - {rpt}/M3_relperm_report.md      (相渗验收报告)")
            
            print("\n【M4 输出】")
            print(f"  - {out}/M4_training_history.png   (训练曲线)")
            print(f"  - {out}/M4_pressure_comparison.png(压力对比)")
            print(f"  - {rpt}/M4_validation_report.md   (PINN验收报告)")
            
            print("\n【M5 输出】")
            print(f"  - {fig}/M5_qg_comparison_*.png    (产气量拟合)")
            print(f"  - {fig}/M5_pwf_inversion_*.png    (p_wf反演)")
            print(f"  - {fig}/M5_training_history.png   (M5训练曲线)")
            print(f"  - {fig}/M5_pde_residual_map.png   (残差热力图)")
            print(f"  - {rpt}/M5_validation_report.md   (M5验收报告)")
            print(f"  - {rpt}/M5_inversion_params.json  (反演参数)")
            
            print("\n【M6 输出】")
            print(f"  - {out}/M6_ablation/<实验名>/figs/  (main 内消融各实验图表)")
            print(f"  - {fig}/M6_ablation_*.png         (独立脚本消融对比图)")
            print(f"  - {fig}/M6_uq_*.png               (UQ P10/P50/P90)")
            print(f"  - {rpt}/M6_ablation_report.md     (消融报告)")
            print(f"  - {rpt}/M6_uq_report.md           (UQ报告)")
            
            print("\n下一步:")
            print("  M7: 论文/答辩准备")
            print()
        
        return True
        
    except Exception as e:
        logger.error(f"执行失败: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
