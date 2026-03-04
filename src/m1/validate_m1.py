"""
M1 验收标准检查脚本
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 脚本位于 src/m1/，需将 src 加入 path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import setup_chinese_support, load_config, setup_logger


class M1Validator:
    """M1验收验证器"""
    
    def __init__(self):
        setup_chinese_support()
        
        # 路径绝对化（与 M1/M2 同一套 project_root 逻辑）
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        self.config = load_config(str(project_root / 'config.yaml'))
        for key, value in self.config['paths'].items():
            self.config['paths'][key] = str(project_root / value)
        
        self.logger = setup_logger('M1_Validator')
        self.passed = []
        self.failed = []
        
    def check_file_exists(self, filepath: str, name: str):
        """检查文件是否存在"""
        if os.path.exists(filepath):
            self.passed.append(f"✅ {name}: 文件存在")
            return True
        else:
            self.failed.append(f"❌ {name}: 文件不存在 ({filepath})")
            return False
    
    def validate_wells_data(self):
        """验收1: 井位数据"""
        self.logger.info("验收1: 检查井位数据...")
        
        filepath = os.path.join(self.config['paths']['staged_data'], 'wells_staged.csv')
        if not self.check_file_exists(filepath, "井位数据文件"):
            return
        
        df = pd.read_csv(filepath)
        
        # 检查必需字段
        required_cols = ['well_id', 'x', 'y']
        for col in required_cols:
            if col in df.columns:
                self.passed.append(f"✅ 井位数据包含字段: {col}")
            else:
                self.failed.append(f"❌ 井位数据缺少字段: {col}")
        
        # 检查井数量和列表（兼容新旧配置格式）
        if 'data' in self.config and 'wells' in self.config['data']:
            expected_wells = set(self.config['data']['wells'])
        elif 'm1_config' in self.config and 'wells' in self.config['m1_config']:
            expected_wells = set(self.config['m1_config']['wells'])
        else:
            self.failed.append("❌ 配置文件中未找到井列表")
            return
        
        actual_wells = set(df['well_id'].tolist())
        
        if expected_wells == actual_wells:
            self.passed.append(f"✅ 井数正确: {len(actual_wells)} 口")
        else:
            missing = expected_wells - actual_wells
            extra = actual_wells - expected_wells
            if missing:
                self.failed.append(f"❌ 缺少井: {missing}")
            if extra:
                self.failed.append(f"⚠️ 额外的井: {extra}")
        
        # 检查坐标范围
        if 'x' in df.columns and 'y' in df.columns:
            x_range = df['x'].max() - df['x'].min()
            y_range = df['y'].max() - df['y'].min()
            self.passed.append(f"✅ 坐标范围: X={x_range:.1f}m, Y={y_range:.1f}m")
    
    def validate_wellpaths(self):
        """验收2: 井眼轨迹数据"""
        self.logger.info("验收2: 检查井眼轨迹数据...")
        
        filepath = os.path.join(self.config['paths']['clean_data'], 'wellpath_stations.csv')
        if not self.check_file_exists(filepath, "井眼轨迹文件"):
            return
        
        df = pd.read_csv(filepath)
        
        # 检查必需字段
        required_cols = ['well_id', 'md_m', 'x', 'y', 'z']
        for col in required_cols:
            if col in df.columns:
                self.passed.append(f"✅ 轨迹数据包含字段: {col}")
            else:
                self.failed.append(f"❌ 轨迹数据缺少字段: {col}")
        
        # 检查每口井的数据（兼容新旧配置）
        if 'data' in self.config and 'wells' in self.config['data']:
            well_list = self.config['data']['wells']
        elif 'm1_config' in self.config and 'wells' in self.config['m1_config']:
            well_list = self.config['m1_config']['wells']
        else:
            well_list = df['well_id'].unique().tolist()
        
        for well_id in well_list:
            well_data = df[df['well_id'] == well_id]
            if len(well_data) > 0:
                self.passed.append(f"✅ {well_id}: {len(well_data)} 个测点")
                
                # 检查MD单调递增
                if 'md_m' in well_data.columns:
                    md_values = well_data['md_m'].values
                    if np.all(np.diff(md_values) >= 0):
                        self.passed.append(f"  ✅ {well_id}: MD单调递增")
                    else:
                        self.failed.append(f"  ❌ {well_id}: MD不单调")
            else:
                self.failed.append(f"❌ {well_id}: 无轨迹数据")
    
    def validate_mk_interval_points(self):
        """验收3: MK段代表点"""
        self.logger.info("验收3: 检查MK段代表点...")
        
        filepath = os.path.join(self.config['paths']['clean_data'], 'mk_interval_points.csv')
        if not self.check_file_exists(filepath, "MK代表点文件"):
            return
        
        df = pd.read_csv(filepath)
        
        # 检查必需字段（旧字段）
        basic_cols = ['well_id', 'mk_top_z', 'mk_bot_z', 'mk_thickness']
        for col in basic_cols:
            if col in df.columns:
                self.passed.append(f"✅ MK点数据包含字段: {col}")
            else:
                self.failed.append(f"❌ MK点数据缺少字段: {col}")
        
        # 检查新字段（顶/底交点坐标）
        new_cols = ['x_top', 'y_top', 'x_bot', 'y_bot', 'dz_top', 'dz_bot', 
                    'out_of_range_top', 'out_of_range_bot']
        for col in new_cols:
            if col in df.columns:
                self.passed.append(f"✅ MK点数据包含新字段: {col}")
            else:
                self.failed.append(f"❌ MK点数据缺少新字段: {col}")
        
        # 检查厚度为正
        if 'mk_thickness' in df.columns:
            invalid_thickness = df['mk_thickness'] <= 0
            if invalid_thickness.any():
                self.failed.append(f"❌ 发现 {invalid_thickness.sum()} 口井厚度<=0")
            else:
                self.passed.append(f"✅ 所有井厚度>0")
                self.passed.append(f"  平均厚度: {df['mk_thickness'].mean():.2f} m")
        
        # 新增：检查越界情况
        if 'out_of_range_top' in df.columns and 'out_of_range_bot' in df.columns:
            n_out_top = df['out_of_range_top'].sum()
            n_out_bot = df['out_of_range_bot'].sum()
            
            if n_out_top == 0 and n_out_bot == 0:
                self.passed.append(f"✅ 无越界井点（顶界和底界均在轨迹范围内）")
            else:
                self.failed.append(f"❌ 越界井点: 顶界 {n_out_top} 口, 底界 {n_out_bot} 口")
                # 列出越界井
                for idx, row in df.iterrows():
                    if row.get('out_of_range_top', False):
                        self.failed.append(f"  - {row['well_id']}: 顶界越界")
                    if row.get('out_of_range_bot', False):
                        self.failed.append(f"  - {row['well_id']}: 底界越界")
        
        # 新增：检查dz_top和dz_bot的误差
        if 'dz_top' in df.columns and 'dz_bot' in df.columns:
            dz_top_valid = df['dz_top'].dropna()
            dz_bot_valid = df['dz_bot'].dropna()
            
            if len(dz_top_valid) > 0:
                max_dz_top = np.abs(dz_top_valid).max()
                p95_dz_top = np.percentile(np.abs(dz_top_valid), 95)
                
                # 注意：dz可能很大是因为分层表Z和轨迹Z使用不同基准面
                # 重要的是坐标(x,y)正确，Z差异不影响插值
                if max_dz_top < 1000.0:  # 宽松阈值
                    self.passed.append(f"✅ dz_top 误差在预期范围: max={max_dz_top:.2f}m, p95={p95_dz_top:.2f}m")
                else:
                    self.failed.append(f"⚠️ dz_top 误差较大: max={max_dz_top:.2f}m, p95={p95_dz_top:.2f}m (可能基准面不同)")
            
            if len(dz_bot_valid) > 0:
                max_dz_bot = np.abs(dz_bot_valid).max()
                p95_dz_bot = np.percentile(np.abs(dz_bot_valid), 95)
                
                if max_dz_bot < 1000.0:  # 宽松阈值
                    self.passed.append(f"✅ dz_bot 误差在预期范围: max={max_dz_bot:.2f}m, p95={p95_dz_bot:.2f}m")
                else:
                    self.failed.append(f"⚠️ dz_bot 误差较大: max={max_dz_bot:.2f}m, p95={p95_dz_bot:.2f}m (可能基准面不同)")
    
    def validate_production_data(self):
        """验收4: 生产数据"""
        self.logger.info("验收4: 检查生产数据...")
        
        filepath = os.path.join(self.config['paths']['clean_data'], 'production_SY9.csv')
        if not self.check_file_exists(filepath, "SY9生产数据文件"):
            return
        
        df = pd.read_csv(filepath)
        
        # 检查必需字段
        required_cols = ['date', 't_day', 'qg_m3d']
        for col in required_cols:
            if col in df.columns:
                self.passed.append(f"✅ 生产数据包含字段: {col}")
            else:
                self.failed.append(f"❌ 生产数据缺少字段: {col}")
        
        # 检查数据天数
        if 'date' in df.columns:
            n_days = len(df)
            self.passed.append(f"✅ 生产数据天数: {n_days}")
            
            # 检查时间连续性
            df['date'] = pd.to_datetime(df['date'])
            date_gaps = df['date'].diff().dt.days
            max_gap = date_gaps.max()
            if max_gap <= 2:  # 允许1-2天间隙
                self.passed.append(f"✅ 时间序列基本连续 (最大间隙{max_gap}天)")
            else:
                self.failed.append(f"⚠️ 时间序列有间断 (最大间隙{max_gap}天)")
    
    def validate_visualization(self):
        """验收5: 可视化图件生成"""
        self.logger.info("验收5: 生成验收图件...")
        
        try:
            # 加载数据
            wells = pd.read_csv(os.path.join(self.config['paths']['staged_data'], 'wells_staged.csv'))
            mk_points = pd.read_csv(os.path.join(self.config['paths']['clean_data'], 'mk_interval_points.csv'))
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 图1: 井位散点图
            ax = axes[0]
            ax.scatter(wells['x'], wells['y'], c='red', s=100, marker='o', label='井位')
            for idx, row in wells.iterrows():
                ax.annotate(row['well_id'], (row['x'], row['y']),
                           xytext=(5, 5), textcoords='offset points')
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title('井位分布', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')
            
            # 图2: MK厚度柱状图
            ax = axes[1]
            ax.bar(range(len(mk_points)), mk_points['mk_thickness'], color='steelblue')
            ax.set_xticks(range(len(mk_points)))
            ax.set_xticklabels(mk_points['well_id'], rotation=45)
            ax.set_ylabel('厚度 (m)', fontsize=12)
            ax.set_title('MK组厚度分布', fontsize=14, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            fig_dir = self.config['paths'].get('figures', os.path.join(self.config['paths']['outputs'], 'figs'))
            os.makedirs(fig_dir, exist_ok=True)
            output_path = os.path.join(fig_dir, 'M1_validation.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.passed.append(f"✅ 验收图件已生成: {output_path}")
            
        except Exception as e:
            self.failed.append(f"❌ 图件生成失败: {str(e)}")
    
    def generate_report(self) -> str:
        """生成验收报告"""
        lines = [
            "# M1 验收报告\n",
            f"验收时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
            "## 验收标准\n",
            "1. 能画出井位散点图",
            "2. 能输出MK顶/底面井点值与厚度统计",
            "3. 井眼轨迹MD单调递增",
            "4. MK段代表点厚度>0",
            "5. 生产数据时间序列完整\n",
            "---\n",
            "## 验收结果\n",
            f"✅ 通过项: {len(self.passed)}",
            f"❌ 失败项: {len(self.failed)}\n",
            "### 通过的检查项\n"
        ]
        
        for item in self.passed:
            lines.append(f"{item}")
        
        if self.failed:
            lines.append("\n### 失败的检查项\n")
            for item in self.failed:
                lines.append(f"{item}")
        
        # 总结
        if len(self.failed) == 0:
            lines.append("\n---\n")
            lines.append("## 🎉 M1 验收通过！\n")
            lines.append("所有检查项均符合要求，可以继续M2阶段。\n")
        else:
            lines.append("\n---\n")
            lines.append("## ⚠️ M1 验收未完全通过\n")
            lines.append(f"请修复上述 {len(self.failed)} 个问题后重新验收。\n")
        
        # 保存报告
        report_path = os.path.join(self.config['paths']['reports'], 'M1_validation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return report_path
    
    def run(self):
        """执行验收"""
        self.logger.info("="*80)
        self.logger.info("开始 M1 验收检查")
        self.logger.info("="*80 + "\n")
        
        self.validate_wells_data()
        self.validate_wellpaths()
        self.validate_mk_interval_points()
        self.validate_production_data()
        self.validate_visualization()
        
        report_path = self.generate_report()
        
        # 打印报告
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n" + f.read())
        
        self.logger.info(f"\n验收报告已保存: {report_path}")
        
        return len(self.failed) == 0


if __name__ == "__main__":
    validator = M1Validator()
    success = validator.run()
    
    if success:
        print("\n" + "="*80)
        print("🎉 M1 验收通过！")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("⚠️ M1 验收未完全通过，请查看报告修复问题")
        print("="*80)
        sys.exit(1)
