"""
M2 验收标准检查脚本
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from shapely.geometry import Point, Polygon

# 脚本位于 src/m2/，需将 src 加入 path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import setup_chinese_support, load_config, setup_logger


class M2Validator:
    """M2验收验证器"""
    
    def __init__(self):
        setup_chinese_support()
        
        # 路径绝对化（与 M1/M2 同一套 project_root 逻辑）
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        self.config = load_config(str(project_root / 'config.yaml'))
        for key, value in self.config['paths'].items():
            self.config['paths'][key] = str(project_root / value)
        
        self.logger = setup_logger('M2_Validator')
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
    
    def validate_boundary(self):
        """验收1: 模型边界"""
        self.logger.info("验收1: 检查模型边界...")
        
        filepath = os.path.join(
            self.config['paths']['geo_data'],
            'boundary',
            'model_boundary.csv'
        )
        
        if not self.check_file_exists(filepath, "边界文件"):
            return
        
        df = pd.read_csv(filepath)
        
        if 'x' in df.columns and 'y' in df.columns:
            self.passed.append(f"✅ 边界包含 {len(df)} 个点")
            
            # 检查边界是否闭合
            if (df.iloc[0]['x'] == df.iloc[-1]['x'] and 
                df.iloc[0]['y'] == df.iloc[-1]['y']):
                self.passed.append(f"✅ 边界闭合")
            else:
                self.failed.append(f"⚠️ 边界未闭合")
        else:
            self.failed.append(f"❌ 边界文件缺少坐标字段")
    
    def validate_surfaces(self):
        """验收2: 顶底面插值结果"""
        self.logger.info("验收2: 检查顶底面插值...")
        
        surfaces = ['mk_top_surface', 'mk_bot_surface', 'mk_thickness']
        
        for surface_name in surfaces:
            filepath = os.path.join(
                self.config['paths']['geo_data'],
                'surfaces',
                f'{surface_name}.csv'
            )
            
            if not self.check_file_exists(filepath, f"{surface_name}"):
                continue
            
            df = pd.read_csv(filepath)
            
            if len(df) > 0:
                self.passed.append(f"✅ {surface_name}: {len(df)} 个网格点")
                
                # 检查数值范围
                if 'z' in df.columns:
                    z_min, z_max = df['z'].min(), df['z'].max()
                    self.passed.append(f"  范围: [{z_min:.2f}, {z_max:.2f}] m")
                    
                    # 对厚度特别检查
                    if surface_name == 'mk_thickness':
                        if df['z'].min() > 0:
                            self.passed.append(f"  ✅ 厚度全部>0")
                        else:
                            self.failed.append(f"  ❌ 存在厚度<=0的点")
            else:
                self.failed.append(f"❌ {surface_name}: 无数据")
    
    def validate_grid(self):
        """验收3: PINN配点网格"""
        self.logger.info("验收3: 检查PINN配点网格...")
        
        # 配点网格
        filepath = os.path.join(
            self.config['paths']['geo_data'],
            'grids',
            'collocation_grid.csv'
        )
        
        if not self.check_file_exists(filepath, "配点网格"):
            return
        
        df = pd.read_csv(filepath)
        
        self.passed.append(f"✅ 配点网格: {len(df)} 个点")
        
        if 'is_near_well' in df.columns:
            n_well_points = df['is_near_well'].sum()
            n_regular = len(df) - n_well_points
            self.passed.append(f"  普通点: {n_regular}, 井周加密点: {n_well_points}")
        
        # 边界点
        boundary_path = os.path.join(
            self.config['paths']['geo_data'],
            'grids',
            'boundary_points.csv'
        )
        
        if self.check_file_exists(boundary_path, "边界采样点"):
            boundary_df = pd.read_csv(boundary_path)
            self.passed.append(f"✅ 边界点: {len(boundary_df)} 个")
    
    def validate_crossvalidation(self):
        """验收4: 交叉验证结果"""
        self.logger.info("验收4: 检查交叉验证结果...")
        
        report_path = os.path.join(
            self.config['paths']['reports'],
            'M2_geological_domain_report.md'
        )
        
        if not os.path.exists(report_path):
            self.failed.append(f"❌ M2报告不存在")
            return
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含交叉验证结果
        if 'MAE' in content and 'RMSE' in content:
            self.passed.append(f"✅ 报告包含交叉验证结果")
        else:
            self.failed.append(f"❌ 报告缺少交叉验证结果")
    
    def validate_visualization(self):
        """验收5: 可视化图件"""
        self.logger.info("验收5: 检查可视化图件...")
        
        figures = [
            'M2_geological_domain.png',
            'M2_collocation_grid.png'
        ]
        
        fig_dir = self.config['paths'].get('figures', os.path.join(self.config['paths']['outputs'], 'figs'))
        for fig_name in figures:
            filepath = os.path.join(fig_dir, fig_name)
            if os.path.exists(filepath):
                self.passed.append(f"✅ 图件存在: {fig_name}")
            else:
                self.failed.append(f"❌ 图件缺失: {fig_name}")
    
    def check_output_quality(self):
        """验收6: 输出质量综合检查"""
        self.logger.info("验收6: 综合质量检查...")
        
        try:
            # 加载MK点和曲面
            mk_points = pd.read_csv(os.path.join(
                self.config['paths']['clean_data'],
                'mk_interval_points.csv'
            ))
            
            top_surface = pd.read_csv(os.path.join(
                self.config['paths']['geo_data'],
                'surfaces',
                'mk_top_surface.csv'
            ))
            
            # 检查：井点处的插值值应该与原始值接近
            for idx, well_row in mk_points.iterrows():
                # 使用顶界交点坐标
                if 'x_top' in well_row and 'y_top' in well_row:
                    wx, wy = well_row['x_top'], well_row['y_top']
                elif 'x_mid' in well_row and 'y_mid' in well_row:
                    wx, wy = well_row['x_mid'], well_row['y_mid']
                else:
                    wx, wy = well_row['x'], well_row['y']
                
                wz = well_row['mk_top_z']
                
                # 找最近网格点
                distances = np.sqrt(
                    (top_surface['x'] - wx)**2 + 
                    (top_surface['y'] - wy)**2
                )
                nearest_idx = distances.idxmin()
                nearest_z = top_surface.loc[nearest_idx, 'z']
                
                error = abs(nearest_z - wz)
                
                # 从配置读取可接受误差阈值
                max_error = self.config['m2_config']['kriging'].get('max_acceptable_error_m', 
                                                                                     self.config['m2_config']['kriging'].get('max_acceptable_error', 50.0))
                
                if error < max_error:
                    self.passed.append(f"  ✅ {well_row['well_id']}: 插值误差 {error:.2f}m")
                else:
                    self.failed.append(f"  ❌ {well_row['well_id']}: 插值误差过大 {error:.2f}m (阈值: {max_error}m)")
        
        except Exception as e:
            self.failed.append(f"❌ 质量检查失败: {str(e)}")
    
    def generate_summary_figure(self):
        """生成验收汇总图"""
        self.logger.info("生成验收汇总图...")
        
        try:
            # 加载数据
            mk_points = pd.read_csv(os.path.join(
                self.config['paths']['clean_data'],
                'mk_interval_points.csv'
            ))
            
            boundary = pd.read_csv(os.path.join(
                self.config['paths']['geo_data'],
                'boundary',
                'model_boundary.csv'
            ))
            
            grid = pd.read_csv(os.path.join(
                self.config['paths']['geo_data'],
                'grids',
                'collocation_grid.csv'
            ))
            
            thickness = pd.read_csv(os.path.join(
                self.config['paths']['geo_data'],
                'surfaces',
                'mk_thickness.csv'
            ))
            
            # 创建图
            fig = plt.figure(figsize=(22, 8))
            
            # 子图1: 井位+边界+网格
            ax1 = plt.subplot(1, 3, 1)
            ax1.scatter(grid['x'], grid['y'], c='lightgray', s=0.5, alpha=0.3, label='配点网格')
            ax1.plot(boundary['x'], boundary['y'], 'b-', linewidth=2, label='模型边界')
            
            # 使用中点坐标绘制井位
            if 'x_mid' in mk_points.columns and 'y_mid' in mk_points.columns:
                ax1.scatter(mk_points['x_mid'], mk_points['y_mid'], c='red', s=100, marker='o', label='井位')
                for idx, row in mk_points.iterrows():
                    ax1.annotate(row['well_id'], (row['x_mid'], row['y_mid']),
                                xytext=(5, 5), textcoords='offset points', fontsize=9, zorder=4)
            else:
                ax1.scatter(mk_points['x'], mk_points['y'], c='red', s=100, marker='o', label='井位')
                for idx, row in mk_points.iterrows():
                    ax1.annotate(row['well_id'], (row['x'], row['y']),
                                xytext=(5, 5), textcoords='offset points', fontsize=9, zorder=4)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('模型域与网格')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')
            ax1.ticklabel_format(style='plain', axis='both', useOffset=False)
            
            # 子图2: 厚度等值图（边界内掩码，与 M2_geological_domain 一致）
            ax2 = plt.subplot(1, 3, 2)
            from scipy.interpolate import griddata
            xi = np.linspace(thickness['x'].min(), thickness['x'].max(), 100)
            yi = np.linspace(thickness['y'].min(), thickness['y'].max(), 100)
            XI, YI = np.meshgrid(xi, yi)
            # 过滤有效厚度点（排除 nan）
            valid = thickness['z'].notna()
            ZI = griddata(
                (thickness.loc[valid, 'x'], thickness.loc[valid, 'y']),
                thickness.loc[valid, 'z'],
                (XI, YI),
                method='cubic'
            )
            # 边界内掩码：边界外置为 nan，热力图只显示有效域
            poly = Polygon(np.column_stack([boundary['x'], boundary['y']]))
            inside_mask = np.zeros(XI.shape, dtype=bool)
            for i in range(XI.shape[0]):
                for j in range(XI.shape[1]):
                    pt = (XI[i, j], YI[i, j])
                    inside_mask[i, j] = poly.contains(Point(pt)) or poly.touches(Point(pt))
            ZI = np.asarray(ZI, dtype=float)
            ZI[~inside_mask] = np.nan
            ax2.set_aspect('equal', adjustable='box')
            ax2.ticklabel_format(style='plain', axis='both', useOffset=False)
            contour = ax2.contourf(XI, YI, ZI, levels=15, cmap='YlOrRd', antialiased=True)
            cbar = plt.colorbar(contour, ax=ax2, pad=0.02)
            cbar.set_label('厚度 (m)')
            cbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            cbar.ax.ticklabel_format(style='plain', useOffset=False)
            
            # 使用中点坐标绘制井位（白边更清晰）
            if 'x_mid' in mk_points.columns and 'y_mid' in mk_points.columns:
                ax2.scatter(mk_points['x_mid'], mk_points['y_mid'],
                           s=60, c='k', edgecolor='white', linewidth=0.8, zorder=3)
                for idx, row in mk_points.iterrows():
                    ax2.annotate(row['well_id'], (row['x_mid'], row['y_mid']),
                                xytext=(5, 5), textcoords='offset points', fontsize=9, zorder=4)
            else:
                ax2.scatter(mk_points['x'], mk_points['y'],
                           s=60, c='k', edgecolor='white', linewidth=0.8, zorder=3)
                for idx, row in mk_points.iterrows():
                    ax2.annotate(row['well_id'], (row['x'], row['y']),
                                xytext=(5, 5), textcoords='offset points', fontsize=9, zorder=4)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('MK组厚度分布')
            ax2.grid(True, alpha=0.3)
            
            # 子图3: 厚度统计
            ax3 = plt.subplot(1, 3, 3)
            ax3.bar(range(len(mk_points)), mk_points['mk_thickness'], color='steelblue', alpha=0.7)
            ax3.axhline(mk_points['mk_thickness'].mean(), color='red', linestyle='--', 
                       label=f"平均: {mk_points['mk_thickness'].mean():.1f}m")
            ax3.set_xticks(range(len(mk_points)))
            ax3.set_xticklabels(mk_points['well_id'], rotation=45)
            ax3.set_ylabel('厚度 (m)')
            ax3.set_title('各井MK组厚度')
            ax3.legend()
            ax3.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            val_fig_dir = self.config['paths'].get('figures', os.path.join(self.config['paths']['outputs'], 'figs'))
            os.makedirs(val_fig_dir, exist_ok=True)
            output_path = os.path.join(val_fig_dir, 'M2_validation_summary.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.passed.append(f"✅ 验收汇总图已生成: {output_path}")
            
        except Exception as e:
            self.failed.append(f"❌ 汇总图生成失败: {str(e)}")
    
    def generate_report(self) -> str:
        """生成验收报告"""
        lines = [
            "# M2 验收报告\n",
            f"验收时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
            "## 验收标准\n",
            "1. 井位散点+模型边界叠加图",
            "2. MK顶面、底面、厚度等值图",
            "3. Leave-one-out交叉验证误差表",
            "4. 配点网格点云图（井周加密+边界均匀）",
            "5. 所有输出文件完整\n",
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
            lines.append("## 🎉 M2 验收通过！\n")
            lines.append("所有检查项均符合要求，地质域构建完成，可以继续后续工作。\n")
        else:
            lines.append("\n---\n")
            lines.append("## ⚠️ M2 验收未完全通过\n")
            lines.append(f"请修复上述 {len(self.failed)} 个问题后重新验收。\n")
        
        # 保存报告
        report_path = os.path.join(
            self.config['paths']['reports'],
            'M2_validation_report.md'
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return report_path
    
    def run(self):
        """执行验收"""
        self.logger.info("="*80)
        self.logger.info("开始 M2 验收检查")
        self.logger.info("="*80 + "\n")
        
        self.validate_boundary()
        self.validate_surfaces()
        self.validate_grid()
        self.validate_crossvalidation()
        self.validate_visualization()
        self.check_output_quality()
        self.generate_summary_figure()
        
        report_path = self.generate_report()
        
        # 打印报告
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n" + f.read())
        
        self.logger.info(f"\n验收报告已保存: {report_path}")
        
        return len(self.failed) == 0


if __name__ == "__main__":
    validator = M2Validator()
    success = validator.run()
    
    if success:
        print("\n" + "="*80)
        print("🎉 M2 验收通过！")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("⚠️ M2 验收未完全通过，请查看报告修复问题")
        print("="*80)
        sys.exit(1)
