"""
M1: 数据层与坐标统一
功能：
1. 加载原始数据
2. 统一字段名和单位
3. 坐标基准统一
4. 井斜数据转换为3D轨迹
5. 生成数据质量报告
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path  # 核心修复工具
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 脚本位于 src/m1/，需将 src 加入 path
import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import (
    setup_chinese_support, load_config, setup_logger,
    ensure_dir, detect_outliers_iqr, calculate_statistics,
    write_markdown_report, DataValidator
)


class M1_DataProcessor:
    """M1数据处理器"""

    def __init__(self, config_filename: str = 'config.yaml'):
        """初始化"""
        setup_chinese_support()

        # --- 🚀 2026年 路径修复核心逻辑 ---
        # 1. 获取当前脚本的绝对路径 (例如 .../src/m1_data_processor.py)
        current_file_path = Path(__file__).resolve()

        # 2. 智能回溯项目根目录（脚本在 src/m1/ 时，根目录为 .parent.parent.parent）
        project_root = current_file_path.parent.parent.parent

        # 3. 拼接配置文件的绝对路径
        config_path = project_root / config_filename

        # 双重保险：如果自动推导失败，尝试直接读取（兼容某些特殊打包环境）
        if not config_path.exists():
            config_path = Path(config_filename).resolve()

        if not config_path.exists():
            # 抛出更详细的错误，告诉用户代码到底在哪找文件
            raise FileNotFoundError(
                f"❌ 致命错误：找不到配置文件！\n"
                f"代码试图在以下位置寻找: {config_path}\n"
                f"请确认 'config.yaml' 是否确实位于项目根目录: {project_root}"
            )
        # ------------------------------------

        self.logger = setup_logger('M1_DataProcessor')
        self.logger.info(f"📂 成功定位配置文件: {config_path}")

        # 加载配置
        self.config = load_config(str(config_path))

        # --- 🚀 路径标准化 ---
        # 配置文件里的路径（如 "data/raw"）是相对的
        # 我们必须把它们全部转换为基于 project_root 的绝对路径
        # 否则后续读取 csv 时还会报 FileNotFoundError
        for key, value in self.config['paths'].items():
            self.config['paths'][key] = str(project_root / value)

        self.validator = DataValidator(self.logger)

        # 创建输出目录
        for path_key in ['staged_data', 'clean_data', 'reports']:
            ensure_dir(self.config['paths'][path_key])

        self.logger.info("="*80)
        self.logger.info("M1 数据处理器初始化完成")
        self.logger.info("="*80)

    def load_well_locations(self) -> pd.DataFrame:
        """
        加载井位数据，并添加归一化坐标

        Returns:
            标准化的井位DataFrame（含归一化坐标）
        """
        self.logger.info("正在加载井位数据...")

        # 读取附表1（跳过前3行，使用自定义列名）
        filepath = os.path.join(self.config['paths']['raw_data'], '附表1-井位数据.csv')

        if not os.path.exists(filepath):
             self.logger.error(f"❌ 找不到井位文件: {filepath}")
             # 这里可以抛出异常，或者返回空df，视业务逻辑而定
             raise FileNotFoundError(f"找不到原始数据文件: {filepath}")

        df = pd.read_csv(filepath, encoding='utf-8', skiprows=3,
                        names=['序号', 'well_id', 'x', 'y', 'kb_elev', 'total_depth'])

        # 移除空行
        df = df.dropna(subset=['well_id'])

        # 如果列名不同，尝试模糊匹配
        if 'well_id' not in df.columns:
            for col in df.columns:
                if '井' in col or 'well' in col.lower():
                    df = df.rename(columns={col: 'well_id'})
                    break

        # 统一well_id格式
        df['well_id'] = df['well_id'].astype(str).str.strip().str.upper()

        # 验证
        self.validator.validate_not_null(df['well_id'], 'well_id')
        self.validator.validate_unique(df['well_id'], 'well_id')

        # 检测坐标离群值
        if 'x' in df.columns and 'y' in df.columns:
            outliers_x = detect_outliers_iqr(df['x'].values)
            outliers_y = detect_outliers_iqr(df['y'].values)
            if outliers_x.any() or outliers_y.any():
                self.logger.warning(f"发现坐标离群值: X方向{outliers_x.sum()}个, Y方向{outliers_y.sum()}个")

        # ========== 新增：坐标归一化 ==========
        if 'x' in df.columns and 'y' in df.columns:
            # 计算归一化参数（Min-Max归一化到[0,1]）
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            
            # 归一化坐标
            df['x_norm'] = (df['x'] - x_min) / (x_max - x_min)
            df['y_norm'] = (df['y'] - y_min) / (y_max - y_min)
            
            # 保存归一化参数到配置（供后续模块使用）
            if 'm1_config' not in self.config:
                self.config['m1_config'] = {}
            
            self.config['m1_config']['normalization_params'] = {
                'x_min': float(x_min),
                'x_max': float(x_max),
                'y_min': float(y_min),
                'y_max': float(y_max)
            }
            
            self.logger.info(f"✅ 坐标归一化完成:")
            self.logger.info(f"   X: [{x_min:.2f}, {x_max:.2f}] → [0, 1]")
            self.logger.info(f"   Y: [{y_min:.2f}, {y_max:.2f}] → [0, 1]")
        # ========================================

        self.logger.info(f"成功加载 {len(df)} 口井的位置数据")

        # 保存到staged
        output_path = os.path.join(self.config['paths']['staged_data'], 'wells_staged.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # BOM 便于 Excel 正确显示中文

        return df

    def load_deviation_survey(self, well_id: str) -> pd.DataFrame:
        """
        加载单井井斜数据（增强版：自动补插井口点、去重、排序）

        Args:
            well_id: 井号

        Returns:
            标准化的井斜DataFrame（保证首测点MD=0）
        """
        filepath = os.path.join(
            self.config['paths']['raw_data'],
            f'附表2-井斜数据__{well_id}.csv'
        )

        if not os.path.exists(filepath):
            self.logger.warning(f"井斜文件不存在: {filepath}")
            return pd.DataFrame()

        # 读取井斜数据：表头为多行（引号内换行），CSV 解析为 1 个逻辑行，故只跳过 1 行
        df = pd.read_csv(filepath, encoding='utf-8', skiprows=1,
                        names=['well_id', 'md_m', 'inc_deg', 'azi_deg', 'tvd_m'])

        # 确保well_id统一
        if 'well_id' in df.columns:
            df['well_id'] = df['well_id'].astype(str).str.strip().str.upper()
        else:
            df['well_id'] = well_id

        # 数值类型转换
        for col in ['md_m', 'inc_deg', 'azi_deg', 'tvd_m']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除含NaN的行
        df = df.dropna(subset=['md_m', 'inc_deg', 'azi_deg'])

        if df.empty:
            self.logger.warning(f"{well_id} 井斜数据为空")
            return df

        # 按MD排序、去重
        df = df.sort_values('md_m').reset_index(drop=True)
        df = df.drop_duplicates(subset=['md_m'], keep='first')

        # 方位角归一化到[0, 360)
        if 'azi_deg' in df.columns:
            df['azi_deg'] = df['azi_deg'] % 360

        # ========== 关键修复：检查并补插井口点 ==========
        md_min = df['md_m'].min()
        if md_min > 0.01:  # 首测点MD不是0（允许1cm误差）
            self.logger.warning(
                f"{well_id} 井斜首测点MD={md_min:.2f}m（非0），自动补插井口点（MD=0, TVD=0）"
            )
            # 补插井口点
            wellhead_row = pd.DataFrame([{
                'well_id': well_id,
                'md_m': 0.0,
                'inc_deg': 0.0,
                'azi_deg': df.loc[0, 'azi_deg'],  # 继承首测点方位
                'tvd_m': 0.0
            }])
            df = pd.concat([wellhead_row, df], ignore_index=True)
            self.logger.info(f"{well_id} 补插井口点后，测点数: {len(df)}")

        return df

    def minimum_curvature(self, md1: float, inc1: float, azi1: float,
                          md2: float, inc2: float, azi2: float) -> Tuple[float, float, float]:
        """
        最小曲率法计算井眼轨迹

        Args:
            md1, inc1, azi1: 上测点的MD(m), 井斜角(度), 方位角(度)
            md2, inc2, azi2: 下测点的MD(m), 井斜角(度), 方位角(度)

        Returns:
            delta_tvd, delta_north, delta_east: 增量(m)
        """
        # 转换为弧度
        inc1_rad = np.radians(inc1)
        inc2_rad = np.radians(inc2)
        azi1_rad = np.radians(azi1)
        azi2_rad = np.radians(azi2)

        # 计算MD增量
        delta_md = md2 - md1

        # 计算dogleg角度
        cos_dl = (np.cos(inc2_rad - inc1_rad) -
                  np.sin(inc1_rad) * np.sin(inc2_rad) *
                  (1 - np.cos(azi2_rad - azi1_rad)))

        # 防止数值误差
        cos_dl = np.clip(cos_dl, -1.0, 1.0)
        dogleg = np.arccos(cos_dl)

        # 计算ratio factor
        if dogleg < 1e-6:  # dogleg接近0
            rf = 1.0
        else:
            rf = 2 / dogleg * np.tan(dogleg / 2)

        # 计算增量
        delta_tvd = 0.5 * delta_md * (np.cos(inc1_rad) + np.cos(inc2_rad)) * rf
        delta_north = 0.5 * delta_md * (
            np.sin(inc1_rad) * np.cos(azi1_rad) +
            np.sin(inc2_rad) * np.cos(azi2_rad)
        ) * rf
        delta_east = 0.5 * delta_md * (
            np.sin(inc1_rad) * np.sin(azi1_rad) +
            np.sin(inc2_rad) * np.sin(azi2_rad)
        ) * rf

        return delta_tvd, delta_north, delta_east

    def calculate_wellpath(self, deviation_df: pd.DataFrame,
                           kb_elev: float, x0: float, y0: float) -> pd.DataFrame:
        """
        计算井眼3D轨迹（修复版：利用tvd_m锚定起点）

        Args:
            deviation_df: 井斜数据（必须已排序、去重）
            kb_elev: 井口标高(m)
            x0, y0: 井口坐标(m)

        Returns:
            包含3D坐标的DataFrame
        """
        df = deviation_df.copy()

        # ========== 修复关键1：利用tvd_m作为起点锚定 ==========
        # 检查是否有tvd_m列且首测点有效
        use_tvd_anchor = False
        if 'tvd_m' in df.columns and not pd.isna(df.loc[0, 'tvd_m']):
            use_tvd_anchor = True
            tvd_anchor = df.loc[0, 'tvd_m']  # 首测点的TVD（通常为0）
        else:
            tvd_anchor = 0.0

        # 初始化累计值（从首测点的tvd_m开始，而非强制0）
        df['tvd_cumsum'] = tvd_anchor
        df['north_cumsum'] = 0.0
        df['east_cumsum'] = 0.0

        # 逐段计算（最小曲率法）
        for i in range(1, len(df)):
            delta_tvd, delta_north, delta_east = self.minimum_curvature(
                df.loc[i-1, 'md_m'], df.loc[i-1, 'inc_deg'], df.loc[i-1, 'azi_deg'],
                df.loc[i, 'md_m'], df.loc[i, 'inc_deg'], df.loc[i, 'azi_deg']
            )

            df.loc[i, 'tvd_cumsum'] = df.loc[i-1, 'tvd_cumsum'] + delta_tvd
            df.loc[i, 'north_cumsum'] = df.loc[i-1, 'north_cumsum'] + delta_north
            df.loc[i, 'east_cumsum'] = df.loc[i-1, 'east_cumsum'] + delta_east

        # ========== 修复关键2：验证tvd_cumsum与tvd_m的一致性 ==========
        if use_tvd_anchor:
            # 计算累计误差（应接近0）
            tvd_error = (df['tvd_cumsum'] - df['tvd_m']).abs()
            max_error = tvd_error.max()
            if max_error > 1.0:  # 误差超过1m则警告
                well_id = df.loc[0, 'well_id'] if 'well_id' in df.columns else 'Unknown'
                self.logger.warning(
                    f"{well_id} TVD累计与井斜表tvd_m的最大误差: {max_error:.2f}m"
                )

        # 计算最终坐标
        # z = 井口标高 - TVD (向下为正的垂深)
        df['z'] = kb_elev - df['tvd_cumsum']
        df['x'] = x0 + df['east_cumsum']
        df['y'] = y0 + df['north_cumsum']

        return df

    def process_all_wellpaths(self, wells_df: pd.DataFrame) -> pd.DataFrame:
        """
        处理所有井的轨迹

        Args:
            wells_df: 井位数据

        Returns:
            所有井的轨迹点数据
        """
        self.logger.info("正在计算井眼3D轨迹...")

        all_paths = []

        for idx, row in wells_df.iterrows():
            well_id = row['well_id']
            self.logger.info(f"  处理 {well_id}...")

            # 加载井斜数据
            dev_df = self.load_deviation_survey(well_id)
            if dev_df.empty:
                continue

            # 获取井口参数
            kb_elev = row.get('kb_elev', 0.0)  # 默认0
            x0 = row.get('x', 0.0)
            y0 = row.get('y', 0.0)

            # 计算轨迹
            path_df = self.calculate_wellpath(dev_df, kb_elev, x0, y0)
            all_paths.append(path_df)

        # 合并所有轨迹（空列表保护）
        if not all_paths:
            self.logger.warning("无井斜数据可生成轨迹，返回空 DataFrame")
            return pd.DataFrame()
        
        wellpaths = pd.concat(all_paths, ignore_index=True)

        self.logger.info(f"成功计算 {len(all_paths)} 口井的轨迹，共 {len(wellpaths)} 个测点")

        # 保存
        output_path = os.path.join(
            self.config['paths']['clean_data'],
            'wellpath_stations.csv'
        )
        wellpaths.to_csv(output_path, index=False, encoding='utf-8-sig')  # BOM 便于 Excel 正确显示中文

        return wellpaths

    def load_layer_data(self) -> pd.DataFrame:
        """
        加载分层数据

        Returns:
            标准化的分层DataFrame
        """
        self.logger.info("正在加载分层数据...")

        filepath = os.path.join(self.config['paths']['raw_data'], '附表4-分层数据.csv')
        df = pd.read_csv(filepath, encoding='utf-8')

        # 标准化列名（注意：CSV文件使用中文括号）
        column_map = {
            '井号': 'well_id',
            '井区': 'block',
            'MK顶界钻井深度（m）': 'mk_top_md',
            'MK底界钻井深度（m）': 'mk_bot_md',
            'MK顶界垂直深度（m）': 'mk_top_tvd',
            'MK底界垂直深度（m）': 'mk_bot_tvd',
            'MK顶界海拔（m）': 'mk_top_z',
            'MK底界海拔（m）': 'mk_bot_z'
        }

        # 尝试重命名
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # 统一well_id
        if 'well_id' in df.columns:
            df['well_id'] = df['well_id'].astype(str).str.strip().str.upper()

        # 计算厚度
        if 'mk_top_z' in df.columns and 'mk_bot_z' in df.columns:
            df['mk_thickness'] = df['mk_top_z'] - df['mk_bot_z']

            # 验证厚度为正
            invalid_thickness = df['mk_thickness'] <= 0
            if invalid_thickness.any():
                self.logger.error(f"发现 {invalid_thickness.sum()} 口井厚度<=0，数据异常！")

        self.logger.info(f"成功加载 {len(df)} 口井的分层数据")

        return df

    def extract_mk_interval_points(self, wellpaths: pd.DataFrame,
                                    layers: pd.DataFrame) -> pd.DataFrame:
        """
        提取MK段代表点 - 包含顶界、底界和中点的完整坐标
        
        改造要求：
        1. 轨迹必须按MD排序、去重、单调性检查
        2. 输出顶/底界交点坐标 (x_top, y_top), (x_bot, y_bot)
        3. 在MD轴上插值（不是Z轴）
        4. 越界检查并标记
        5. 输出QA字段 dz_top, dz_bot

        Args:
            wellpaths: 井眼轨迹数据
            layers: 分层数据

        Returns:
            MK段代表点数据（含顶/底/中点坐标）
        """
        self.logger.info("正在提取MK段代表点...")

        mk_points = []
        problematic_wells = []

        for idx, layer_row in layers.iterrows():
            well_id = layer_row['well_id']

            # 获取该井的轨迹
            well_path = wellpaths[wellpaths['well_id'] == well_id].copy()
            if well_path.empty:
                self.logger.warning(f"{well_id} 没有轨迹数据")
                problematic_wells.append((well_id, "no_trajectory"))
                continue

            # ========== A. 输入前置校验（必须） ==========
            # 1. 按 MD 升序排序
            well_path = well_path.sort_values('md_m').reset_index(drop=True)

            # 2. 去重 MD（保留第一个）
            md_before = len(well_path)
            well_path = well_path.drop_duplicates(subset=['md_m'], keep='first')
            if len(well_path) < md_before:
                self.logger.warning(f"{well_id} MD去重：{md_before} → {len(well_path)} 点")

            # 3. 检查单调性
            if len(well_path) > 1:
                md_diff = np.diff(well_path['md_m'].values)
                if np.any(md_diff <= 0):
                    self.logger.error(f"{well_id} MD不单调！轨迹异常，跳过")
                    problematic_wells.append((well_id, "non_monotonic_md"))
                    continue

            # 获取MK顶底界的MD和Z值
            md_top = layer_row.get('mk_top_md', np.nan)
            md_bot = layer_row.get('mk_bot_md', np.nan)
            mk_top_z = layer_row.get('mk_top_z', np.nan)
            mk_bot_z = layer_row.get('mk_bot_z', np.nan)

            if pd.isna(md_top) or pd.isna(md_bot) or pd.isna(mk_top_z) or pd.isna(mk_bot_z):
                self.logger.warning(f"{well_id} MK顶底界数据缺失")
                problematic_wells.append((well_id, "missing_mk_bounds"))
                continue

            # MD范围
            md_min = well_path['md_m'].min()
            md_max = well_path['md_m'].max()

            # ========== B. 顶界交点坐标（必须输出） ==========
            out_of_range_top = False
            if md_top < md_min or md_top > md_max:
                self.logger.warning(f"{well_id} md_top={md_top:.1f} 越界 [{md_min:.1f}, {md_max:.1f}]")
                out_of_range_top = True
                x_top, y_top, z_top_traj = np.nan, np.nan, np.nan
            else:
                # 在MD轴上插值求交点
                x_top = np.interp(md_top, well_path['md_m'], well_path['x'])
                y_top = np.interp(md_top, well_path['md_m'], well_path['y'])
                z_top_traj = np.interp(md_top, well_path['md_m'], well_path['z'])

            # ========== C. 底界交点坐标（必须输出） ==========
            out_of_range_bot = False
            if md_bot < md_min or md_bot > md_max:
                self.logger.warning(f"{well_id} md_bot={md_bot:.1f} 越界 [{md_min:.1f}, {md_max:.1f}]")
                out_of_range_bot = True
                x_bot, y_bot, z_bot_traj = np.nan, np.nan, np.nan
            else:
                # 在MD轴上插值求交点
                x_bot = np.interp(md_bot, well_path['md_m'], well_path['x'])
                y_bot = np.interp(md_bot, well_path['md_m'], well_path['y'])
                z_bot_traj = np.interp(md_bot, well_path['md_m'], well_path['z'])

            # ========== D. 中点（保留，用于井控/井周加密） ==========
            md_mid = (md_top + md_bot) / 2
            z_mid = (mk_top_z + mk_bot_z) / 2

            if md_mid >= md_min and md_mid <= md_max:
                x_mid = np.interp(md_mid, well_path['md_m'], well_path['x'])
                y_mid = np.interp(md_mid, well_path['md_m'], well_path['y'])
            else:
                # 中点越界则用top和bot的平均（如果它们有效）
                if not out_of_range_top and not out_of_range_bot:
                    x_mid = (x_top + x_bot) / 2
                    y_mid = (y_top + y_bot) / 2
                else:
                    x_mid, y_mid = np.nan, np.nan

            # ========== E. QA字段 ==========
            # dz = 轨迹上的Z - 分层表的Z（理想应该接近0）
            dz_top = z_top_traj - mk_top_z if not out_of_range_top else np.nan
            dz_bot = z_bot_traj - mk_bot_z if not out_of_range_bot else np.nan

            mk_points.append({
                'well_id': well_id,
                # 顶界
                'md_top': md_top,
                'mk_top_z': mk_top_z,
                'x_top': x_top,
                'y_top': y_top,
                'z_top_traj': z_top_traj,
                # 底界
                'md_bot': md_bot,
                'mk_bot_z': mk_bot_z,
                'x_bot': x_bot,
                'y_bot': y_bot,
                'z_bot_traj': z_bot_traj,
                # 中点
                'md_mid': md_mid,
                'z_mid': z_mid,
                'x_mid': x_mid,
                'y_mid': y_mid,
                # 厚度
                'mk_thickness': mk_top_z - mk_bot_z,
                # QA
                'dz_top': dz_top,
                'dz_bot': dz_bot,
                'out_of_range_top': out_of_range_top,
                'out_of_range_bot': out_of_range_bot
            })

        mk_df = pd.DataFrame(mk_points)

        # 报告异常井
        if problematic_wells:
            self.logger.warning(f"异常井清单：{problematic_wells}")

        self.logger.info(f"成功提取 {len(mk_df)} 口井的MK段代表点")

        # 统计越界情况
        n_out_top = mk_df['out_of_range_top'].sum()
        n_out_bot = mk_df['out_of_range_bot'].sum()
        if n_out_top > 0 or n_out_bot > 0:
            self.logger.warning(f"越界统计: top={n_out_top}, bot={n_out_bot}")

        # ========== 新增：对MK段坐标进行归一化 ==========
        if 'm1_config' in self.config and 'normalization_params' in self.config['m1_config']:
            norm_params = self.config['m1_config']['normalization_params']
            x_min = norm_params['x_min']
            x_max = norm_params['x_max']
            y_min = norm_params['y_min']
            y_max = norm_params['y_max']
            
            # 归一化顶界坐标
            mk_df['x_top_norm'] = (mk_df['x_top'] - x_min) / (x_max - x_min)
            mk_df['y_top_norm'] = (mk_df['y_top'] - y_min) / (y_max - y_min)
            
            # 归一化底界坐标
            mk_df['x_bot_norm'] = (mk_df['x_bot'] - x_min) / (x_max - x_min)
            mk_df['y_bot_norm'] = (mk_df['y_bot'] - y_min) / (y_max - y_min)
            
            # 归一化中点坐标
            mk_df['x_mid_norm'] = (mk_df['x_mid'] - x_min) / (x_max - x_min)
            mk_df['y_mid_norm'] = (mk_df['y_mid'] - y_min) / (y_max - y_min)
            
            self.logger.info(f"✅ MK段坐标归一化完成（使用统一归一化参数）")
        else:
            self.logger.warning("⚠️ 未找到归一化参数，MK段坐标未归一化")
        # ========================================

        # 保存
        output_path = os.path.join(
            self.config['paths']['clean_data'],
            'mk_interval_points.csv'
        )
        mk_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # BOM 便于 Excel 正确显示中文

        return mk_df

    def load_production_data(self, well_id: str = 'SY9') -> pd.DataFrame:
        """
        加载生产数据

        Args:
            well_id: 井号（默认SY9）

        Returns:
            清洗后的生产数据
        """
        self.logger.info(f"正在加载 {well_id} 生产数据...")

        filepath = os.path.join(
            self.config['paths']['raw_data'],
            f'附表10-{well_id}单井日生产数据.csv'
        )

        df = pd.read_csv(filepath, encoding='utf-8')

        # 标准化列名
        column_map = {
            '生产日期': 'date',
            '日产气量_(m^3)': 'qg_m3d',
            '日产水量_(t)': 'qw_td',
            '套压(MPa)_关井': 'casing_p_shut',
            '套压(MPa)_平均': 'casing_p_avg',
            '油压(MPa)_关井': 'tubing_p_shut',
            '油压(MPa)_平均': 'tubing_p_avg',
        }

        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # 日期解析
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')

            # 去重（同一天取平均）
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df.groupby('date', as_index=False)[numeric_cols].mean()

            # 计算模型时间（天）
            t0_date = df['date'].min()
            df['t_day'] = (df['date'] - t0_date).dt.days

            # 保存t0_date到配置
            self.config['m1_config']['t0_date'] = t0_date.strftime('%Y-%m-%d')

        df['well_id'] = well_id

        # ===== v4.0: 工况标签 (operating condition) =====
        # 基于生产时间和产气量自动分类: open / throttled / shut_in
        # 供 M5 分权重训练使用 (关井段低权重, 节流段中权重, 正常开井高权重)
        prod_hours_col = '生产时间_(H)'
        if prod_hours_col in df.columns:
            prod_hours = pd.to_numeric(df[prod_hours_col], errors='coerce').fillna(0.0)
            qg = pd.to_numeric(df.get('qg_m3d', pd.Series(dtype=float)), errors='coerce').fillna(0.0)

            conditions = []
            for h, q in zip(prod_hours, qg):
                if h < 0.5 or q < 100.0:
                    conditions.append('shut_in')
                elif h < 18.0 or q < 5e4:
                    conditions.append('throttled')
                else:
                    conditions.append('open')
            df['op_condition'] = conditions

            # 有效产气标记 (非关井期的观测视为可信)
            df['qg_valid'] = (df['op_condition'] != 'shut_in').astype(int)

            n_open = sum(1 for c in conditions if c == 'open')
            n_throttle = sum(1 for c in conditions if c == 'throttled')
            n_shutin = sum(1 for c in conditions if c == 'shut_in')
            self.logger.info(
                f"  工况标签: open={n_open}, throttled={n_throttle}, shut_in={n_shutin}"
            )
        else:
            self.logger.warning(f"  未找到 '{prod_hours_col}' 列, 跳过工况标签")

        self.logger.info(f"成功加载 {len(df)} 天的生产数据")

        # 保存
        output_path = os.path.join(
            self.config['paths']['clean_data'],
            f'production_{well_id}.csv'
        )
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # BOM 便于 Excel 正确显示中文

        return df

    def generate_data_quality_report(self, wells_df: pd.DataFrame,
                                      wellpaths: pd.DataFrame,
                                      mk_points: pd.DataFrame,
                                      prod_df: pd.DataFrame) -> str:
        """
        生成数据质量报告

        Returns:
            报告文件路径
        """
        self.logger.info("正在生成数据质量报告...")

        lines = [
            "# M1 数据质量报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
            "## 1. 井位数据\n",
            f"- 总井数: {len(wells_df)}",
            f"- 井号列表: {', '.join(wells_df['well_id'].tolist())}\n",
        ]

        # 坐标统计
        if 'x' in wells_df.columns and 'y' in wells_df.columns:
            lines.extend([
                "### 坐标范围（原始值）",
                f"- X: [{wells_df['x'].min():.2f}, {wells_df['x'].max():.2f}] m",
                f"- Y: [{wells_df['y'].min():.2f}, {wells_df['y'].max():.2f}] m\n"
            ])
        
        # ========== 新增：归一化信息 ==========
        if 'x_norm' in wells_df.columns and 'y_norm' in wells_df.columns:
            lines.extend([
                "### 坐标归一化",
                "✅ 已对坐标进行Min-Max归一化到[0, 1]区间",
                f"- X归一化范围: [{wells_df['x_norm'].min():.4f}, {wells_df['x_norm'].max():.4f}]",
                f"- Y归一化范围: [{wells_df['y_norm'].min():.4f}, {wells_df['y_norm'].max():.4f}]"
            ])
            
            # 如果有归一化参数，显示详细信息
            if 'm1_config' in self.config and 'normalization_params' in self.config['m1_config']:
                norm_params = self.config['m1_config']['normalization_params']
                lines.extend([
                    "\n**归一化参数**:",
                    f"- x_min: {norm_params['x_min']:.2f} m",
                    f"- x_max: {norm_params['x_max']:.2f} m",
                    f"- y_min: {norm_params['y_min']:.2f} m",
                    f"- y_max: {norm_params['y_max']:.2f} m",
                    "\n**反归一化公式**:",
                    "- x_原始 = x_norm × (x_max - x_min) + x_min",
                    "- y_原始 = y_norm × (y_max - y_min) + y_min\n"
                ])
        # ======================================

        # 井眼轨迹
        lines.extend([
            "## 2. 井眼轨迹数据\n",
            f"- 总测点数: {len(wellpaths)}",
        ])

        for well_id in wells_df['well_id']:
            well_data = wellpaths[wellpaths['well_id'] == well_id]
            lines.append(f"- {well_id}: {len(well_data)} 个测点")
        lines.append("")

        # MK段数据
        lines.extend([
            "## 3. MK组分层数据\n",
            f"- 有效井数: {len(mk_points)}",
            f"- 平均厚度: {mk_points['mk_thickness'].mean():.2f} m",
            f"- 厚度范围: [{mk_points['mk_thickness'].min():.2f}, {mk_points['mk_thickness'].max():.2f}] m\n"
        ])

        # ========== 新增：QA统计（dz分析） ==========
        lines.extend([
            "### 3.1 轨迹-分层匹配质量（QA: dz统计）\n",
            "**dz定义**: 轨迹z坐标 - 分层海拔，理想应接近0（表示轨迹与分层海拔一致）\n"
        ])

        # 统计dz_top
        if 'dz_top' in mk_points.columns:
            dz_top_valid = mk_points['dz_top'].dropna()
            if len(dz_top_valid) > 0:
                lines.extend([
                    "**dz_top统计**:",
                    f"- 均值: {dz_top_valid.mean():.4f} m",
                    f"- 中位数: {dz_top_valid.median():.4f} m",
                    f"- 标准差: {dz_top_valid.std():.4f} m",
                    f"- 范围: [{dz_top_valid.min():.4f}, {dz_top_valid.max():.4f}] m",
                    f"- 95%分位: {dz_top_valid.quantile(0.95):.4f} m\n"
                ])

        # 统计dz_bot
        if 'dz_bot' in mk_points.columns:
            dz_bot_valid = mk_points['dz_bot'].dropna()
            if len(dz_bot_valid) > 0:
                lines.extend([
                    "**dz_bot统计**:",
                    f"- 均值: {dz_bot_valid.mean():.4f} m",
                    f"- 中位数: {dz_bot_valid.median():.4f} m",
                    f"- 标准差: {dz_bot_valid.std():.4f} m",
                    f"- 范围: [{dz_bot_valid.min():.4f}, {dz_bot_valid.max():.4f}] m",
                    f"- 95%分位: {dz_bot_valid.quantile(0.95):.4f} m\n"
                ])

        # 列出 |dz| > 1m 的异常井
        threshold = 1.0  # 阈值：1m
        if 'dz_top' in mk_points.columns and 'dz_bot' in mk_points.columns:
            abnormal_wells = mk_points[
                (mk_points['dz_top'].abs() > threshold) | 
                (mk_points['dz_bot'].abs() > threshold)
            ]
            if len(abnormal_wells) > 0:
                lines.extend([
                    f"**警告：发现 {len(abnormal_wells)} 口井 |dz| > {threshold}m（需检查井斜数据）**:\n"
                ])
                for idx, row in abnormal_wells.iterrows():
                    lines.append(
                        f"- {row['well_id']}: dz_top={row['dz_top']:.2f}m, "
                        f"dz_bot={row['dz_bot']:.2f}m"
                    )
                lines.append("")
            else:
                lines.extend([
                    f"✅ 所有井 |dz| < {threshold}m，轨迹与分层海拔匹配良好！\n"
                ])
        # ========================================

        # 生产数据
        lines.extend([
            "## 4. 生产数据 (SY9)\n",
            f"- 数据天数: {len(prod_df)}",
            f"- 起始日期: {prod_df['date'].min()}",
            f"- 结束日期: {prod_df['date'].max()}\n"
        ])

        # 验证结果
        lines.extend([
            "## 5. 数据验证\n",
            self.validator.get_report()
        ])

        # 写入文件
        report_path = os.path.join(
            self.config['paths']['reports'],
            'M1_data_quality_report.md'
        )
        write_markdown_report(lines, report_path)

        self.logger.info(f"数据质量报告已保存: {report_path}")

        return report_path

    def run(self):
        """执行完整的M1流程"""
        self.logger.info("\n" + "="*80)
        self.logger.info("开始执行 M1 数据处理流程")
        self.logger.info("="*80 + "\n")

        try:
            # 1. 加载井位
            wells_df = self.load_well_locations()

            # 2. 计算井眼轨迹
            wellpaths = self.process_all_wellpaths(wells_df)

            # 3. 加载分层数据
            layers = self.load_layer_data()

            # 4. 提取MK段代表点
            mk_points = self.extract_mk_interval_points(wellpaths, layers)

            # 5. 加载生产数据
            prod_df = self.load_production_data('SY9')

            # 6. 生成报告
            report_path = self.generate_data_quality_report(
                wells_df, wellpaths, mk_points, prod_df
            )

            # 7. 保存归一化参数（供后续模块使用）
            if 'm1_config' in self.config and 'normalization_params' in self.config['m1_config']:
                import json
                norm_params_path = os.path.join(
                    self.config['paths']['clean_data'],
                    'normalization_params.json'
                )
                with open(norm_params_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config['m1_config']['normalization_params'], f, indent=2, ensure_ascii=False)
                self.logger.info(f"✅ 归一化参数已保存: {norm_params_path}")

            self.logger.info("\n" + "="*80)
            self.logger.info("✅ M1 数据处理流程执行完成！")
            self.logger.info("="*80 + "\n")

            return {
                'wells': wells_df,
                'wellpaths': wellpaths,
                'mk_points': mk_points,
                'production': prod_df,
                'report': report_path
            }

        except Exception as e:
            self.logger.error(f"❌ M1流程执行失败: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    processor = M1_DataProcessor()
    results = processor.run()
    print("\n✅ M1模块测试完成")