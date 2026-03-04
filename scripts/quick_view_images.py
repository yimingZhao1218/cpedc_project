#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速图片查看工具
==================
用于在 Cursor 图片预览失败时，使用系统默认查看器查看训练结果图片

使用方法:
    python scripts/quick_view_images.py
    python scripts/quick_view_images.py --all  # 一次性打开所有图片
    python scripts/quick_view_images.py --filter M5  # 只显示包含 M5 的图片
"""

import os
import sys
import argparse
from pathlib import Path

# 设置 UTF-8 输出（Windows 兼容）
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
    except:
        pass
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

project_root = Path(__file__).resolve().parent.parent

def list_images(filter_pattern=None):
    """列出所有可用图片"""
    figs_dir = project_root / 'outputs' / 'mk_pinn_dt_v2' / 'figs'
    
    if not figs_dir.exists():
        print(f"错误: 图片目录不存在: {figs_dir}")
        return []
    
    images = sorted([f for f in os.listdir(figs_dir) if f.endswith('.png')])
    
    if filter_pattern:
        images = [img for img in images if filter_pattern in img]
    
    return figs_dir, images

def open_image(img_path):
    """使用系统默认应用打开图片"""
    if sys.platform == 'win32':
        os.startfile(img_path)
    elif sys.platform == 'darwin':  # macOS
        os.system(f'open "{img_path}"')
    else:  # Linux
        os.system(f'xdg-open "{img_path}"')

def main():
    parser = argparse.ArgumentParser(description='快速查看训练结果图片')
    parser.add_argument('--all', action='store_true', help='打开所有图片')
    parser.add_argument('--filter', type=str, help='过滤图片名称（如: M5, M6）')
    args = parser.parse_args()
    
    figs_dir, images = list_images(args.filter)
    
    if not images:
        print("未找到图片文件")
        return
    
    print(f"{'='*60}")
    print(f"图片目录: {figs_dir}")
    print(f"共找到 {len(images)} 张图片")
    print(f"{'='*60}\n")
    
    if args.all:
        print(f"正在打开所有 {len(images)} 张图片...")
        for img_name in images:
            img_path = figs_dir / img_name
            open_image(str(img_path))
        print("完成!")
        return
    
    # 交互式选择
    for i, img_name in enumerate(images, 1):
        # 计算文件大小
        img_path = figs_dir / img_name
        size_kb = img_path.stat().st_size / 1024
        print(f"  {i:2d}. {img_name:<40s} ({size_kb:>7.1f} KB)")
    
    print(f"\n{'='*60}")
    while True:
        choice = input("输入图片编号查看 (输入 'q' 退出, 'a' 全部打开): ").strip()
        
        if choice.lower() == 'q':
            print("退出")
            break
        elif choice.lower() == 'a':
            print(f"正在打开所有 {len(images)} 张图片...")
            for img_name in images:
                img_path = figs_dir / img_name
                open_image(str(img_path))
            print("完成!")
            break
        elif choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(images):
                img_path = figs_dir / images[idx-1]
                print(f"正在打开: {images[idx-1]}")
                open_image(str(img_path))
            else:
                print(f"错误: 请输入 1-{len(images)} 之间的数字")
        else:
            print("无效输入，请输入数字、'a' 或 'q'")

if __name__ == '__main__':
    main()
