#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成训练结果 HTML 预览页面
============================
在浏览器中批量查看所有训练输出图片

使用方法:
    python scripts/generate_html_preview.py
    python scripts/generate_html_preview.py --output custom_preview.html
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 设置 UTF-8 输出（Windows 兼容）
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
    except:
        pass
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

project_root = Path(__file__).resolve().parent.parent

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CPEDC 项目训练结果预览</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 
                         'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 60px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}
        
        .image-card {{
            background: #f8f9fa;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .image-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }}
        
        .image-card-header {{
            padding: 20px;
            background: white;
        }}
        
        .image-card-header h3 {{
            color: #333;
            font-size: 1.2em;
            margin-bottom: 5px;
        }}
        
        .image-card-header .meta {{
            color: #999;
            font-size: 0.9em;
        }}
        
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            align-items: center;
            justify-content: center;
        }}
        
        .modal.active {{
            display: flex;
        }}
        
        .modal img {{
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }}
        
        .close-modal {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }}
        
        .close-modal:hover {{
            color: #667eea;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
        
        @media (max-width: 768px) {{
            .image-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 CPEDC 项目训练结果</h1>
            <p>Physics-Informed Neural Networks for Carbonate Reservoir</p>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{total_images}</div>
                <div class="stat-label">总图片数</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{total_size}</div>
                <div class="stat-label">总大小</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{generation_time}</div>
                <div class="stat-label">生成时间</div>
            </div>
        </div>
        
        <div class="content">
            {sections_html}
        </div>
        
        <div class="footer">
            <p>生成于 {timestamp}</p>
            <p style="margin-top: 10px;">CPEDC Project - 天然气藏数值模拟与人工智能融合研究</p>
        </div>
    </div>
    
    <div class="modal" id="imageModal" onclick="closeModal()">
        <span class="close-modal">&times;</span>
        <img id="modalImage" src="" alt="Full size image">
    </div>
    
    <script>
        function openModal(imgSrc) {{
            document.getElementById('imageModal').classList.add('active');
            document.getElementById('modalImage').src = imgSrc;
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').classList.remove('active');
        }}
        
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
</body>
</html>
"""

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def generate_html(output_path='preview.html'):
    """生成 HTML 预览页面"""
    figs_dir = project_root / 'outputs' / 'mk_pinn_dt_v2' / 'figs'
    
    if not figs_dir.exists():
        print(f"错误: 图片目录不存在: {figs_dir}")
        return
    
    images = sorted([f for f in os.listdir(figs_dir) if f.endswith('.png')])
    
    if not images:
        print("未找到图片文件")
        return
    
    # 按模块分组
    sections = {
        'M1': {'title': 'M1 - 数据验证', 'images': []},
        'M2': {'title': 'M2 - 地质建模', 'images': []},
        'M3': {'title': 'M3 - PVT & 相渗', 'images': []},
        'M4': {'title': 'M4 - 压力场训练', 'images': []},
        'M5': {'title': 'M5 - 井藏耦合同化', 'images': []},
        'M6': {'title': 'M6 - 不确定性量化 & 消融实验', 'images': []},
        'M7': {'title': 'M7 - 水侵策略优化', 'images': []},
    }
    
    total_size = 0
    for img_name in images:
        img_path = figs_dir / img_name
        size = img_path.stat().st_size
        total_size += size
        
        prefix = img_name.split('_')[0]
        if prefix in sections:
            sections[prefix]['images'].append({
                'name': img_name,
                'title': img_name.replace('_', ' ').replace('.png', ''),
                'size': format_size(size),
                'path': f'figs/{img_name}'
            })
    
    # 生成各部分 HTML
    sections_html = ''
    for module_id, section_data in sections.items():
        if not section_data['images']:
            continue
        
        sections_html += f'<div class="section">\n'
        sections_html += f'    <h2 class="section-title">{section_data["title"]}</h2>\n'
        sections_html += f'    <div class="image-grid">\n'
        
        for img in section_data['images']:
            sections_html += f'''
        <div class="image-card">
            <div class="image-card-header">
                <h3>{img["title"]}</h3>
                <div class="meta">{img["size"]}</div>
            </div>
            <img src="{img["path"]}" alt="{img["title"]}" onclick="openModal('{img["path"]}')" loading="lazy">
        </div>
'''
        
        sections_html += '    </div>\n'
        sections_html += '</div>\n\n'
    
    # 填充模板
    html_content = HTML_TEMPLATE.format(
        total_images=len(images),
        total_size=format_size(total_size),
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M'),
        timestamp=datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'),
        sections_html=sections_html
    )
    
    # 写入文件
    output_file = project_root / 'outputs' / 'mk_pinn_dt_v2' / output_path
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML 预览页面已生成: {output_file}")
    print(f"✓ 共包含 {len(images)} 张图片，总大小 {format_size(total_size)}")
    print(f"\n在浏览器中打开:")
    print(f"  {output_file}")
    print(f"\n或运行命令:")
    print(f"  start {output_file}")

def main():
    parser = argparse.ArgumentParser(description='生成训练结果 HTML 预览')
    parser.add_argument('--output', default='preview.html', help='输出文件名')
    args = parser.parse_args()
    
    generate_html(args.output)

if __name__ == '__main__':
    main()
