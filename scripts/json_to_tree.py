#!/usr/bin/env python3
"""将 resolved_config_m5.json 转为树形文本输出。"""
import json
import sys
from pathlib import Path

def tree(obj, prefix=''):
    lines = []
    if isinstance(obj, dict):
        items = list(obj.items())
        for i, (k, v) in enumerate(items):
            last = i == len(items) - 1
            branch = '+-- ' if last else '|-- '
            lines.append(prefix + branch + str(k))
            if isinstance(v, (dict, list)) and v:
                ext = '    ' if last else '|   '
                lines.extend(tree(v, prefix + ext))
            elif not isinstance(v, (dict, list)):
                s = repr(v)
                if len(s) > 72:
                    s = s[:69] + '...'
                lines.append(prefix + ('    ' if last else '|   ') + '= ' + s)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            last = i == len(obj) - 1
            branch = '+-- ' if last else '|-- '
            if isinstance(v, (dict, list)):
                lines.append(prefix + branch + '[' + str(i) + ']')
                ext = '    ' if last else '|   '
                lines.extend(tree(v, prefix + ext))
            else:
                s = repr(v)
                if len(s) > 72:
                    s = s[:69] + '...'
                lines.append(prefix + branch + s)
    return lines


def main():
    src = Path(__file__).resolve().parent.parent / 'outputs' / 'mk_pinn_dt_v2' / 'resolved_config_m5.json'
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    root_name = src.name
    out_lines = [root_name, ''] + tree(data)
    out_path = src.parent / (src.stem + '_tree.txt')
    text = '\n'.join(out_lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('Tree saved to:', out_path)


if __name__ == '__main__':
    main()
