# -*- coding: utf-8 -*-
"""按《画图说明 2.0》"单色 / 黑白改绘"映射，将 6 张彩色附图转纯黑白线稿。"""
import os, re, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, 'fig1.svg')  # placeholder, real loop below
DST  = os.path.join(ROOT, '..', 'figures_bw')

os.makedirs(DST, exist_ok=True)

# 配色映射 (浅填色 -> 无, 边/线 -> 黑, 红虚框 -> 点划)
FILL_MAP = {
    '#FFFFFF': 'none',         # 白色
    '#F1EFE8': 'none',         # 浅灰
    '#E6F1FB': 'none',         # 浅蓝
    '#FAEEDA': 'none',         # 浅橙
    '#FBEAEA': 'none',         # 浅红
    '#F7FBFF': 'none',         # 浅蓝(泳道)
    '#FAFAF8': 'none',         # 浅灰(泳道)
}
STROKE_MAP = {
    '#185FA5': '#000000',      # 蓝 -> 黑
    '#0C447C': '#000000',      # 深蓝 -> 黑
    '#5F5E5A': '#000000',      # 灰 -> 黑
    '#444441': '#000000',      # 深灰 -> 黑
    '#BA7517': '#000000',      # 橙 -> 黑
    '#633806': '#000000',      # 深橙 -> 黑
    '#A32D2D': '#000000',      # 红 -> 黑
    '#DCE9F5': '#808080',      # 浅蓝边 -> 浅灰
    '#E6E4DE': '#808080',      # 浅灰边 -> 浅灰
    '#D4A574': '#808080',      # 淡橙线 -> 浅灰
    '#6B96C8': '#808080',      # 淡蓝线 -> 浅灰
}

def convert(src_path, dst_path):
    with open(src_path, encoding='utf-8') as f:
        s = f.read()
    # fill -> none
    for k, v in FILL_MAP.items():
        s = s.replace(f'fill="{k}"', f'fill="{v}"')
    # stroke / arrow fill 颜色
    for k, v in STROKE_MAP.items():
        s = s.replace(f'fill="{k}"', f'fill="{v}"')
        s = s.replace(f'stroke="{k}"', f'stroke="{v}"')
    # 写头注释
    head = '<!-- BLACK-AND-WHITE VERSION: derived by figures/fig_bw_convert.py; for CNIPA paper-filing -->\n'
    s = re.sub(r'<svg ', head + '<svg ', s, count=1)
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(s)

for n in range(1, 7):
    src = os.path.join(ROOT, f'fig{n}.svg')
    dst = os.path.join(DST, f'fig{n}.svg')
    convert(src, dst)
    print(f'wrote {dst}')
