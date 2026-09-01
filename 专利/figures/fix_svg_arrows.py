# -*- coding: utf-8 -*-
"""
把 SVG 里依赖 <marker> + url(#id) 的箭头，替换为显式的实心三角形 <polygon>。

动机：
1. 原 marker 画的是 fill="none" 的细线 V 形，尺寸只有 ~6x6px，缩到预览尺寸几乎不可见；
2. url(#id) 引用在部分预览器 /  sanitizer / 老渲染器里会被丢弃，导致"图片一个箭头都没有"；
3. 专利附图规范本来就要求实心箭头。

用法：python fix_svg_arrows.py
会自动处理当前目录下所有 fig*.svg。已处理过的文件无 marker 引用，重复运行是 no-op。
"""
import re
import math
import glob
import os

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
NUM = re.compile(r'-?\d+(?:\.\d+)?')


def parse_attrs(tag):
    return dict(re.findall(r'([\w:-]+)="([^"]*)"', tag))


def pts_from_d(d):
    """按坐标对顺序解析 M/L/C/Q 路径。
    对三次贝塞尔，终点切线 = 终点 - 最后一个控制点；对二次贝塞尔同理。
    本项目的路径全部只含坐标对，因此该简化解析是精确的。"""
    v = [float(x) for x in NUM.findall(d)]
    return [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]


def pts_from_points(p):
    v = [float(x) for x in NUM.findall(p)]
    return [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]


def unit(vx, vy):
    n = math.hypot(vx, vy)
    return (vx / n, vy / n) if n else (0.0, 0.0)


def arrow_polygon(tip, direction, color, length, width):
    ux, uy = direction
    px, py = -uy, ux                      # 垂直方向
    bx, by = tip[0] - ux * length, tip[1] - uy * length   # 底边中点
    a = (tip[0], tip[1])
    b = (bx + px * width, by + py * width)
    c = (bx - px * width, by - py * width)
    return ('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="%s"/>'
            % (a[0], a[1], b[0], b[1], c[0], c[1], color))


def convert(src, tag_re, get_pts):
    out = []
    for m in tag_re.finditer(src):
        tag = m.group(0)
        a = parse_attrs(tag)
        has_end = 'marker-end' in a
        has_start = 'marker-start' in a
        if not (has_end or has_start):
            continue
        pts = get_pts(a)
        if len(pts) < 2:
            continue

        color = a.get('stroke', '#444441')
        if not color or color == 'none':
            color = '#444441'

        sw = float(a.get('stroke-width', 1.2) or 1.2)
        length = min(max(9.0 * sw, 11.0), 16.0)
        width = length * 0.40

        arrows = []
        if has_start:
            # marker-start + auto-start-reverse：指向与路径方向相反
            d = unit(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
            arrows.append(arrow_polygon(pts[0], d, color, length, width))
        if has_end:
            d = unit(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
            arrows.append(arrow_polygon(pts[-1], d, color, length, width))

        new_tag = tag
        for k in ('marker-end', 'marker-start', 'marker-mid'):
            new_tag = re.sub(r'\s*' + k + r'="[^"]*"', '', new_tag)

        out.append((m.start(), m.end(), new_tag + '\n' + '\n'.join(arrows)))

    res, last = [], 0
    for s, e, rep in out:
        res.append(src[last:s])
        res.append(rep)
        last = e
    res.append(src[last:])
    return ''.join(res), len(out)


def main():
    total = 0
    for path in sorted(glob.glob(os.path.join(FIG_DIR, 'fig*.svg'))):
        src = open(path, encoding='utf-8').read()
        n = 0

        src, k = convert(src, re.compile(r'<line\b[^>]*>'),
                         lambda a: [(float(a['x1']), float(a['y1'])),
                                    (float(a['x2']), float(a['y2']))])
        n += k
        src, k = convert(src, re.compile(r'<path\b[^>]*>'),
                         lambda a: pts_from_d(a.get('d', '')))
        n += k
        src, k = convert(src, re.compile(r'<polyline\b[^>]*>'),
                         lambda a: pts_from_points(a.get('points', '')))
        n += k

        # 移除已不再被引用、且含 fill="none" 细线箭头的 marker 定义
        src = re.sub(r'\s*<marker\b.*?</marker>', '', src, flags=re.S)
        src = re.sub(r'<defs>\s*</defs>\n?', '', src, flags=re.S)

        open(path, 'w', encoding='utf-8').write(src)
        total += n
        print('%-12s  转换箭头 %d 处  (%s)' % (os.path.basename(path), n, k and '' or ''))

    print('\n共转换 %d 处箭头为实心三角形。' % total)


if __name__ == '__main__':
    main()
