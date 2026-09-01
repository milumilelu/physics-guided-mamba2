# -*- coding: utf-8 -*-
"""专利附图自检：字号违规 / 文字宽度估算 / 箭头几何 / 硬编码色号统计"""
import re, glob, os, collections

CJK = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')

def est_width(text, size):
    w = 0.0
    for ch in text:
        if CJK.match(ch):
            w += size * 1.0
        elif ch == ' ':
            w += size * 0.28
        else:
            w += size * 0.52 if ch.islower() else size * 0.58
    return w

def text_content(frag):
    """抽取 <text> 内可见文字（含 tspan，忽略 dy 位移）"""
    frag = re.sub(r'<tspan[^>]*>', '', frag)
    frag = frag.replace('</tspan>', '')
    frag = re.sub(r'<[^>]+>', '', frag)
    return frag

def main():
    for f in sorted(glob.glob('fig*.svg')):
        src = open(f, encoding='utf-8').read()
        print('=' * 70)
        print(f)
        print('=' * 70)

        # ---- 1. 字号分布 ----
        sizes = collections.Counter()
        for m in re.finditer(r'font-size="([\d.]+)"', src):
            sizes[float(m.group(1))] += 1
        small = {s: c for s, c in sizes.items() if s < 22}
        print('[字号] 分布:', dict(sorted(sizes.items())))
        if small:
            bad = {s: c for s, c in small.items() if s < 18}
            print('  ! 小于22(辅助说明下限):', small)
            if bad:
                print('  !! 小于18(数学下标下限) — 违规:', bad)

        # ---- 2. 文字宽度估算（text-anchor=middle 的居中文本）----
        print('[宽度] 居中文本估算 (半宽 vs 可用半宽):')
        for m in re.finditer(r'<text ([^>]*)>(.*?)</text>', src, re.S):
            attrs, body = m.group(1), m.group(2)
            if 'text-anchor="middle"' not in attrs:
                continue
            a = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', attrs))
            x = float(a.get('x', 0)); fs = float(a.get('font-size', 26))
            txt = text_content(body)
            w = est_width(txt, fs)
            half = w / 2
            flag = ''
            if x - half < 8 or x + half > 1392:
                flag = '   <<< 超出画布'
            print('   x=%-6.0f fs=%-4.0f 半宽=%-6.0f [%.0f,%.0f] %r%s'
                  % (x, fs, half, x - half, x + half, txt[:34], flag))

        # ---- 3. 硬编码色号 ----
        cols = collections.Counter(re.findall(r'#[0-9A-Fa-f]{6}', src))
        print('[配色] 出现色号:', dict(cols))

        # ---- 4. 箭头 polygon 与线段端点一致性粗查 ----
        polys = re.findall(r'<polygon points="([^"]+)" fill="(#[0-9A-Fa-f]{6})"/>', src)
        print('[箭头] polygon 数量:', len(polys))
        for pts, fill in polys:
            nums = [float(v) for v in re.split(r'[ ,]+', pts.strip())]
            if len(nums) != 6:
                print('   ! 非三角 polygon:', pts)
                continue
            xs, ys = nums[0::2], nums[1::2]
            # 三条边长度，形状过于扁平/畸形则报警
            import math
            def d(i, j): return math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            sides = sorted([d(0, 1), d(1, 2), d(2, 0)])
            if sides[0] < 3 or sides[2] > 40 or sides[2] / max(sides[0], 0.01) > 6:
                print('   ! 可疑箭头 pts=%s 三边=%.1f/%.1f/%.1f' % (pts, *sides))

        # ---- 5. marker 残留 ----
        if 'marker-end' in src:
            print('[箭头] ! 仍存在 marker-end 引用:', src.count('marker-end'))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
