#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专利附图 碰撞/溢出 全量核查
用法: python collide_figs.py [figN.svg ...]
检查项:
  1. 文字 bbox 超出画布 (1400x820)
  2. 文字 bbox 超出其所在最小容器 rect (留 6px 安全边)
  3. 图形元素 (line/path/circle) 穿过文字 bbox  -> 压字
  4. 附图标记数字 (纯数字 text) 距最近容器边界 < 8px -> 贴线
"""
import re, sys, os, io
import xml.etree.ElementTree as ET

NS = '{http://www.w3.org/2000/svg}'
CJK = re.compile(r'[\u2e80-\u9fff\uff00-\uffef]')

def est_width(t, size):
    w = 0.0
    for ch in t:
        if CJK.match(ch):
            w += size * 1.0
        elif ch == ' ':
            w += size * 0.28
        else:
            w += size * (0.52 if ch.islower() else 0.58)
    return w

def txt_of(el):
    return ''.join(el.itertext())

def text_width(el, fs):
    """按 tspan 各自 font-size 累加，避免把下标按主字号估算导致宽度虚高"""
    parts = [((el.text or ''), fs)]
    for ch in el:
        tag = ch.tag.replace(NS, '')
        if tag != 'tspan':
            continue
        try:
            cfs = float(ch.get('font-size', fs))
        except (TypeError, ValueError):
            cfs = fs
        parts.append((ch.text or '', cfs))
        if ch.tail:
            parts.append((ch.tail, fs))
    return sum(est_width(t, s) for t, s in parts)


def parse(path):
    tree = ET.parse(path)
    root = tree.getroot()
    texts, rects, segs, circles = [], [], [], []
    for el in root.iter():
        tag = el.tag.replace(NS, '')
        if tag == 'text':
            try:
                x = float(el.get('x')); y = float(el.get('y'))
                fs = float(el.get('font-size', '22'))
            except (TypeError, ValueError):
                continue
            s = txt_of(el).strip()
            if not s:
                continue
            w = text_width(el, fs)
            a = el.get('text-anchor', 'start')
            if a == 'middle':
                x0 = x - w / 2
            elif a == 'end':
                x0 = x - w
            else:
                x0 = x
            texts.append(dict(s=s, fs=fs, x0=x0, x1=x0 + w,
                              y0=y - fs * 0.90, y1=y + fs * 0.26, cy=y - fs * 0.3))
        elif tag == 'rect':
            try:
                rects.append(dict(x=float(el.get('x', 0)), y=float(el.get('y', 0)),
                                  w=float(el.get('width', 0)), h=float(el.get('height', 0))))
            except (TypeError, ValueError):
                pass
        elif tag == 'line':
            try:
                segs.append((float(el.get('x1')), float(el.get('y1')),
                             float(el.get('x2')), float(el.get('y2'))))
            except (TypeError, ValueError):
                pass
        elif tag == 'circle':
            try:
                circles.append((float(el.get('cx')), float(el.get('cy')), float(el.get('r'))))
            except (TypeError, ValueError):
                pass
        elif tag == 'path':
            d = el.get('d', '')
            pts = re.findall(r'[ML]\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)', d)
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    segs.append((float(pts[i][0]), float(pts[i][1]),
                                 float(pts[i + 1][0]), float(pts[i + 1][1])))
    return texts, rects, segs, circles

def container(t, rects):
    """包含文字中心的最小 rect"""
    cx, cy = (t['x0'] + t['x1']) / 2, t['cy']
    best = None
    for r in rects:
        if r['w'] <= 0 or r['h'] <= 0:
            continue
        if r['x'] <= cx <= r['x'] + r['w'] and r['y'] <= cy <= r['y'] + r['h']:
            if best is None or r['w'] * r['h'] < best['w'] * best['h']:
                best = r
    return best

def seg_hits_box(s, t, shrink=1.5):
    x0, y0, x1, y1 = t['x0'] + shrink, t['y0'] + shrink, t['x1'] - shrink, t['y1'] - shrink
    if x1 <= x0 or y1 <= y0:
        return False
    ax, ay, bx, by = s
    # 采样判断线段是否进入文本框
    n = max(2, int(max(abs(bx - ax), abs(by - ay)) / 2))
    for i in range(n + 1):
        u = i / n
        px, py = ax + (bx - ax) * u, ay + (by - ay) * u
        if x0 <= px <= x1 and y0 <= py <= y1:
            return True
    return False

def circle_hits_box(c, t, shrink=1.5):
    cx, cy, r = c
    x0, y0, x1, y1 = t['x0'] + shrink, t['y0'] + shrink, t['x1'] - shrink, t['y1'] - shrink
    if x1 <= x0 or y1 <= y0:
        return False
    px = max(x0, min(cx, x1)); py = max(y0, min(cy, y1))
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r

def audit(path):
    texts, rects, segs, circles = parse(path)
    name = os.path.basename(path)
    out = []
    out.append('=' * 68)
    out.append(name)
    out.append('=' * 68)
    n_canvas = n_box = n_hit = n_num = 0
    for t in texts:
        s = t['s']
        if t['x0'] < 4 or t['x1'] > 1396 or t['y0'] < 2 or t['y1'] > 818:
            out.append(f"  [画布溢出] '{s[:20]}' x=[{t['x0']:.0f},{t['x1']:.0f}] y=[{t['y0']:.0f},{t['y1']:.0f}]")
            n_canvas += 1
        c = container(t, rects)
        if c:
            ov = []
            if t['x0'] < c['x'] + 6: ov.append(f"左溢{c['x'] + 6 - t['x0']:.0f}")
            if t['x1'] > c['x'] + c['w'] - 6: ov.append(f"右溢{t['x1'] - (c['x'] + c['w'] - 6):.0f}")
            if t['y0'] < c['y'] + 2: ov.append(f"顶溢{c['y'] + 2 - t['y0']:.0f}")
            if t['y1'] > c['y'] + c['h'] - 2: ov.append(f"底溢{t['y1'] - (c['y'] + c['h'] - 2):.0f}")
            if ov:
                out.append(f"  [容器溢出] '{s[:18]}' 框({c['x']:.0f},{c['y']:.0f},{c['w']:.0f}x{c['h']:.0f}) -> {','.join(ov)}")
                n_box += 1
        if s.isdigit() and len(s) <= 2 and c:
            gap = min(t['x0'] - c['x'], c['x'] + c['w'] - t['x1'],
                      t['y0'] - c['y'], c['y'] + c['h'] - t['y1'])
            if gap < 8:
                out.append(f"  [标记贴线] 标记 '{s}' 距框边 {gap:.0f}px (框 {c['x']:.0f},{c['y']:.0f},{c['w']:.0f}x{c['h']:.0f})")
                n_num += 1
        hits = sum(1 for sg in segs if seg_hits_box(sg, t))
        hits += sum(1 for cc in circles if circle_hits_box(cc, t))
        if hits:
            out.append(f"  [图形压字] '{s[:20]}' 被 {hits} 条线/圆穿过 (y=[{t['y0']:.0f},{t['y1']:.0f}])")
            n_hit += 1
    out.append(f"  >> 合计: 画布溢出 {n_canvas} / 容器溢出 {n_box} / 图形压字 {n_hit} / 标记贴线 {n_num}")
    return '\n'.join(out)

if __name__ == '__main__':
    files = sys.argv[1:] or [f'fig{i}.svg' for i in range(1, 7)]
    for f in files:
        if os.path.exists(f):
            print(audit(f))
