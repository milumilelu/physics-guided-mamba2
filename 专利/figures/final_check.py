#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""终检：字号档位 + 裸数字附图标记清点"""
import re, os, io, sys
import xml.etree.ElementTree as ET

NS = '{http://www.w3.org/2000/svg}'
OK = {18, 20, 22, 24, 26, 28, 30, 32}

def check(path):
    root = ET.parse(path).getroot()
    bad, nums = [], []
    for el in root.iter():
        if el.tag.replace(NS, '') not in ('text', 'tspan'):
            continue
        raw = ''.join(el.itertext()).strip()
        try:
            fs = int(float(el.get('font-size', '22')))
        except (TypeError, ValueError):
            continue
        if not raw:
            continue
        if fs not in OK:
            bad.append((raw[:18], fs))
        if el.tag.replace(NS, '') == 'text' and raw.isdigit() and len(raw) <= 2:
            nums.append(raw)
    return bad, nums

if __name__ == '__main__':
    print('===== 字号违规（非档位/过小）=====')
    for i in range(1, 7):
        f = f'fig{i}.svg'
        if not os.path.exists(f):
            continue
        bad, nums = check(f)
        print(f'  {f}: 违规 {len(bad)} 处', bad if bad else '')
    print('===== 裸数字标记清点 =====')
    for i in range(1, 7):
        f = f'fig{i}.svg'
        if not os.path.exists(f):
            continue
        bad, nums = check(f)
        print(f'  {f}: {len(nums)} 处 -> {nums}')
