#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WP3 前置步骤：冻结 CAG--CSV 等价验证的 fixture 选择表。

为什么必须单独一个脚本
----------------------
WP3 的等价验证要拿「解码器输出」去比「KEYENCE 官方导出」。如果先看了
比较误差再决定比哪些样本，就等于允许"挑好做的样本凑 PASS"。所以选择
规则必须先跑、先落盘、先哈希，然后才允许看见任何误差。

本脚本只做**选择**，不做任何比较，也从不读取目标 CSV 的数值内容。
它唯一读的 CSV 元数据是来源类型（官方 / 解码器），用来判断哪些样本
已经不需要再导出。

选择规则（已冻结，见 --rule-id）
--------------------------------
1. 已有官方导出的 measurement 直接入选，不消耗配额，也不需要用户操作。
2. 其余候选按**实测槽深** `P50 - P1` 分入 3 个深度带（秩三分位；并列
   整块同带，不拆开）。这是计划 §8.1 说的「不依赖单阈值的 contrast
   diagnostics」—— 三分位是秩统计量，没有阈值，且对整体缩放不敏感。
3. 每个深度带内取 `n_distinct_levels` 最大者 —— 取值层数越多，舍入与
   比例错误越无处可藏。这是计划要求的「≥3 且覆盖浅、中、深」保底集合。
4. 若目标数 > 3，用贪心补齐：每次选入能覆盖最多**未覆盖设计因子水平**的
   候选。按「水平」而不是按「组合」计数 —— 这些设计表是 DOE 表，每一行
   的 (次数, 脉宽, 频率, 线间距, 速度) 组合都唯一，按组合计数会让目标
   退化成"谁对比度高选谁"，参数覆盖形同虚设。
5. 报告方向可检测性，并作为贪心的次级排序键（越大越优先），但**不作为
   否决条件** —— 与 LUT 偏移量那条接缝启发式是同一条原则：没有鉴别力的
   检查可以不出声，但不允许它否决。

所有 morphology 指标只用于**排序**，不用于判定，因此不引入阈值依赖。

关于方向风险的两点澄清
----------------------
- **转置**在 2048×1536 上结构性不可能：转置后形状变成 1536×2048，
  计划 §8.3 的 "width / height 完全一致" 直接拦下，不需要形貌指标。
- **镜像翻转**才是真实风险，且它**可以**逃过检查：若加工区恰好位于
  视场中心，水平翻转后它几乎落回原位，四角值也相差无几。所以用
  `flip_norm = median|z - mirror(z)| / (P99-P1)` 量化，归一化后无阈值。

规则版本
--------
- v1：第 4 步按设计参数**组合**计数 → 在 DOE 表上退化为"选对比度最高的"。
- v2：改用因子**水平**计数；但深度带仍按设计参数「次数」划分，实测
  发现次数与槽深只是弱相关（次数=3 的样本比次数=5 的还深），带间没有
  分离；方向指标也算错了对象（算了转置而非翻转）。
- v3：深度带改用实测槽深；方向指标改用镜像残差。

v1、v2 均在**任何比较之前**废弃，从未产生过一次 CAG–CSV 对比。冻结的
约束是"不得在看到误差后换样本"，不是"不得在看到形貌诊断后修规则"。

输出
----
  config/equivalence_fixture_selection.csv          冻结的选择表
  outputs/.../phase0/equivalence/fixture_selection.csv
  outputs/.../phase0/equivalence/fixture_candidates.csv   全体候选与得分（审计用）
  outputs/.../phase0/equivalence/fixture_selection.json    含选择表 sha256
  outputs/.../phase0/equivalence/FIXTURE_EXPORT_REQUEST.md 给用户的操作清单
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402

RULE_ID = "fixture-selection-v4-user-fixture-set-before-comparison"
DESIGN_FACTORS = ("次数", "脉宽", "频率", "线间距", "速度")

FIELDNAMES = [
    "session_id",
    "measurement_id",
    "cag_data_name",
    "expected_csv_filename",
    "expected_csv_relative_path",
    "slot_1_sample_id",
    "slot_2_sample_id",
    "design_factor_tuple",
    "depth_proxy_max_pass",
    "depth_band",
    "selection_reason",
    "existing_official_csv",
    "action_required",
    "depth_proxy_um",
    "robust_range_um",
    "iqr_um",
    "n_distinct_levels",
    "valid_fraction",
    "width",
    "height",
    "flip_norm_x",
    "flip_norm_y",
    "direction_min_flip_norm",
    "transposition_structurally_impossible",
]


# --------------------------------------------------------------------------- #
# 输入
# --------------------------------------------------------------------------- #
def _read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict]:
    with path.open("r", encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def load_design(path: Path) -> dict[int, dict]:
    """设计表按「加工顺序」索引。这些文件是 GBK 编码，不是 UTF-8。"""
    out: dict[int, dict] = {}
    for row in _read_csv(path, encoding="gbk"):
        sid = int(float(row["加工顺序"]))
        out[sid] = {k: row.get(k, "") for k in DESIGN_FACTORS}
    return out


def load_design_safely(path: Path) -> dict[int, dict]:
    """GBK 优先，失败回退 UTF-8-BOM。有些表被 Excel 另存过。"""
    try:
        return load_design(path)
    except UnicodeDecodeError:
        return {int(float(r["加工顺序"])): dict(r) for r in _read_csv(path)}


# --------------------------------------------------------------------------- #
# 形貌诊断（来自 CAG，与任何目标 CSV 无关）
# --------------------------------------------------------------------------- #
def contrast_diagnostics(hm) -> dict:
    """计算不依赖阈值的形貌对比度与方向可检测性指标。"""
    z = hm.z
    mask = hm.valid_mask
    valid = z[mask]
    height, width = z.shape

    if valid.size == 0:
        return {
            "depth_proxy_um": float("nan"),
            "robust_range_um": float("nan"),
            "iqr_um": float("nan"),
            "n_distinct_levels": 0,
            "valid_fraction": 0.0,
            "width": width,
            "height": height,
            "flip_norm_x": float("nan"),
            "flip_norm_y": float("nan"),
            "direction_min_flip_norm": float("nan"),
            "transposition_structurally_impossible": bool(width != height),
        }

    p01, p25, p50, p75, p99 = np.percentile(valid, [1, 25, 50, 75, 99])
    robust_range = float(p99 - p01)

    # 有效像素的取值层数：越多越能暴露舍入/比例错误。
    milli = np.rint(valid * 1000.0).astype(np.int64)
    n_levels = int(np.unique(milli).size)

    # 镜像翻转残差，用动态范围归一化。值越小说明该视场越接近镜像对称，
    # 翻转错误越难被发现 —— 加工区居中的视场会落在这里。
    zz = np.where(mask, z, np.nan)
    both_x = mask & mask[:, ::-1]
    both_y = mask & mask[::-1, :]
    scale = robust_range if robust_range > 0 else 1.0
    d_x = np.abs(zz - zz[:, ::-1])
    d_y = np.abs(zz - zz[::-1, :])
    flip_x = float(np.nanmedian(d_x[both_x]) / scale) if both_x.any() else float("nan")
    flip_y = float(np.nanmedian(d_y[both_y]) / scale) if both_y.any() else float("nan")

    return {
        # 槽深：未加工面（中位数）到槽底（P1）的距离。加工区通常只占
        # 视场的几个百分点，所以 P1 已经足够接近槽底，而 P0.1 太噪。
        # 这是秩统计量，对整体缩放不敏感，用于分带而非判定。
        "depth_proxy_um": float(p50 - p01),
        "robust_range_um": robust_range,
        "iqr_um": float(p75 - p25),
        "p50_um": float(p50),
        "n_distinct_levels": n_levels,
        "valid_fraction": float(valid.size / z.size),
        "width": width,
        "height": height,
        "flip_norm_x": flip_x,
        "flip_norm_y": flip_y,
        "direction_min_flip_norm": float(np.nanmin([flip_x, flip_y])),
        "transposition_structurally_impossible": bool(width != height),
    }


# --------------------------------------------------------------------------- #
# 选择
# --------------------------------------------------------------------------- #
def rank_tertile(ordered_proxies: list[float]) -> list[int]:
    """把已按 (depth_proxy, measurement_id) 升序排列的序列切成三个秩带。

    并列（同一个 depth_proxy）**不拆开**：整块落在同一个带里，否则
    "浅 / 中 / 深"的语义会被打散。整块的位置取块内下标的均值。
    """
    n = len(ordered_proxies)
    if n == 0:
        return []
    cut1, cut2 = n / 3.0, 2.0 * n / 3.0

    bands = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered_proxies[j + 1] == ordered_proxies[i]:
            j += 1
        mid_pos = (i + j) / 2.0
        b = 0 if mid_pos < cut1 else (1 if mid_pos < cut2 else 2)
        for k in range(i, j + 1):
            bands[k] = b
        i = j + 1
    return bands


def select_session(session_id: str, candidates: list[dict], target: int,
                   minimum: int) -> list[dict]:
    """在一个 session 内执行冻结的选择规则。"""
    picked: list[dict] = []
    picked_ids: set = set()

    # 1) 已有官方导出 → 直接入选
    for c in candidates:
        if c["existing_official_csv"]:
            c["selection_reason"] = "existing_official_export"
            c["action_required"] = "none"
            picked.append(c)
            picked_ids.add(c["measurement_id"])

    remaining = [c for c in candidates if c["measurement_id"] not in picked_ids]
    # Once the hard minimum is met, freeze every existing independent export
    # and do not request extra samples merely to reach the aspirational target.
    # This is especially important when a user exported a different, still
    # pre-comparison set from an earlier request table.
    if len(picked) >= minimum:
        return picked
    if not remaining:
        return picked

    # 2) 按 depth_proxy 分三个深度带
    ordered = sorted(remaining, key=lambda c: (c["depth_proxy_um"], c["measurement_id"]))
    for c, b in zip(ordered, rank_tertile([c["depth_proxy_um"] for c in ordered])):
        c["depth_band"] = b

    def band_key(c):
        # 带内取信息量最大者：取值层数越多，舍入/比例错误越难藏。
        return (-c["n_distinct_levels"], c["measurement_id"])

    # 3) 每带取信息量第一名 —— 保底 3 个（浅 / 中 / 深）
    for b in (0, 1, 2):
        pool = sorted([c for c in remaining if c["depth_band"] == b], key=band_key)
        if not pool:
            continue
        chosen = pool[0]
        chosen["selection_reason"] = f"depth_band_{b}_max_levels"
        chosen["action_required"] = "export_from_keyence"
        picked.append(chosen)
        picked_ids.add(chosen["measurement_id"])

    # 4) 贪心补齐设计因子**水平**覆盖
    covered: set[tuple[int, str]] = set()
    for c in picked:
        covered |= c["_levels"]

    while len(picked) < target:
        pool = [c for c in remaining if c["measurement_id"] not in picked_ids]
        if not pool:
            break
        best, best_key = None, None
        for c in pool:
            gain = len(c["_levels"] - covered)
            flip = c["direction_min_flip_norm"]
            flip = flip if np.isfinite(flip) else -1.0
            key = (-gain, -flip, -c["n_distinct_levels"], c["measurement_id"])
            if best_key is None or key < best_key:
                best, best_key = c, key
        best["selection_reason"] = "design_level_coverage_greedy"
        best["action_required"] = "export_from_keyence"
        covered |= best["_levels"]
        picked.append(best)
        picked_ids.add(best["measurement_id"])

    return picked


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="冻结 WP3 等价验证的 fixture 选择表")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--output-dir", default="outputs/rectangle_registration")
    ap.add_argument("--target-per-session", type=int, default=6,
                    help="每个 session 期望的 fixture 总数（含已有官方导出）")
    ap.add_argument("--min-per-session", type=int, default=3,
                    help="计划 §8.1 的硬下限")
    ap.add_argument("--sessions", default="",
                    help="逗号分隔，只处理指定 session（默认全部）")
    ap.add_argument("--skip-decode", action="store_true",
                    help="跳过 CAG 解码诊断（仅重建清单，不用于正式冻结）")
    args = ap.parse_args()

    root = Path.cwd()
    cfg = root / args.config_dir
    out_root = root / args.output_dir
    eq_dir = out_root / "phase0" / "equivalence"
    eq_dir.mkdir(parents=True, exist_ok=True)

    sessions = _read_csv(cfg / "session_manifest.csv")
    sources = _read_csv(cfg / "height_source_manifest.csv")

    if args.sessions:
        wanted = set(args.sessions.split(","))
        sessions = [s for s in sessions if s["session_id"] in wanted]

    official_by_session: dict[str, set[int]] = {}
    for r in sources:
        if r["csv_source_type"] == "keyence_official_export":
            official_by_session.setdefault(r["session_id"], set()).add(int(r["measurement_id"]))

    all_rows: list[dict] = []
    picked_rows: list[dict] = []
    report: dict = {
        "rule_id": RULE_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target_per_session": args.target_per_session,
        "min_per_session": args.min_per_session,
        "sessions": {},
    }

    for ses in sessions:
        sid = ses["session_id"]
        cag_path = root / ses["cag_path"]
        design = load_design_safely(root / ses["design_path"])
        csv_subdir = ses["csv_subdir"]
        official = official_by_session.get(sid, set())

        # measurement -> 它覆盖的 sample_id。配对关系只从来源清单读，
        # 绝不假设 left=2m-1 / right=2m（60Pass组.cag 是蛇形扫描，顺序是反的）。
        meas_samples: dict[int, list[int]] = {}
        for r in sources:
            if r["session_id"] != sid:
                continue
            mid = int(r["measurement_id"])
            slots = []
            for key in ("slot_1_sample_id", "slot_2_sample_id"):
                if r.get(key):
                    slots.append(int(r[key]))
            meas_samples[mid] = slots or [mid]

        candidates: list[dict] = []
        reader = None
        if not args.skip_decode:
            reader = CagHeightReader(cag_path, verify_lut=False)

        for r in sources:
            if r["session_id"] != sid:
                continue
            mid = int(r["measurement_id"])
            samples = meas_samples.get(mid) or [mid]
            drows = [design.get(s, {}) for s in samples]
            drows = [d for d in drows if d]
            passes = [float(d.get("次数", "nan")) for d in drows]
            passes = [p for p in passes if np.isfinite(p)]
            depth_proxy = max(passes) if passes else float("nan")

            factors = []
            for d in drows:
                factors.append(tuple(d.get(k, "") for k in DESIGN_FACTORS))
            ftuple = factors[0] if factors else tuple([""] * len(DESIGN_FACTORS))

            # 该 measurement 覆盖的「因子水平」集合。paired measurement 的
            # 两个样本参数不同，两边都算进来。
            levels: set[tuple[int, str]] = set()
            for d in drows:
                for idx, k in enumerate(DESIGN_FACTORS):
                    v = d.get(k, "")
                    if v != "":
                        levels.add((idx, v))

            data_name = r["cag_data_name"]
            fname = f"{data_name}_高度.csv"
            rec = {
                "session_id": sid,
                "measurement_id": mid,
                "cag_data_name": data_name,
                "expected_csv_filename": fname,
                "expected_csv_relative_path": f"氧化锆/pass实验数据/csv文件/{csv_subdir}/{fname}",
                "slot_1_sample_id": r["slot_1_sample_id"],
                "slot_2_sample_id": r["slot_2_sample_id"],
                "design_factor_tuple": ftuple,
                "_levels": levels,
                "depth_proxy_max_pass": depth_proxy,
                "depth_band": -1,
                "selection_reason": "not_selected",
                "existing_official_csv": mid in official,
                "action_required": "none" if mid in official else "export_from_keyence",
            }
            if reader is not None:
                hm = reader.read_height_map(mid)
                rec.update(contrast_diagnostics(hm))
            else:
                rec.update({k: float("nan") for k in FIELDNAMES if k not in rec})
                rec["n_distinct_levels"] = 0
                rec["direction_min_flip_norm"] = float("nan")
            candidates.append(rec)

        if reader is not None:
            reader.close()

        picked = select_session(sid, candidates, args.target_per_session,
                                args.min_per_session)
        all_rows.extend(candidates)
        picked_rows.extend(picked)

        n_official = len(official)
        n_new = sum(1 for c in picked if c["action_required"] == "export_from_keyence")

        # 覆盖度自检：规则 4 说要覆盖因子水平，就得真的量出来，
        # 否则"覆盖"两个字又是空话。
        all_levels: set[tuple[int, str]] = set()
        for c in candidates:
            all_levels |= c["_levels"]
        sel_levels: set[tuple[int, str]] = set()
        for c in picked:
            sel_levels |= c["_levels"]
        per_factor = {}
        for idx, name in enumerate(DESIGN_FACTORS):
            tot = {v for (i, v) in all_levels if i == idx}
            got = {v for (i, v) in sel_levels if i == idx}
            per_factor[name] = {"levels_total": len(tot), "levels_covered": len(got)}
        # 带间分离度自检：如果浅/中/深三带的槽深中位数没有拉开，
        # 那么"覆盖浅中深"这句话就是空的，必须报出来。
        band_ranges = {}
        for b in (0, 1, 2):
            vals = [c["depth_proxy_um"] for c in picked
                    if c.get("depth_band") == b and np.isfinite(c["depth_proxy_um"])]
            band_ranges[f"band_{b}"] = float(np.median(vals)) if vals else None
        # 不输出布尔结论 —— "三个深度带是否算分开"没有天然判据。
        # 曾经试过 1.5 倍相邻比，结果 120正式 的 band_1/band_2 = 1.46 被判
        # False，而 13.7 / 41.9 / 61.3 µm 显然是三个不同量级。给一个编造的
        # 阈值再反过来调它，就是把结论当成了目标。所以只报比值，让人判断。
        b0, b1, b2 = (band_ranges.get(f"band_{i}") for i in (0, 1, 2))
        present = [v for v in (b0, b1, b2) if v is not None]
        band_ratios = {
            "adjacent": [round(b1 / b0, 3) if b0 else None,
                         round(b2 / b1, 3) if b1 else None],
            "span_max_over_min": round(max(present) / min(present), 3) if len(present) >= 2 else None,
        }

        flip_vals = [c["direction_min_flip_norm"] for c in picked
                     if np.isfinite(c["direction_min_flip_norm"])]
        min_flip = float(min(flip_vals)) if flip_vals else None
        report["sessions"][sid] = {
            "n_measurements": len(candidates),
            "n_existing_official": n_official,
            "n_selected": len(picked),
            "n_new_exports_needed": n_new,
            "n_official_after_export": n_official + n_new,
            "meets_minimum": bool(n_official + n_new >= args.min_per_session),
            "factor_level_coverage": per_factor,
            "median_depth_proxy_um_by_depth_band": band_ranges,
            "depth_band_ratios": band_ratios,
            "min_direction_flip_norm_over_selected": min_flip,
            "transposition_structurally_impossible_all": bool(
                all(c["transposition_structurally_impossible"] for c in picked)),
            "blocked_reason": None if n_new == 0 else f"need {n_new} manual KEYENCE exports",
            "selected_measurement_ids": [c["measurement_id"] for c in picked],
            "existing_official_measurement_ids": sorted(official),
            "csv_subdir": csv_subdir,
            "cag_path": ses["cag_path"],
        }

    # ---- 落盘 ------------------------------------------------------------ #
    def dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=lambda x: (x["session_id"], x["measurement_id"])):
                w.writerow({k: r.get(k, "") for k in FIELDNAMES})

    frozen = cfg / "equivalence_fixture_selection.csv"
    dump(frozen, picked_rows)
    dump(eq_dir / "fixture_selection.csv", picked_rows)
    dump(eq_dir / "fixture_candidates.csv", all_rows)

    sel_sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    report["selection_table_path"] = str(frozen.relative_to(root))
    report["selection_table_sha256"] = sel_sha
    (eq_dir / "fixture_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    write_request_md(eq_dir / "FIXTURE_EXPORT_REQUEST.md", report, picked_rows, args)

    # ---- 控制台 ---------------------------------------------------------- #
    print(f"rule = {RULE_ID}   target/session = {args.target_per_session}")
    print(f"frozen table sha256 = {sel_sha[:16]}...")
    print()
    for sid, info in report["sessions"].items():
        flag = "OK  " if info["blocked_reason"] is None else "STOP"
        print(f"[{flag}] {sid}: 官方已有 {info['n_existing_official']}，"
              f"需新导出 {info['n_new_exports_needed']}，共 {info['n_selected']}")
    print()
    need = [(sid, i) for sid, i in report["sessions"].items() if i["n_new_exports_needed"] > 0]
    if need:
        print("=== 需要你手工导出的清单 ===")
        for sid, info in need:
            print(f"\n--- {sid}  →  目录 氧化锆/pass实验数据/csv文件/{info['csv_subdir']}/")
            for r in picked_rows:
                if r["session_id"] == sid and r["action_required"] == "export_from_keyence":
                    print(f"    data name  {r['cag_data_name']:<10} "
                          f"→ {r['expected_csv_filename']:<18} "
                          f"(样本 {r['slot_1_sample_id']}"
                          f"{',' + r['slot_2_sample_id'] if r['slot_2_sample_id'] else ''}"
                          f", 次数≤{r['depth_proxy_max_pass']})")
    return 0


def write_request_md(path: Path, report: dict, picked: list[dict], args) -> None:
    lines = [
        "# WP3 等价验证：需要手工补导出的官方 CSV",
        "",
        f"- 生成时间（UTC）：{report['generated_utc']}",
        f"- 选择规则 ID：`{report['rule_id']}`（**已冻结**，见下方 sha256）",
        f"- 冻结选择表：`{report['selection_table_path']}`",
        f"- 选择表 sha256：`{report['selection_table_sha256']}`",
        "",
        "这张表在**看到任何比较误差之前**生成并哈希。若之后需要更换样本，",
        "必须新建规则 ID 并重新冻结，不得原地改表。",
        "",
        "## 为什么需要手工导出",
        "",
        "我们自己的解码器从 `.cag` 读高度。要证明它读对了，必须拿 KEYENCE",
        "软件自己导出的 CSV 当独立真值。现在 120正式 的 120 个、60pass 的 29 个",
        "CSV 全是脚本生成的 —— 拿解码器的输出比解码器的输出，只能证明自洽，",
        "不能证明正确。",
        "",
        "## 操作步骤（非常重要）",
        "",
        "1. 在 KEYENCE VK 分析软件中打开对应的 `.cag`；",
        "2. 对下表指定的 measurement 执行**与当年做 20补充pass 时完全相同**的",
        "   导出操作（ImageDataCsv / 高度数据 CSV）；",
        "3. 存到下表指定目录，**用软件自动生成的名字，不要重命名**；",
        "4. **不要覆盖**同一目录下已有的官方文件；",
        "5. 导出完成后**不要再做任何整理、压缩、拷贝**，直接告诉我，我重新跑",
        "   provenance 判定后再比较。",
        "",
        "> 如果软件支持批量导出整个 series，建议直接全部导出 —— 3 个只是",
        "> 计划规定的下限，证据越多越好，成本几乎一样。",
        "",
    ]
    for sid, info in report["sessions"].items():
        rows = [r for r in picked if r["session_id"] == sid]
        new = [r for r in rows if r["action_required"] == "export_from_keyence"]
        lines += [
            f"## {sid}",
            "",
            f"- CAG：`{info['cag_path']}`",
            f"- 输出目录：`氧化锆/pass实验数据/csv文件/{info['csv_subdir']}/`",
            f"- 已有官方导出：{info['n_existing_official']} 个"
            f"（measurement {info['existing_official_measurement_ids']}）—— 无需操作",
            f"- 本次需新导出：**{len(new)} 个**",
            "",
        ]
        if not new:
            lines += ["该 session 无需操作。", ""]
            continue
        lines += [
            "| measurement | CAG data name | 导出后文件名 | 覆盖样本 | 深度带 |",
            "|---:|---|---|---|---|",
        ]
        for r in new:
            samples = r["slot_1_sample_id"] + ("," + r["slot_2_sample_id"] if r["slot_2_sample_id"] else "")
            band = {0: "浅", 1: "中", 2: "深"}.get(r.get("depth_band"), "-")
            lines.append(
                f"| {r['measurement_id']} | `{r['cag_data_name']}` | "
                f"`{r['expected_csv_filename']}` | {samples} | {band} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
