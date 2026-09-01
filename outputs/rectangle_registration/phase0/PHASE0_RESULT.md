# Phase 0 数据可用性验收结果

## 结论

**PASS**

- session：3
- measurement：160
- sample mapping：200
- CAG–CSV 等价证据：PASS
- blocker：0

## 阻断项

- 无

## 规则

只有 `phase0_validation.json` 的 `decision` 为 `PASS` 且 blockers 为空，才允许进入 Phase A。
本报告由 `scripts/00_validate_inputs.py` 自动生成，不应手工修改。
