# Phase 0 数据可用性验收结果

## 结论

**STOP：当前不允许进入 Phase A1。**

本次验收使用真实工作区文件，完成了三份候选 session 的 CAG 容器、设计表、嵌入 VK4 头部、软件版本和输入 SHA-256 清单。样本映射已经根据用户确认闭合；当前唯一未闭合部分是 CAG 高度解码与 Keyence CSV 的数值等价证据。

## 已确认事实

| session | 设计行数 | CAG 测量数 | 嵌入 VK4 数 | 网格 |
|---|---:|---:|---:|---|
| zro2_120_formal | 120 | 120 | 120 | 2048×1536，0.344174 μm/px |
| zro2_60_pass | 60 | 30 | 30 | 2048×1536，0.344174 μm/px |
| zro2_20_supplement | 20 | 10 | 10 | 2048×1536，0.344174 μm/px |

三组 VK4 高度头均声明 32-bit 高度阵列，数组字节数与宽×高×4 一致，Z 量化步长为 0.0001 μm。

这些事实只证明容器和高度头结构可读，**不证明解码后的高度值、无效 mask 和坐标与 Keyence CSV 等价**。

## 阻断项

1. 工作区中没有同系列真实 `*_高度.csv`。
2. 因缺少 CSV fixture，尚未建立 CAG–CSV 的高度、像素间距和有效 mask 数值等价证据。

## 已闭合的样本映射

- `zro2_120_formal`：`sample_id == measurement_id`，每张测量图一个矩形，共120组。
- `zro2_60_pass`：measurement `m` 左侧矩形为 `sample_id=2m-1`，右侧为 `sample_id=2m`。
- `zro2_20_supplement`：采用相同的左奇数、右偶数规则。

映射 provenance 记录为 `user_confirmed_2026-08-30`。CAG 的 MeasurementDataMap 不含“1 2”这类显示标签，因此报告不会错误声称左右顺序来自容器元数据。

## 解除门禁所需材料

至少提供：

- 每个 session 各3个覆盖浅、中、深形貌的 Keyence 高度 CSV；
- 每个 CSV 对应的 CAG `measurement_id`；
- 若发生过重新装夹或采集方向变化，提供更细的 session 划分。

材料补齐后重新运行：

```powershell
.\.venv\Scripts\python.exe scripts\00_validate_inputs.py
```

只有 `phase0_validation.json` 中 `decision` 变为 `PASS`，才允许开始 Phase A1。
