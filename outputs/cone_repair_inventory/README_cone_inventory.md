# 圆锥伪影修复盘点（15 个 pilot 组）

## 方法口径
与 `extract_one` 内部一致：先 `locate_cut_corridor` 得到切割走廊，再把它作为
`allowed_mask` 传给 `repair_conical_dropouts`。Config 取默认（cone_repair_enabled=True,
half_window_px=12, seed_sigma=6.0,
grow_sigma=1.5, min_seed_depth_um=0.8,
max_component_span_px=36）。

## 总体
- 修复组数：15（status=ok: 15）
- 修复锥总数：290，修复像素总数：48256
- 每组建模：mean N_cones = 19.3，max = 40，min = 3
- 最大单点修正（全局）：16.618 um
- 残余强缺陷（走廊内未被修复的向下尖刺，疑似漏检或合理拒绝真实陡壁）：0 处

## 第一遍质检（需人工目视确认）
- 贴壁锥（centroid 落在 Y 边界 2px 内，可能是真实陡壁而非测量锥）：0 个
- 大修正锥（max_correction > 2.0 um，需重点核对是否误修真实几何）：210 个
- 注意：修复只向上修正（np.maximum），不会把真实沟槽往下填；列入"需核对"仅表示可疑，不等于错误。

## 文件
- cone_repair_group_summary.csv：每组统计
- cone_repair_artifact_table.csv：每个锥的尺寸/位置/修正
- cone_repair_inventory.png：汇总图
