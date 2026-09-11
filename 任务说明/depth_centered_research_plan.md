# 以加工深度为目标的工艺—表面形貌信息压缩研究

## 1. 结论与研究定位

**这条路线可行，而且比要求模型同时复现全部表面形貌更聚焦。最值得研究的问题，是表面形貌中的哪些信息与加工深度有关、这些信息是否吸收了工艺参数对深度的预测贡献，以及深度能否作为统一部分形貌演化的加工进度坐标。**

建议保留“工艺参数 → 表面形貌 → 深度”的总体方向，但不要在实验之前把它规定为没有旁路的因果链。第一阶段的正式表述应为“深度导向的形貌表征与工艺信息压缩”，允许工艺参数直接参与深度预测。待留出检验证明这些直接输入基本冗余，再收缩为严格的形貌中间层。

这里有两个不同、但互补的问题。第一个是“从形貌看深度”：去除绝对高度后，形貌能否预测深度，并提供工艺参数之外的信息？第二个是“用深度解释形貌”：在同一深度下，不同工艺形成的哪些形貌差异消失，哪些仍然保留？前者是新主任务，后者是理解“加工中吸收了什么信息”的重要诊断。

研究的最终目标仍是标量深度 D。可选的小维度中间表示 Z_D 只是筛选深度相关形貌的工具，不是把研究目标改回多目标形貌预测，也不是要求首先实现深度神经网络。

本报告以仓库提交 `f80bca0fa604197bc217ca725fd573431483d04b` 中实际读取的观察算子、谱组成实现、历史可预测性表和 MAIN180/E02 逐目标结果为依据。下文的既有数值是仓库记录，不是重新训练所得；新增路线和验收条件均是建议方案。E02 首批报告不能代表所有后续任务的最新完成状态。[1–7]

## 2. 当前结果为什么支持这一转向

### 2.1 深度本身已有预测基础，但方向性强不意味着对深度重要

Phase 2.8r1 的实际 CSV 记录如下。表中保留原文件的 skill 口径，来源分组为 src_gkf，工艺分组为 proc_gkf。[1]

| 目标 | 完整工艺，来源分组 | 仅 hatch，来源分组 | 去掉 hatch，来源分组 | 完整工艺，工艺分组 |
|---|---:|---:|---:|---:|
| 深度 D | 0.552 | 0.044 | 0.501 | 0.577 |
| 幅值 A | 0.160 | 0.001 | 0.142 | 0.157 |
| 谱组成 P | 0.308 | 0.145 | 0.145 | 0.339 |
| 方向性 A2 | 0.641 | 0.615 | 0.005 | 0.552 |
| 方向熵 | 0.662 | 0.582 | 0.015 | 0.645 |

它支持三点判断。首先，工艺直接预测深度已经有可比较的基线。其次，方向统计主要响应 hatch，但深度并不主要由 hatch 单变量预测。最后，不能因为方向形貌最容易被工艺预测，就将其设为深度预测的唯一中间表示。

这也不意味着 hatch 对深度无效。原表中深度的折内配对 hatch 增量中位数约为 0.090；该统计量不是“两个中位数直接相减”。是否需要保留 hatch，要由深度任务的条件增益检验决定。[1]

### 2.2 最相关的新证据：深度与细—粗尺度对比有关，而与方向性关系较弱

现有 `02_depth_compression.py` 对应配置的选择目标是 A/P/T 三块风险，因此它回答的是“D、D+h 或 U 能否解释形貌”，不是“形貌能否预测 D”。从其 `native_metrics.csv` 逐目标读取，可见以下结果。[2–4]

| MAIN180 目标 | GAM：仅实测 D | GAM：D+h | GAM：完整 U | Ridge：仅实测 D | Ridge：完整 U |
|---|---:|---:|---:|---:|---:|
| 幅值 A_med | 0.168 | 0.170 | −0.006 | 0.137 | 0.131 |
| 谱对比 ilr_z1 | 0.332 | 0.350 | 0.345 | 0.342 | 0.416 |
| 谱对比 ilr_z2 | −0.061 | 0.353 | 0.256 | −0.031 | 0.159 |
| 方向性 A2，8–16 µm | −0.002 | 0.483 | 0.409 | 0.043 | 0.382 |
| 方向熵，8–16 µm | 0.085 | 0.536 | 0.514 | 0.123 | 0.457 |

数值是 group-balanced Q²，越大越好；不能与上表不同数据、不同聚合协议的数值混为同一实验。这里也没有展示每个对比的重新拟合置信区间，因此应称稳定研究线索，而不是已完成的充分性证明。

谱对比的含义可以从实际代码确定。五个组成部分按 `<8、8–16、16–32、32–64、≥64 µm` 排列，第一对比为：[5]

\[
z_1=\sqrt{6/5}\log\frac{\sqrt{p_{<8}p_{8-16}}}{(p_{16-32}p_{32-64}p_{\ge64})^{1/3}}.
\]

因此它是细尺度组与较大尺度组的几何平均能量比例对比；并非简单的“某一带能量”或“细尺度总能量比”。第二对比是 `<8 µm` 与 `8–16 µm` 的对数比。

**当前最有价值的假设是：深度可能组织了部分细—粗尺度能量分配，而 hatch 组织了另一些尺度内对比和方向信息。** 这不等于已经知道随深度增加表面一定变粗或变细；Q² 本身不能给出单调方向。

还有一个防止过度解释的结果：GAM 中 U+D 对幅值的 Q² 为 0.097，反而低于 D-only 的 0.168；U+D 的方向性也不优于 U。实际拟合在小样本、共同正则化和相关输入下可能出现性能下降，因此不能把某次增减输入的符号直接当作真实信息增减。[3]

### 2.3 总体风险不足以说明各块关系

首批 E02 的三块归一化风险中，GAM 的 U、D 和 D+h 分别为 0.850、0.935 和 0.757；Ridge 则分别为 0.791、0.932 和 0.812。U 比总剂量 Qa 更好，但加入由 U 交叉拟合得到的 Dhat 没有稳定总体额外增益。[4]

这意味着“D 对全部形貌不充分”与“D 对某些形貌成分很有意义”可以同时成立。新路线不应把总体风险中的弱结果看作深度主线的失败，也不应把某一块的正结果倒用为 M→D 已经成功。

## 3. 先把深度、形貌和‘吸收’定义清楚

### 3.1 深度不是粗糙度，也不是峰谷高度

仓库以未加工外部表面校正测量平面；主数据是中央 80×80 µm 的高度 ROI。当前观察算子定义：[6–7]

\[
D=-\operatorname{median}_{\Omega}H(x,y),\qquad
r(x,y)=H(x,y)-\operatorname{median}_{\Omega}H(x,y).
\]

主深度是参考平面下的 ROI 中位去除深度，不是整个槽的最大深度、峰谷高度或总体积。新实验必须继续使用这一固定定义；其他深度定义只能作为注明含义的敏感性任务。

若把完整 H 直接输入模型，让它预测 D，模型只是近似一个已知中位数算子。对形貌—深度关联有科学意义的主输入，应为 r 的特征，并排除原始高度均值、原始最小值、DC 项、已含 D 的体积和用真实 D 归一化的特征。

一个必要的单元测试是向整个加工 ROI 加常数 c。相对形貌 r 应保持不变，D 则变为 D−c。由此也可以直接证明：仅凭相对形貌，不存在适用于所有表面的绝对深度唯一反演。可研究的是特定材料、工艺范围和观测条件下的统计关系，而不是普遍几何恒等式。

现有代码已经分离深度和残差，不应把这种潜在风险写成“仓库已发生标签泄漏”。新分支只需要保持这一边界，并补上防回归测试。[6]

### 3.2 三种‘吸收’不能混用

| 含义 | 可操作的问题 | 当前数据能支持到什么程度 |
|---|---|---|
| 预测信息被形貌承载 | 加入 M 后，某个工艺输入是否不再改善 D 的留出预测？ | 可以检验模型与数据范围内的冗余 |
| 深度作为加工进度压缩 | 给定 D 后，工艺对某些形貌成分的额外预测贡献是否变小？ | 可以检验部分形貌的统计统一 |
| 光学能量吸收或因果中介 | 某参数是否通过形貌改变实际吸收率，从而改变深度？ | 终态回归不能单独辨识，需要额外假设与观测/干预 |

建议论文主文使用“信息承载、条件冗余、残余贡献、深度相关表征”。“吸收率”“热积累”“缺陷密度”等物理名称，必须具有独立证据才能赋给模型潜变量。

### 3.3 同时保留两个统计方向，但只设一个工程目标

设 U 为工艺参数，M 为不含绝对高度的形貌，Z_D 为深度相关的压缩形貌。

主任务检验：

\[
\mathbb E[D\mid U,M]\approx \mathbb E[D\mid U,Z_D].
\]

这检验小表示是否保留深度的条件均值信息。更强的纯形貌链假设是：

\[
\mathbb E[D\mid U,Z_D]\approx\mathbb E[D\mid Z_D].
\]

第二式只有在加入 U 的改善足够小、且不确定性允许作非劣判断时才得到支持。平方误差下主要检验条件均值；不能据此宣布整个条件分布、极端深度风险或未来状态都已充分。条件均值降维与完整分布降维在统计理论上本来就是不同问题。[11]

反向诊断则对每个形貌块比较 M_b←D 与 M_b←D+U。它不改变最终目标 D，而是用来解释究竟哪些工艺—形貌差异被加工进度统一了。

## 4. 文献如何支持与限制该路线

### 4.1 2026 年的直接相关工作：不同空间尺度以不同速度遗忘初始表面

Kažukauskas 等在 Optics Express 2026 年论文中研究飞秒激光深雕熔石英的谱演化。不同频率的初始形貌衰减速率不同；粗糙度可以进入统计稳态，而较大尺度的起伏随后仍可能发展。初始表面条件也影响收敛过程。[8]

这支持“尺度依赖的信息保留与遗忘”这一研究问题，但不支持将该材料的扫描次数、演化速率或频率阈值直接搬到氧化锆。更重要的是，这项工作已经说明“谱分析研究粗糙度演化”本身不足以构成本项目的新意。本项目应增加深度条件化、工艺残余贡献和任务导向压缩的检验。

### 4.2 氧化锆文献提示：必须区分相对特征深度与绝对去除深度

Henriques 等 2023 年在氧化锆 DLIP 研究中发现，减小 hatch 在一定范围内增加纹理特征深度，但进一步增加重叠会同时烧蚀脊部，使特征深度不再增加甚至降低。[9]

这是对“纹理更深就意味着总去除更多”的直接警示。该工作使用 532 nm、10 ps 干涉加工，既不同于本项目的光学条件，其相对特征深度也不同于参考平面下的 ROI 中位深度。它支持观测定义的重要性，而不是提供可直接套用的深度公式。

### 4.3 光吸收与材料去除之间并非单调对应

Mustafa 等对锌和钢的超短脉冲研究显示，初始粗糙度增加可提高吸收，同时增大的有效表面积也提高了单脉冲烧蚀阈值；增加的能量并不必然全部转化为去除。[10]

因此，即使本项目发现粗糙度、坡度和深度有关，也不能把回归系数直接解释为吸收效率。文献提供的是候选机制以及防止单调化解释的依据，不能证明氧化锆发生了相同机制。

### 4.4 方法上更接近目标导向压缩，而不是全形貌自编码

信息瓶颈方法关注压缩输入，同时保留对指定目标有用的信息；条件均值降维关注保留回归目标而非重构输入。这与“最终目标是深度”的要求一致。[11–12]

实际建模无需一开始估计高维互信息或训练变分信息瓶颈。先比较监督 PLS、正则化模型和小维度表示的留出风险曲线，可以更直接地回答任务。图像重构最好的潜变量可能保留了大量方向、相位和纹理信息，却未必最适合预测深度。

### 4.5 串联回归并不自动识别中介机制

因果中介分析需要明确识别假设和敏感性分析。终态 M 和 D 同时测得时，二者可能是共同历史状态的输出，也可能存在加工过程中的双向反馈，不能仅根据 U→M 和 M→D 的拟合结果认定形貌导致深度。[13]

本阶段的准确表述是“统计信息路径”或“预测路径”。机制阶段应转向加工前一刻的形貌 M_N 对下一步深度增量的作用。

## 5. 技术路线总览

建议形成一个共享数据协议、两个解释方向、一个最终目标的路线：

**工艺 U → 相对形貌 M → 深度相关表示 Z_D → 深度 D；保留 U→D 的可检验旁路。**

主分析回答 M 是否帮助预测 D；反向诊断回答 D 是否统一部分 M；物理模块只用于解释深度响应与提升域外可信度，不再以复现所有纹理为先决条件。

| 阶段 | 主要问题 | 产出 | 通过后才开展 |
|---|---|---|---|
| D0 | 深度与形貌是否定义独立、分组正确？ | 标签/特征合同、泄漏测试、分割清单 | 所有拟合 |
| D1 | 哪类形貌携带深度信息？ | 分块预测、工艺/形貌联合对照、OOF 表 | 压缩与解释 |
| D2 | 哪些工艺信息被形貌或深度承载？ | 双向条件贡献矩阵、支持域匹配图 | ‘吸收’类论文结论 |
| D3 | 深度相关形貌能否低维表达？ | 维数—误差曲线、稳定性和非劣检验 | 工艺→压缩形貌→深度 |
| D4 | 深度是否组织跨 pass 的形貌变化？ | N=1–4 家族级分析、同深度/同形貌对照 | 动态机制假设 |
| D5 | 需要哪些物理状态？ | 深度导向 P00/P10/P01/P11 比较 | 小型学习闭包或序列模型 |
| D6，可选 | 中间形貌是否影响下一步去除？ | 真正同位逐步测量与干预 | 因果与状态充分性结论 |

## 6. D0：数据合同与观测边界

### 6.1 保持已有来源层级

历史数据为 200 ROI，但只有 160 个共享测量来源。MAIN180 的 E00 版本构建了 126 个关联分量；SUPP20 的家族清除轨道在该报告中要求从 MAIN180 删除 40 行，留下 140 行训练。这些是具体协议下的结构，不能将 200 ROI 视为 200 个独立重复。[4,7]

建议复用并审计来源—工艺家族关联图，验证同一原始测量、同一派生来源和按当前任务定义必须整体留出的工艺家族不跨训练/测试。先冻结 split，然后统一供所有模型使用。新增深度主任务可以继承分割，但不能沿用旧 A/P/T 的超参数选择结果而不重做训练侧选择。

SUPP20 应当作为有明确历史使用记录的补充域验证。只有确认此前未被用于该假设选择，才能称为未触碰测试集；不得因为本次换了研究问题就将旧数据重新命名为独立盲测。

### 6.2 防泄漏与敏感性测试

主标签采用 canonical D，主形貌只从中心化残差计算。主数据保持 `height_raw`，修复版本单独敏感性分析；修复比例与缺失比例进入 QA/分层诊断，不默认作为材料机理特征。

平移不变性测试必须通过。对于同一 r，任意绝对偏移不得改变 M。不得使用 D、绝对高度均值、原始极值、原始体积及以 D 为分母的派生量作为 M 的输入。

主分析只去常数，避免通过过度局部平面或曲率拟合事先删除可能与深度相关的大尺度信息。不同去趋势强度、ROI 尺寸、滤波和修复方式作为预先定义的敏感性支路，而不是根据结果挑选最优版本。

### 6.3 不能承诺观测不到的尺度

0.5 µm 像素间距的理论采样极限对应约 1 µm 周期；接近极限的特征还受光学分辨率和滤波影响。80 µm 视场也不能可靠识别远大于视场的完整形貌周期。≥64 µm 频带应保留其有限窗口含义，而不是当成已经观测到宏观起伏。

只出现于扫描电镜的纳米孔、成分变化和吸收变化，不能被假定已经由共聚焦形貌完整记录。这是深度预测的信息边界，不是换模型就能消除的问题。

## 7. D1：以深度为唯一主终点的最小实验矩阵

### 7.1 输入按科学含义组织

| 输入组 | 建议主特征 | 作用 |
|---|---|---|
| U：工艺 | log τ、log f、log v、log h、N 的冻结坐标；物理派生量作为单独消融 | 工艺直接基线 |
| A：幅度 | A_med、均值中心 Sq、稳健分位宽度 | 总起伏与尾部 |
| P：尺度 | 现有 4 个 ILR；绝对非 DC 分带能量另设输入版本 | 能量组成与绝对幅值的区别 |
| G：几何 | 固定滤波下的均方根坡度、相关长度等少量特征 | 局部坡度与横向组织 |
| T：方向 | A2、方向熵；8–16 µm 主版本，其他波段作敏感性 | 检验几何纹理信息是否与 D 冗余 |

首轮以约 10–20 个有定义的标量特征为宜，不建议立即从 160×160 像素训练高容量编码器。若多个特征仅是数学重复，应分组消融或正则化，不能把重复表示的增益当成独立机制。

**代码细节：A_med 不完全等于通常意义的 Sq。** 当前 A_med 是围绕中位数的均方根；正交 DCT 去掉 DC 后的能量对应围绕均值的方差。因此，若希望将谱比例 p_b 转换为单位 µm² 的绝对分带能量，应直接使用同一非 DC DCT 系数并除以像素数，或在一致归一化下使用 p_b×Sq_mean²，不能无条件替换为 p_b×A_med²。[5–6]

λ/h、h 归一化频谱属于显式使用工艺的特征，应标为 U+M，不能在 M-only 模型中悄悄包含 h。方向强度可作为旋转不变量使用，但缺少扫描方向来源时不解释为真实扫描轴效应。

### 7.2 必做模型组

| 编号 | 输入→输出 | 判读问题 |
|---|---|---|
| B0 | 训练侧常数→D | 是否超越零模型 |
| BQ | 总剂量 Qa→D | 总能量摘要是否足够 |
| BU | 完整 U→D | 必须超越或非劣的工艺基线 |
| BA/BP/BG/BT | 单一形貌块→D | 哪种形貌含有深度相关信息 |
| BM | 全部非 DC 形貌 M→D | 无绝对高度条件下的诊断能力 |
| BUM | U+M→D | 形貌与工艺是否互补 |
| BZ/BUZ | Z_D→D、U+Z_D→D | 小维度形貌是否保留深度性能 |
| BCHAIN | U→预测 Z_D→D | 加工前是否能够使用该路线 |

首轮模型为 Ridge 与受正则化约束的 GAM，再加入一个有相同调参预算的小型非线性树模型作为模型偏置检查。模型复杂度应在相同外层训练侧选择；不应只比较弱线性工艺模型与高容量联合模型。

优先报告分量加权 MAE（µm），并报告 RMSE（µm）、一致定义的 OOF Q²、各来源/工艺家族误差、外推支持情况。报告样本覆盖、无定义特征和失败模型，不能只展示能成功拟合的子集。

## 8. D2：把‘吸收了什么’变成可以计算的对照

### 8.1 形貌承载了多少深度相关信息

令 R_D(X) 表示同一外层协议下用 X 预测 D 的风险，越低越好。定义：

\[
\Delta_{M\mid U}=R_D(U)-R_D(U,M),\qquad
\Delta_{U\mid M}=R_D(M)-R_D(U,M).
\]

第一项回答“实际形貌提供了多少工艺之外的预测帮助”；第二项回答“已知形貌后工艺还剩多少必要信息”。这些是预测风险差，不是物理能量比例，也不是互信息的比特数。

| 结果 | 合理解释 | 路线选择 |
|---|---|---|
| ΔM∣U 明显为正，ΔU∣M 很小 | 形貌与深度关联强，工艺在该范围内可能近似冗余 | 可尝试纯形貌压缩 |
| 两者均明显为正 | 两种信息互补 | 保留 U+Z_D |
| ΔM∣U 很小，ΔU∣M 明显为正 | 现有形貌未提供足够增益 | 深度工艺基线为主，形貌作解释 |
| 两者都很小，但 BU/BM 都有能力 | 信息大幅重叠 | 依据成本与部署选表示 |
| 所有模型都弱 | 观测、标签或支持范围不足 | 不应强行做串联模型 |

“差异不显著”不能被当成信息已吸收。需要预先规定允许损失 ε_D，并要求风险差的上置信界落在容限内。ε_D 应由深度测量重复性或应用容差确定；当前单对严格重复不能构成可靠噪声下限，若没有独立依据，应报告容限敏感性而非随意固定百分比。[7]

### 8.2 哪个工艺因素的深度贡献被形貌承载

对因素或因素组 j，比较：

\[
G_j^{\rm before}=R_D(U_{-j})-R_D(U),
\]
\[
G_j^{\rm after}=R_D(U_{-j},M)-R_D(U,M).
\]

只有当 before 确实可辨、after 在容限内变小，且趋势对合理模型和分组稳健，才能说该输入的深度预测贡献被形貌近似替代。不要把 after/before 或 1−after/before 命名为“中介比例”；分母小、输入相关和交互均会破坏这种直觉。

P 固定导致 f 与 E_p 完全耦合，应当作为 frequency/pulse-energy 耦合因素讨论。总剂量是其他输入的确定性组合，也不是一个独立随机因素。N≥5 与补充 session 混杂，应单列而非通过回归系数强行分离。[7]

### 8.3 深度究竟统一了哪些工艺—形貌差异

这是第二个、尤其符合“深度作为压缩目标”的诊断方向。对每个形貌块 b 比较：

\[
R_{M_b}(D),\quad R_{M_b}(U),\quad R_{M_b}(D,U),\quad R_{M_b}(D,h).
\]

进一步逐因素比较给定 D 前后的预测贡献。若 ilr_z1 在给定 D 后几乎不再需要某些工艺变量，可称这些变量对该形貌成分的差异被深度进度坐标近似统一；若 A2 仍强烈需要 h，则该纹理信息没有被 D 压缩掉。

已有 E02 正好可以成为该方向的起点，但需要重新做逐目标配对区间、模型稳健性和支持域限制，而不是只引用总体 A/P/T 风险。

### 8.4 残差和匹配只作交叉验证诊断

可在训练侧交叉拟合 f(U) 与 g(U)，得到深度残差 r_D 和形貌残差 r_M，并检验其留出预测关系。这有助于区分“共同由 U 驱动”和“形貌还记录了未建模的终态差异”，但残差关系仍非因果，残差不相关也不等于条件独立。

建立两个配对图集：同深度、异工艺/异形貌；相似形貌、不同深度。配对必须限制共同支持域与 session，展示所有符合规则的配对并按家族汇总，不能按预期故事挑图。

## 9. D3：深度导向的压缩与加工前串联模型

### 9.1 从可解释的小表示开始

先比较 PCA 与监督 PLS 的 1、2、3、4 维表示，以及少量谱对比的稀疏组合。PCA 是不使用深度标签的对照，PLS 或监督指数用于寻找深度相关方向。全部标准化、降维和维数选择都必须在训练侧完成。

接受一个压缩表示的条件不是图像重构漂亮，而是 Z_D 或 U+Z_D 对深度的留出风险在预定容限内不劣于完整 M 或 U+M，同时对工艺家族与预处理方式稳定。若真正关心极深加工风险，还需检查残差分布和区间覆盖，不能只保留均值性能。

标量预测 D 并不证明底层物理只有一维：对平方损失，E[D∣M] 本身就是一个标量。只有在限制编码器复杂度、跨工艺稳定性和解释一致性之后，低维结构才具有更强科学意义。[11–12]

### 9.2 加工后诊断与加工前预测必须分开记分

加工后诊断使用真实 M。它可以解释加工结果，但既然已有参考高度场，直接读取 D 通常更加直接；其工程意义主要是理解机制、使用无参考表面观测估计深度，或为未来在线传感代理建立依据，不能夸大为当前计量成本降低。

加工前只能使用 U 预测形貌或深度相关表示。因此要另外评价：

\[
U\to\widehat Z_D(U)\to\widehat D.
\]

测试时不能给第二级模型真实形貌。训练时第二级应尽可能使用交叉拟合的预测表示，或者对“真实中间表示训练、预测中间表示测试”的分布差异进行明确建模，不能把后验成绩当作加工前成绩。

如果 \(\widehat Z_D=g(U)\) 是确定性函数，它没有给测试样本增加 U 之外的观测信息。它仍可能通过监督信号、物理归纳偏置或小样本正则化改善性能，但必须与同预算的 U→D 直接模型公平比较。

中间表示还存在坐标一致性问题：不同折独立学习的潜变量可能旋转或翻号，不能不加处理地拼成同一训练表。第一版优先使用意义固定的谱对比、幅值等中间量；学习潜表示时，应把编码器、工艺映射和深度读出作为整体管线嵌套评价，或者只在训练侧完成坐标对齐。

一个折内流程是：外层训练 T 中再做分组内层切分；所有编码器、特征选择、形貌预测器和超参数只看对应内层训练部分；用内层留出表示训练/选择后级；冻结选择后在 T 重拟合；外层测试 V 只允许以 U 生成中间表示并输出深度。最终保存每个预测的上游样本集合和模型来源。

### 9.3 深度神经模型只是后续选项

若简单监督表示与非线性静态模型仍存在稳定差距，可以考虑小型变分信息瓶颈或带稀疏约束的编码器。但不建议把互信息估计或大图像网络作为第一阶段硬依赖。网络的性能结论与“发现了材料状态”的结论应分别验收。

## 10. D4：利用 pass 数据研究加工进度，但不伪造时间序列

当前 N=1–4 数据是 15 个基础工艺条件各加工到四个不同终态，测量来自不同空间位置，不是同一表面的重复跟踪。N=5–6 又来自补充 session。因此现有数据只能称为横截面伪时序；N=4→5 不能独立识别 pass 与批次效应。[7]

建议主分析限制在 N=1–4，按基础工艺家族整体留出。对每个形貌块比较 M_b=f_b(N)、M_b=f_b(D)、M_b=f_b(D,h) 和 M_b=f_b(D,U_other)。若 D 比 N 更好地统一跨家族的形貌变化，它可能是比“扫描次数”更有解释力的加工进度坐标。

第一优先目标是 ilr_z1，因为已有 D-only 的可预测线索；幅值、绝对分带能量和方向性作为平行检验。应允许非单调和平台期，不预先把曲线强制为随 D 增长或下降。

可以比较条件层面的 D_N+1−D_N，但必须写清这是不同加工位置的组间差，不是同一表面的真实瞬时增量。数据不能支持将各终态高度图逐像素对齐为一条真实动态轨迹。

如果后续允许补充测量，最有价值的不是简单增加很多终态 ROI，而是对同一位置在 N 和 N+1 前后测量：比较 `(D_N,U_next)→ΔD` 与 `(D_N,M_N,U_next)→ΔD`。这才能把“终态指纹”推进到“下一步响应所需状态”。新增设计的样本量应按测量误差、预期效应和家族变异规划，不应照搬一个固定经验数量。

## 11. D5：以深度为中心复用物理模块

### 11.1 先检验明确存在的几何反馈

仓库说明聚焦策略固定在原始顶面；现有物理模块已经包含局部深度导致的离焦以及有效孵化状态。因此，与其首先增加不受观测约束的材料记忆，建议先比较 P00、P10、P01、P11 在深度任务上的表现。[7,14]

| 模型 | 离焦 | 有效孵化 | 主要检验 |
|---|---|---|---|
| P00 | 无 | 无 | 基本阈值去除与剂量累积 |
| P10 | 有 | 无 | 深度几何反馈是否必要 |
| P01 | 无 | 有 | 有效阈值演化是否有额外解释力 |
| P11 | 有 | 有 | 两种反馈是否互补且可辨识 |

原模块中的主要形式是：[14]

\[
w(d)=w_0\sqrt{1+(d/z_R)^2},
\]
\[
d_{t+1}(x,y)=d_t(x,y)+\delta\,[\ln(F_t/F_{th}(q_t))]_+.
\]

其中 q 是模型中的有效孵化状态，不是实测温度或缺陷密度。数值正确性、梯度检查和物理参数辨识必须分开报告。光腰等 nominal 光学参数不能直接等同于加工槽内的真实光场。

### 11.2 从场到深度仍须使用同一观察算子

物理更新作用于局部 d(x,y)，最后用 canonical observer 输出 D。中位数不是线性算子，不能把各步的中位深度增量直接相加就视为完整场递推的精确结果。

物理校准只用外层训练数据；所有候选必须使用同一参数预算。先固定可独立测定的光学量，只校准少数有效参数。若同时自由学习吸收倍率 η 和阈值 F_th，深度对数比通常主要看到 η/F_th，二者容易互相补偿；应固定一项或引入独立测量，报告 profile/可接受参数集合，而不是只报告一个最优值。

### 11.3 学习闭包只在基线不足时进入

可对阈值或有效去除长度中的一个量加入受限低维修正，检验深度相关形貌是否解释物理残差。不要同时给多个不可辨识参数都加灵活神经函数。

若修正使用真实终态 M，它是后验诊断而非正演模型。真正正演时需使用预测中间状态或上一阶段已观测形貌，并避免将待预测的终态深度通过形貌返回模型。

单线宽度或谱峰模型在过去失败，不等于所有深度物理基线都会失败；反之，预测深度成功也不证明模型重现了完整谱结构。深度主任务应该独立报告其有效性范围。

## 12. 不确定性、验收与失败结果的利用

主分析至少预先定义两个风险差：ΔM∣U 和 ΔU∣M；其余分块、维数和因素删除作为有组织的次分析。置信区间以工艺家族/关联分量为重采样单位，不能把像素、五个折或大量重叠配对当独立样本。

固定 OOF 上的分量 bootstrap 只反映条件于既有模型的变化。关键结论最好增加训练重采样或重复完整分组拟合；若计算预算不足，要明确区间未覆盖模型选择不确定性。历史数据经过探索也不能通过 bootstrap 变成确认性新数据。

| 验收门 | 所需证据 | 没通过时的合理结论 |
|---|---|---|
| G0：无标签捷径 | 位移不变性、来源清除、输入清单通过 | 暂停主实验，先修定义 |
| G1：形貌有深度价值 | 分块或联合增益达到预定有意义范围 | 现有观测下不支持形貌提升；保留工艺深度模型 |
| G2：信息近似冗余 | 加入 U 的风险差上界小于 ε_D | 纯形貌链不能成立，保留工艺旁路 |
| G3：压缩有效 | 低维风险非劣、跨家族和预处理稳定 | 增加合理维数或保留全特征，不包装一维规律 |
| G4：部署可用 | 只用 U 生成中间表示后仍有效 | 仅后验解释成立，不宣称加工前预测改善 |
| G5：状态解释 | 真正前后测量/干预支持下一步增量关系 | 仅终态关联，不宣称材料记忆或因果中介 |

失败同样可能产生论文价值。例如：方向纹理高度记录工艺几何，却不能明显改善深度预测；粗糙度进入平台后深度继续变化；有限观测窗口无法承载某些深度相关信息。这些是明确的边界结果，而不是为了保留串联网络必须掩盖的负例。

## 13. 最值得发展的论文贡献

### 13.1 深度相关形貌与工艺几何纹理的分离

最接近现有证据的中心问题是：能否用少量幅值和细—粗尺度对比保留深度预测能力，同时证明某些高度可预测的方向纹理主要记录几何工艺而非深度进度？它要求双向条件增益和跨工艺验证，而不只是相关系数或重要性图。

建议题目方向：**“氧化锆超快激光加工中面向去除深度的形貌信息压缩与工艺残余贡献”**。标题中的“吸收”可以在正文定义为信息承载，避免与实际光吸收混淆。

### 13.2 深度作为多尺度形貌演化的进度坐标

研究 N、剂量与 D 哪个更能统一跨工艺的谱变化；量化哪些波段保留初始/工艺差异，哪些波段趋于同一终态。与已有谱演化论文的区别，应是深度条件化和工艺信息压缩的可检验边界，而不是首次计算 PSD。[8]

### 13.3 深度反馈下的最小有效状态

在静态分析获得明确证据后，再比较离焦、有效孵化和小维度形貌闭包。核心贡献是“为指定深度响应需要保留什么状态”，不是“使用了 Mamba”。只有真实序列或可核验的前向预测收益支持时，才推进序列网络。

## 14. 建议的执行顺序与成果清单

**首先完成 D0、D1、D2。** 最先应得到的不是新的高容量模型，而是 U-only、M-only、U+M 三组深度留出结果，以及按形貌块和因素分解的双向贡献矩阵。现有 E02 的 D→形貌表可以直接作为反向诊断基线，但新主任务必须以 D 重新做内层模型选择。

随后执行 D3 的低维压缩，输出维数—深度误差的非劣曲线，并单独评价 U→预测表示→D。D4 充分使用现有 N=1–4 的条件家族，形成有限但诚实的加工进度分析。D5 按实际静态结论选择必要物理状态，不将求解器数值门或网络实现当作科学结论。

建议每个新 run 至少保存：`protocol_snapshot.yaml`、`input_provenance.csv`、`split_manifest.csv`、`feature_audit.csv`、`oof_depth_predictions.csv`、`depth_metrics.csv`、`paired_risk_intervals.csv`、`factor_retention.csv`、`morphology_given_depth.csv`、`compression_curve.csv`、`failure_registry.csv` 和 `compute_cost.csv`。这些是拟新增的成果名称，不是声称仓库已经具有相应实验。

最终应能回答四句话：哪些形貌对深度有用；哪些工艺贡献在这些形貌中近似冗余；哪些信息不能被深度统一；在加工前无法获得真实形貌时，这种压缩是否仍有实际预测价值。

## 附录 A：建议新增的代码位置

以下均为新分支建议，不覆盖已有 `phase2_8_r1` 或 `task_state_v1/E02` 历史产物。

```text
config/depth_target_v1/protocol.yaml
src/depth_target/features.py
src/depth_target/splits.py
src/depth_target/models.py
src/depth_target/conditional_gain.py
src/depth_target/compression.py
experiments/depth_target_v1/00_contract.py
experiments/depth_target_v1/01_depth_baselines.py
experiments/depth_target_v1/02_bidirectional_information.py
experiments/depth_target_v1/03_task_compression.py
experiments/depth_target_v1/04_process_to_depth_chain.py
experiments/depth_target_v1/05_pass_progress.py
experiments/depth_target_v1/06_depth_physics.py
outputs/depth_target_v1/<explicit_run_id>/
```

可复用的已有模块包括 `src/task_state_learning/observers.py`、`src/composition.py`、`src/task_state_learning/grouping.py` 和 `src/task_state_learning/physics.py`。复用不意味着可以省略新标签任务的分组与误差定义测试。[5–6,14]

## 附录 B：主协议草案

```yaml
schema: depth_target_v1_draft
status: PROPOSED_NOT_EXECUTED
repository_ref: f80bca0fa604197bc217ca725fd573431483d04b
primary_target:
  name: D_canonical
  unit: um
  definition: negative_median_of_external_plane_referenced_ROI
primary_height: height_raw
morphology:
  remove_absolute_offset: true
  primary_detrend: median_only
  forbidden:
    - canonical_depth
    - absolute_height_mean_or_minimum
    - raw_height_DC
    - target_normalized_features
    - target_equivalent_volume
  blocks: [amplitude, spectral_composition, absolute_band_energy, geometry, direction]
cohorts:
  primary: MAIN180
  historical: HIST200_descriptive_only
  supplement: SUPP20_family_purged_domain_stress_with_usage_audit
splits:
  unit: audited_source_and_process_family_components
  outer: reuse_validated_frozen_partitions_after_audit
  inner: grouped_training_only
models:
  primary: [Ridge, regularized_GAM]
  nonlinear_check: small_tree_model_with_matched_tuning_budget
  compression_dimensions: [1, 2, 3, 4]
metrics:
  primary: component_balanced_MAE_um
  secondary: [RMSE_um, pooled_OOF_Q2, groupwise_error, coverage]
primary_comparisons:
  - M_added_to_U_for_D
  - U_added_to_M_for_D
secondary_comparisons:
  - morphology_blocks_for_D
  - factor_gain_before_and_after_M
  - morphology_conditioned_on_D_and_U
  - compressed_vs_full_morphology
  - process_predicted_representation_vs_direct_process_model
noninferiority:
  tolerance_um: UNSET_requires_metrology_or_application_justification
  significance_is_not_equivalence: true
uncertainty:
  resampling_unit: independent_components_or_families
  distinguish_fixed_OOF_from_full_refit: true
sequence_claim:
  N1_to_4: cross_sectional_pseudo_temporal_only
  N4_to_5: pass_session_confounded
causal_claims: not_authorized_by_terminal_prediction_alone
optical_absorption_claims: require_independent_evidence
```

## 来源

仓库来源均固定到同一提交；目录存在不作为结果证据。期刊研究只在各自材料、光学条件和测量定义下成立。

[1] physics-guided-mamba2，`outputs/phase2_8_r1/predictability_spectrum.csv`。https://github.com/milumilelu/physics-guided-mamba2/blob/f80bca0fa604197bc217ca725fd573431483d04b/outputs/phase2_8_r1/predictability_spectrum.csv

[2] physics-guided-mamba2，`config/task_state_v1/depth_compression.yaml`。https://github.com/milumilelu/physics-guided-mamba2/blob/f80bca0fa604197bc217ca725fd573431483d04b/config/task_state_v1/depth_compression.yaml

[3] physics-guided-mamba2，E02 MAIN180 `native_metrics.csv`，run `20260906T111859Z_E02_b901bea5_533009c_68a562`。https://github.com/milumilelu/physics-guided-mamba2/blob/f80bca0fa604197bc217ca725fd573431483d04b/outputs/task_state_v1/20260906T111859Z_E02_b901bea5_533009c_68a562/native_metrics.csv

[4] physics-guided-mamba2，`outputs/task_state_v1/EXECUTION_STATUS_ZH.md`，2026-09-06 首批报告，非全部后续实验完成证明。https://github.com/milumilelu/physics-guided-mamba2/blob/f80bca0fa604197bc217ca725fd573431483d04b/outputs/task_state_v1/EXECUTION_STATUS_ZH.md

[5] physics-guided-mamba2，`src/composition.py`，五部分非 DC 组成与 ILR 对比矩阵。https://github.com/milumilelu/physics-guided-mamba2/blob/f80bca0fa604197bc217ca725fd573431483d04b/src/composition.py

[6] physics-guided-mamba2，`src/task_state_learning/observers.py`，深度与形貌观察算子。https://github.com/milumilelu/physics-guided-mamba2/blob/f80bca0fa604197bc217ca725fd573431483d04b/src/task_state_learning/observers.py

[7] physics-guided-mamba2，`现有数据基础说明_v2.md`，尤其数据层级、外部平面预处理、pass 结构与固定焦点说明。https://github.com/milumilelu/physics-guided-mamba2/blob/f80bca0fa604197bc217ca725fd573431483d04b/现有数据基础说明_v2.md

[8] Kažukauskas E., Butkus S., Latvys T., Jukna V., Paipulas D. Spectral domain analysis of surface roughness evolution during femtosecond laser deep engraving of glass. Optics Express 34, 10334–10350, 2026. https://doi.org/10.1364/OE.588502

[9] Henriques B., Fabris D., Voisiat B., Boccaccini A.R., Lasagni A.F. Fabrication of functional zirconia surfaces using a two-beam interference setup employing a picosecond laser system with 532-nm wavelength: Morphology, microstructure, and wettability. Journal of the American Ceramic Society 106, 7189–7193, 2023. https://doi.org/10.1111/jace.19357

[10] Mustafa H., Mezera M., Matthews D.T.A., Römer G.R.B.E. Effect of surface roughness on the ultrashort pulsed laser ablation fluence threshold of zinc and steel. Applied Surface Science 488, 10–21, 2019. https://doi.org/10.1016/j.apsusc.2019.05.066

[11] Cook R.D., Li B. Dimension reduction for conditional mean in regression. Annals of Statistics 30(2), 455–474, 2002. https://doi.org/10.1214/aos/1021379861

[12] Tishby N., Pereira F.C., Bialek W. The information bottleneck method. Allerton Conference, 1999; arXiv manuscript posted 2000. https://arxiv.org/abs/physics/0004057

[13] Imai K., Keele L., Tingley D. A general approach to causal mediation analysis. Psychological Methods 15(4), 309–334, 2010. https://doi.org/10.1037/a0020761

[14] physics-guided-mamba2，`src/task_state_learning/physics.py`，P00/P10/P01/P11 与局部离焦/有效孵化更新。https://github.com/milumilelu/physics-guided-mamba2/blob/f80bca0fa604197bc217ca725fd573431483d04b/src/task_state_learning/physics.py
