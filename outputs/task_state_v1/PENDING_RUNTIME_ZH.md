# 待续运行记录

2026-09-07：pip 安装 numba==0.67.0 / llvmlite==0.49.0，统一 exec session **30305**。最近轮询仍为活动状态；下一轮必须重新调用 write_stdin 确认，不能仅凭本文件认定在跑。下载完成后检查安装结果，再运行：

```powershell
$env:NUMBA_NUM_THREADS='8'
.venv/Scripts/python.exe -X utf8 -B -m unittest discover -s tests/task_state_v1 -p test_finite_scan.py -v
.venv/Scripts/python.exe -X utf8 -B experiments/task_state_v1/03_validate_finite_roi.py
```

有限参考只因模型无横向状态耦合，才可以独立计算 ROI 点；这不是对含全局 learned closure 的训练加速证明。所有扫描道/脉冲位置来自完整有限 200 μm 调度；tail_waists 截断误差仍须 6/8 对比。窗口扩展会从整个多 pass 初态重跑，不能直接补尾部曝光。正式校准未启动。
