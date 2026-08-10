# D: wind_u 交叉特征在增量目标下重测 · 探索标记（whatwedo.md）

- **标记时间**：2026-08-10 12:20（东八区）
- **任务**：D: wind_u 交叉特征在增量目标下重测（exp/model_enhancement/d_wind_cross/）
- **背景**：T1 在绝对浓度目标下测过 wind_u（20 层边际降 2.1%，5 层反而增）；M5 证明 wind_u 短滞（≤2d）是唯一有信号的气象直驱（wind_u→cyano τ=1 r=+0.072、wind_u→temp_surf τ=2 r=+0.125）。**关键洞察**：绝对浓度里自相关占主导、气象被淹没；增量目标（Δ=conc_{t+h}-conc_t）去掉自相关后，wind_u 的"风大时浓度如何偏离"信号可能浮现
- **探索依据**：exp/EXPLORATION_POLICY.md（文件级探索标记）；协议复用 B7 abs_delta 增量基线（训练 730d/测试 90d/步长 45d，17 窗口）
- **方法**：H100（sensecore）同一增量协议下对比 4 特征变体（唯一差异 = 特征通道，3 seed × 30 epoch）：base（现输入 27 通道）/ wind_u（+短滞通道 28）/ cross（+wind_u×conc 交叉 28）/ both（+两者 29）。wind_u = wind_speed×cos(deg2rad(wind_dir))（与 T1/M5 一致）；cross = 每时刻 wind_u×conc_0.5，T=24 窗口覆盖 0~3 天短滞。评估还原 conc 单位：逐视界 CRPS/RMSE vs base + 各自持久化
- **产物**：
  - `run_d.py`（实验脚本，本地 CPU 冒烟通过）
  - `trydoing.jsonl`（执行日志）
  - `rethinking.md`（思考记录）
  - `results.md`（结论，全量跑完后写）
  - `results.json`（统计量，无原始数据）
- **合规**：只输出统计量/CRPS/RMSE；不打印原始数据行；不触碰 rams/ 冻结代码（仅 import 复用）
- **状态**：🔄 探索进行中（本地冒烟通过，全量待算力机 H100 跑）
- **边界决策**：待全量结果后写入（理解偏差/效果偏差/损失）——见 lossing.md
