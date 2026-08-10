# C 跨 seed 集成 · 探索标记（whatwedo.md）

- **标记时间**：2026-08-10 13:20（东八区）
- **任务**：C 跨 seed 集成（exp/model_enhancement/c_seed_ensemble/）
- **背景**：文献共识"单模型 ≈ 小集成"。对 B7 最优的 abs_delta 增量分位数模型，用不同随机种子训练多副本，p10/p50/p90 各自平均，预期降方差、稳定 CRPS/覆盖率
- **探索依据**：exp/EXPLORATION_POLICY.md（文件级探索标记）；复用 B1/B2/B7 同一滚动窗口协议（训练 730d/测试 90d/步长 45d，17 窗口）
- **方法**：H100（sensecore）5 seed（0/1/2/3/4）× 17 窗口 × 30 epoch，唯一变量=随机种子；集成=分位数跨 seed 平均；对照 avg_single（最公平）/best_single（Oracle 上界）/worst_single；评估逐视界 CRPS + p50RMSE + 覆盖率（重点看窗口间方差）
- **产物**：
  - `run_c.py`（实验脚本，H100 冒烟通过，与 B7 同一窗口逐位对齐验证）
  - `trydoing.jsonl`（执行日志）
  - `rethinking.md`（思考记录）
  - `results.md`（结论）
  - `results.json`（统计量，无原始数据）
- **合规**：只输出统计量/CRPS/RMSE/覆盖率，不打印原始数据行；不触碰 rams/ 冻结代码（仅 import 复用）；GPU 共享不抢占 hunyuanvideo 作业
- **状态**：✅ 探索完成（2026-08-10 13:20，H100 全量 85 模型，59.7 分钟含 GPU 竞争）
- **结论摘要**：跨 seed 集成**边际值得（borderline）**——CRPS 0.8716 vs avg_single 0.8932（**-2.42%，17/17 窗口全优、8/8 视界全优**），RMSE 1.745 vs 1.776（**-1.72%，未过 2% 线**），覆盖率 0.820 vs 0.805（**+1.5pp 但窗口 std 仅 -1.2%，方差未降**）；集成 vs best_single 反而差 1.46%（best 是不可实现 Oracle）。根因：该模型 seed 间预测高度接近（seed CRPS std 均值 0.029，仅占均值 3.3%），"单模型≈小集成"成立得彻底，集成收益被封死。价值在**确定性**（消除选错 seed 的坏运气）而非精度。详见 results.md。
- **边界决策**：预算敏感→正式模型单 seed 即可（差 2.4% CRPS / 1.7% RMSE，省 5× 算力）；稳健性优先→可上集成（17/17 窗口零翻车、实现零成本，代价 5× 训练时间）；要更大集成收益需先引入 seed 间多样性（数据子采样/bagging/多 dropout），另开方向——待主会话审核
