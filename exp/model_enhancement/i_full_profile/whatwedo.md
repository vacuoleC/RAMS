# I 预测全剖面 20 层 · 探索标记（whatwedo.md）

- **标记时间**：2026-08-10 14:45（东八区）
- **任务**：I 预测全剖面（exp/model_enhancement/i_full_profile/）
- **背景**：目前所有模型都预测 conc_0.5（表层单层），但数据有 20 层完整剖面（M3 已验证 20 层输入降误差 32%）。短板识别——数据没用足：只预测单层，但数据是全剖面
- **探索依据**：exp/EXPLORATION_POLICY.md（文件级探索标记）；同一滚动窗口协议（训练 730d/测试 90d/步长 45d，17 窗口）
- **方法**：H100（sensecore）同一协议下对比 3 变体（3 seed [0,1,2]，全部还原 conc 单位评估）：surface（表层单层基线）/ full（全剖面 20 层）/ key（关键层 {0.5,3.5,7.0}）；目标按层定义 abs_delta Δ_h(d)=conc_{t+h}(d)-conc_t(d)；评估表层 CRPS/RMSE/覆盖率（3 变体可比）+ full 全剖面重建 RMSE（表层/中层/底层）
- **关键决策**：冻结 Trainer/MultiTaskLoss 只支持单层目标 → 在 run_i.py 内复制冻结训练协议（分位数损失+M2/M4 交叉熵+同权重，仅泛化 M1 分位数损失到多层），不碰 rams/ 冻结代码；冒烟验证 surface 变体与 B7/C 基线逐位一致（CRPS 2.4357）
- **产物**：
  - `run_i.py`（实验脚本）
  - `trydoing.jsonl`（执行日志）
  - `rethinking.md`（思考记录）
  - `results.md`（结论，全量跑完后写）
  - `results.json`（统计量，无原始数据）
- **合规**：只输出统计量/CRPS/RMSE/覆盖率，不打印原始数据行；不触碰 rams/ 冻结代码（import 复用组件）
- **状态**：🔄 运行中（H100 全量 3 变体 × 3 seed × 17 窗口，2026-08-10 14:42 启动，预计 ~2 小时）
- **冒烟发现**（1 窗口 × 1 epoch × 1 seed，仅链路验证非结论）：key 表层 CRPS 2.242 < surface 2.436 ≈ full 2.412（关键层多任务对表层有增益信号，待全量证实）
