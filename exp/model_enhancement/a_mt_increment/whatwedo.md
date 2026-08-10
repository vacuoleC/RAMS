# A 方向探索：增量目标 × 完整多任务 · 事件记录（whatwedo.md）

> 关联：`rethinking.md`（思考）、`trydoing.jsonl`（执行）、`results.md`（结果）
> 政策：`exp/EXPLORATION_POLICY.md`（探索只在 exp/ 下，git 原子提交，双同步）

## 事件

### [2026-08-10 13:40] 探索建立
- 方向：增量目标 × 完整多任务（M1 增量 + M2 分层 + M4 预警）是否叠加
- 背景：B1/B2/B7 隔离测增量（但协议不同：B1 单任务 28-fold，B7 多任务 17 窗口）；框架比较证明多任务单任务差 2.25（绝对浓度口径）。两线从未同协议组合。
- 产出：`run_a.py`、`rethinking.md`、`trydoing.jsonl`（本记录）
- 协议一致性已校验：17 窗口持久化 MAE=1.1281 = B7 crps_persist，测试段与 B7 完全相同。

### [2026-08-10 13:50] 本地冒烟
- `--smoke`（1 窗口 × 3 seed × 2 epoch CPU）全链路通过
- 修复：m2/m4 头直接接 raw 输入 bug → 改 model.forward 取隐藏态
- 冒烟 2 epoch 不收敛，数字不用于结论；窗口 1 高浓度段持久化 MAE=3.01

### [2026-08-10 14:05] H100 全量启动
- 2 arm × 3 seed × 17 窗口 × 30 epoch = 102 次训练
- nohup 后台运行（PID 686926），40.8 分钟完成

### [2026-08-10 15:10] 全量完成 + 结论
- `results.json` + `run_full.log` 从算力机 pull 回本地
- 结论：多任务不叠加点精度（轻微稀释，不显著），但把区间校准拉回理想（覆盖 0.672→0.806）
- 协议一致性：多任务 arm 复现 B7 abs_delta；持久化 1.1281 与 B7 逐位一致
- 产出：`results.md`（完整报告）、rethinking.md / trydoing.jsonl 回填结论
