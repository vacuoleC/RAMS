# K 方向探索：两阶段训练（解"精度 vs 校准"矛盾）· 事件记录（whatwedo.md）

> 关联：`rethinking.md`（思考）、`trydoing.jsonl`（执行）、`results.md`（结果）
> 政策：`exp/EXPLORATION_POLICY.md`（探索只在 exp/ 下，git 原子提交，双同步）
> 参考：`exp/model_enhancement/a_mt_increment/`（A 方向基准：单任务 0.857/1.665/覆盖0.672，多任务 0.891/1.774/覆盖0.806）

## 事件

### [2026-08-10 14:30] 探索建立
- 方向：两阶段训练（Stage1 单任务训 M1 增量保精度 → Stage2 冻结/解冻 backbone 微调多任务头保校准）能否解 A 方向的精度-校准矛盾
- 产出：`run_k.py`、`rethinking.md`、`trydoing.jsonl`（本记录）
- 设计：与 A 完全同口径（17 窗口协议/特征/增量目标/评估），single/multi 从 A 复现作对照；ts_freeze / ts_full 两个 Stage2 变体隔离"冻结"变量；3 seed

### [2026-08-10 15:05] 本地冒烟
- `--smoke`（1 窗口 × 3 seed × 2 epoch CPU，9.6 分钟）全链路通过
- 四 arm 逐视界 CRPS/RMSE/覆盖率均输出，results_smoke.json 生成
- 窗口1=高浓度段（持久化 3.0136，与 A 窗口1 一致 → 协议对齐）；M2/M4 acc 正常（multi M2=1.0/M4=0.748，ts_freeze 0.996/0.757，ts_full 0.985/0.766）
- 冒烟 2 epoch 未收敛，数字仅验证链路

### [2026-08-10 15:15] H100 全量启动
- 4 arm × 3 seed × 17 窗口 = 204 次训练（single/multi 30ep，ts 20+10ep）
- nohup 后台运行（PID 948383），`run_full.log`

### [2026-08-10 16:40] H100 全量完成 + 结论
- 4 arm × 3 seed × 17 窗口全量完成，`results.json` + `run_full.log` pull 回本地
- 协议一致性：single/multi 逐位复现 A（0.857/1.665/0.672 与 0.891/1.774/0.806）；持久化 1.1281 = A/B7
- 结论：**ts_freeze（Stage1 单任务 + Stage2 冻结 backbone）是 4 arm 最优点精度**（CRPS 0.808 / RMSE 1.583，全部 8 视界最优，16:1 胜 p<0.001），覆盖率显著抬到 0.704（+3.2pp p<0.001）但未追平多任务 0.806 / 80% 目标
- ts_full（解冻）两头不占优 → 冻结 backbone 是保住 Stage1 精度的必要条件
- 产出：`results.md`（完整报告）、rethinking.md / trydoing.jsonl 回填结论
