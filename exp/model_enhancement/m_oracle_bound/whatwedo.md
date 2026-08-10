# M 方向探索：oracle 信息上限探针 · 事件记录（whatwedo.md）

> 关联：`rethinking.md`（思考）、`trydoing.jsonl`（执行）、`results.md`（结果）
> 政策：`exp/EXPLORATION_POLICY.md`（探索只在 exp/ 下，git 原子提交，双同步）

## 事件

### [2026-08-10 15:40] 探索建立
- 方向：oracle 信息上限探针——藻类浓度在这个单站数据下的**可预测上限**在哪？我们接近了吗？
- 核心问题：当前增量模型 CRPS≈0.86（相对持久化 +21~24%）。世界观 A（接近上限）/ B（没接近）/ C（中间态）三选一。
- 方法：给模型输入"未来 N 天真实 conc"（oracle，诊断探针，正常推理不可能有），看 CRPS 能降到多少 = 数据在当前特征集下的信息上限。
- **关键防退化设计**：探针视界 H=64（8 天）而非 H=8——若 H=8（1 天），oracle N≥1 天完整覆盖目标窗 → 全档 CRPS≈0 退化。H=64 让 oracle 只覆盖前 N 天，尾部是"诚实"预报区，N=1/3/7 有可区分读数，且能测"轨迹信息随视界的衰减"。
- 产出：`run_m.py`、`probe_h64.py`（horizon 稀释 + oracle 收敛诊断）、`rethinking.md`、`trydoing.jsonl`

### [2026-08-10 15:45] 本地冒烟
- `--smoke`（1 窗口 × 1 seed × 2 epoch CPU）全链路通过
- 修复：OracleRamsNet 缺 predict_mean/predict_interval（Trainer 验证调用）→ 补上与 RamsNet 一致接口

### [2026-08-10 15:50] 设计验证（窗口 1，30 epoch，CPU）
- **horizon 稀释诊断**（probe_h64.py）：base8 H=8 CRPS=2.44 vs base64 H=64 第1天 3.13（稀释 1.28×）——H=64 共享骨干会稀释短视界精度，但所有 oracle 臂同 H=64 对称可比，且绝对锚定由 base8 臂提供，设计成立
- **oracle 收敛诊断**：oracle_7 学会利用未来轨迹（CRPS_all 3.72→2.47，尾部 d8 4.13→2.73）
- **头注入改进**：把 oracle 通道末时刻值直接拼到 M1 头输入（hidden ⊕ oracle）→ 已知段"复制"近线性化，探针用得更充分
- **有效指标**：已知天标 0（完美复制=可达的信息上限），诚实尾用模型读数——避免"模型复制不完美"污染信息上限估计

### [2026-08-10 16:00] 全量启动（v1 未掩蔽版）
- 6 arm × 3 seed × 17 窗口 × 30 epoch = 306 次训练
- 算力机 sensecore H100（80GB 空闲）
- 命令：`nohup python3 exp/model_enhancement/m_oracle_bound/run_m.py --device cuda > run_full.log 2>&1 &`
- 44.0 分钟完成

### [2026-08-10 16:50] v1 发现探针缺陷 → v2 掩蔽版
- v1 用未掩蔽损失 → oracle 臂未充分复制已知轨迹（oracle_1 已知第1天 raw CRPS=0.81 应≈0，
  oracle_7 已知段 0.94）：复制任务在 loss 占主导稀释尾部训练 → 探针低估信息上限（假世界A偏置）
- **v2 修复**：oracle 臂 M1 分位数损失只算诚实尾（h≥M）；有效按天 CRPS（已知天标 0，
  诚实尾用模型）+ known_crps 通道利用诊断。本地 window1×30epoch 验证后重启全量

### [2026-08-10 17:00] 全量完成（v2 掩蔽版，31.8 分钟）+ 结论
- **协议可信**：base8 精确复现 A 方向 multi arm（CRPS 0.8912 / persist 1.1281 / +21%）
- **世界观判定：World A/C（接近信息上限）**
  - oracle 阶梯：known0 0.949 → known1 1.132 → known3 1.111 → known7 1.115（多知道真实轨迹
    不降 1 天前 CRPS，反而略升）
  - 纯自回归最优：ar_only 第1天 0.872 < base 0.949（temp/气象 −8.8%）；与 E/D/J/M5 互洽
  - 长视界 oracle "提升"（d4 +20.8%、d8 +32.1%）是距离效应，不可桥接、无部署价值
  - World C 微弱（corr(gap, 范围)=+0.75 但仅暴发 w1 gap=+0.265，16/17 窗口 base 更好）
- **结论**：可预测上限 ≈ 当前水平（oracle 底 1.11 vs 当前模型 0.95，模型已更好），
  继续榨模型收益递减；再压误差走数据量/多站方向
- 产出：`results.md`（完整报告）、rethinking.md / trydoing.jsonl 回填结论
