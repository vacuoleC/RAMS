# B_direction_prob · 探索标记（whatwedo.md）

- **标记时间**：2026-08-10 13:05（东八区）
- **任务**：B_direction_prob：P(Δ>0) 方向概率判别力实证（exp/model_enhancement/b_direction_prob/）
- **背景**：B2/B7 两轮探索都证 p50 方向判别弱（sign 命中 0.53≈随机，平凡基线 0.49-0.51），B7 排除目标尺度因素后判断方向判别弱是 Δ 本身难判别；两轮建议另配 P(Δ>0) 概率输出 + 阈值校准。本实验直接检验该建议
- **探索依据**：exp/EXPLORATION_POLICY.md（文件级探索标记）；协议同 B7（滚动窗口 820d / 步长 45d，测试段与 B7 一致）
- **方法**：GRU 骨干 + M1 p10/p50/p90 + M2 + M4 + **方向头（BCE 联合训练）**；P(Δ>0) 5 来源（p50_sign / gauss / linear / pchip / binary）对照；阈值校准（校准段扫 τ）+ Isotonic + ECE + 校准曲线；3 seed × 17 窗口
- **关键理论**：凡 P 是 q50 的单调函数（gauss/pchip/linear），其 ROC-AUC 与 acc@0.5 与 p50 符号恒等（单调变换保序）——只有二分类头可能真正改变判别力；AUC 是阈值无关的诚实指标
- **产物**：
  - `run_direction_prob.py`（实验脚本，本地+远程冒烟通过）
  - `launch.py`（GPU 等待器：显存≥15GB 启动全量，不抢占主作业）
  - `trydoing.jsonl`（执行日志）
  - `rethinking.md`（思考记录）
  - `results.md` / `results.json`（结论，待全量跑完）
- **合规**：只输出统计量/覆盖率/CRPS/AUC/Brier/ECE，不打印原始数据行；不触碰 rams/ 冻结代码（仅 import 复用 + 实验内复用 M1Head/M2Head/M4Head/SharedGRU 组件）
- **状态**：✅ 探索完成（2026-08-10 14:05，H100 全量 17 窗口 × 3 seed × 30 epoch，36.9 分钟）
- **结论摘要**：P(Δ>0) **弱判别**（AUC≈0.59 全部方法，窗口 std≈0.03），不是 B2/B7 sign 口径看到的"≈随机 0.53"——sign-acc 低估了连续 q50 的方向信息；但 P(Δ>0) **硬 yes/no 分类不可用**（sign-acc 0.539 仅 +1.1pp 基线、Brier≈常量无技能、校准曲线扁平、中间段压缩）。**二分类头无增量**（binary AUC 0.594 vs gauss 0.591，+0.003）；阈值校准无收益（τ 最优≈0.5）、Isotonic 反而恶化。建议：方向提示 = 分位数反推 gauss + 高置信阈值（P>0.7 才提示）+ 软信号定位，不值得加专用方向头。详见 results.md。
- **边界决策**：方向预警定位为"软信号"（置信加权修饰区间预警），不宣称硬方向命中；正式评估方向用 ROC-AUC 而非 sign-acc——待主会话审核
