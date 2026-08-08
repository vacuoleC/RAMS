# RAMS 项目 · 执行日志（LOG）

> **这是一个活文件。** Cline 每次任务开始前读它（看已有产物、避免重复踩坑），每次任务结束后**追加**一条记录。
> 记录重点：终端结果摘要、遇到的失误及修复、新产出的交付物。
> **保密**：禁止把原始数据数值写进本文件——只写形状、列名、统计量、报错信息。

---

## 一、阶段交付产物清单

> 每产出一个交付物就在这里登记一行。让任何时候都能一眼看清"现在手上有什么"。

| 阶段 | 交付物 | 路径 | 状态 | 备注 |
|---|---|---|---|---|
| 0 | 项目目录树 | `configs/  data/raw  data/interim  data/processed  rams/{data,models,training,inference,interpret,eval}  scripts  notebooks  tests  docs` | ✅ | 0.1 |
| 0 | `rams` Python 包 + 6 个子包 `__init__.py` | `rams/__init__.py`（version=0.1.0） + `rams/{data,models,training,inference,interpret,eval}/__init__.py` + `tests/__init__.py` | ✅ | 0.1 |
| 0 | `environment.yml` | `./environment.yml` | ✅ | 0.1，含 conda 依赖 + pip 块 |
| 0 | `pyproject.toml` | `./pyproject.toml` | ✅ | 0.1，含 ruff / black / mypy / pytest / coverage 配置 |
| 0 | `.gitignore` | `./.gitignore` | ✅ | 0.1，含 `data/ *.xls* *.parquet __pycache__/ *.pt *.ckpt mlruns/ .venv/ outputs/` 等 |
| 0 | `FLOW.md` / `LOG.md` 活文件副本 | `./FLOW.md`、`./LOG.md` | ✅ | 0.1，由 `.clinerules/` 复制到项目根；后续只改根版本 |
| 0 | `README.md` | `./README.md` | ✅ | 0.1（已在之前产出） |
| 0 | Git 初始化（首次提交 `3a169fe`） | `./.git/` | ✅ | 0.3，`data/` 不被追踪，保密满足 |
| 1 | 数据预处理模块 | `rams/data/preprocessing.py` | ✅ | 探索推进，真实数据跑通，产出 standard.parquet（258,542×23） |
| 1 | 张量构建模块 | `rams/data/tensor_builder.py` | ✅ | 探索推进，长表→(B,T,D,C) + 分层标签，冒烟通过 |
| 1 | 数据画像脚本 | `scripts/explore/profile_data.py` | ✅ | 探索推进，无原始数值泄露 |
| 3 | 模型（共享 GRU + M1/M2 双头 + 分位数） | `rams/models/rams_net.py` | ✅ | 探索推进，27,674 参数 |
| 3 | 训练编排 | `rams/training/trainer.py` | ✅ | 探索推进，30ep RMSE=3.58/acc=0.965/覆盖85% |
| 3 | CLI 入口 | `scripts/build_dataset.py`、`scripts/train.py` | ✅ | 探索推进 |
| 3 | 冒烟测试 | `tests/test_smoke.py` | ✅ | 6/6 通过 |
| 3 | 探索测试脚本 | `scripts/explore/`（p1-p4,m4,prep,profile） | ✅ | H100 实证 |
| 6 | 架构蓝图 | `docs/architecture_blueprint.md` | ✅ | 2026-08-09 |
| 6 | 文献综述 | `docs/literature_review.md` | ✅ | 2026-08-09 |
| 6 | 站位计划 | `docs/standing_plan.md` | ✅ | 2026-08-09 |
| | | | | |

---

## 二、失误与修复记录

> 踩到的坑都记这里：报错信息 + 根因 + 怎么修的。下次开工前先扫一遍，别重复犯。
> 这一节将来直接复用进报告的"工程问题排查"小节。

| 日期 | 阶段/步骤 | 现象（报错摘要） | 根因 | 修复方法 |
|---|---|---|---|---|
| 2026-06-08 | 0.1 移动 xlsx | `Move-Item : δҵ·еĳ֡` | (1) `data/raw` 目录尚未存在；(2) PowerShell 默认输出编码 GBK，文件名为中文时控制台显示乱码（实际是编码问题不是文件问题） | 先执行 `New-Item -ItemType Directory data\raw` 创好目录；再重跑 `Move-Item`；乱码问题仅影响控制台显示，不影响脚本逻辑。 |
| 2026-06-08 | 0.1 补充 README 更新 | 对 README.md 三次**并行** `replace_in_file` 全部失败，工具提示 `The file was reverted to its original state: <empty>`，实际 `README.md` 被清空丢失 | 在一次响应中**并行发起多个对同一文件的 replace_in_file** 会触发工具的“回滚到原状态”机制，第一个失败时另外两个也会被撤，最终文件被置空。 | （1）以后修改同一个文件**串行**做，且一个响应里只发一个 `replace_in_file`；（2）遇失败时优先 `read_file` 拿真实内容再重试；（3）实在不行用 `write_to_file` 整体重写——但要谨慎，会丢失原内容。最终用户要求“更新一下”时直接走 `write_to_file` 重建 + 一次性嵌入更新。 |
| | | | | |

---

## 三、执行日志（按时间追加，最新在最上）

> 每完成一步追加一条。模板见文末。

<!-- 在这一行下方追加新记录 -->

### [2026-08-09 04:05] Stage 0-3 探索推进 —— 算力机实证 + 核心代码 + Git 初始化
- 目标：按新需求（真实项目/五任务全做/算力充足/追求最大精度/不追迁移）在 sensecore H100 上探索性推进，验证架构方向并落地核心代码
- 前置：用户明确了项目新定位，确定了算力环境（sensecore 商汤 H100，可扩容）；数据上传至 `/data/RAMS/`
- 探索测试（H100 实证，3-seed）：
  - P1 数据管线：发现**各深度层时间戳错位 4-5 分钟（精确交集=0）**，floor 到 3h 网格后 97.8% 对齐 → 清洗脚本用 merge_asof
  - P2 时序基线：GRU 预测 24h RMSE=4.26 vs 持久化 12.96（好 67%）
  - P3 垂直信息：**单层 6.51 → 20 层 4.22（降 35%）**，证实垂直分层是核心资产
  - P4 loss 加权：时序任务下**归一化 > 未归一化**（修正第一轮当期回归的假象）；M2 分类优先（w2=3.0）最优
  - P5 分位数：覆盖率 80.9%，RMSE 不损失
- 正式代码（写入 `rams/` 包）：
  - `rams/data/preprocessing.py`：xlsx → standard.parquet（258,542×23），真实数据跑通
  - `rams/data/tensor_builder.py`：长表 → (B,T,D,C) + 分层标签
  - `rams/models/rams_net.py`：共享 GRU backbone + M1 分位数头 + M2 分层头
  - `rams/training/trainer.py`：多任务 loss 加权 + 分位数损失 + fast_dev_run
  - `scripts/build_dataset.py` / `scripts/train.py`：CLI 入口
  - `tests/test_smoke.py`：6 项冒烟测试全部通过
- 正式训练结果：30 epoch → M1 RMSE=3.58、M2 acc=0.965、p10-p90 覆盖率 85%
- 文档：`docs/architecture_blueprint.md`（架构决策实证）、`docs/literature_review.md`（文献综述）、`docs/standing_plan.md`（站位计划）
- Git：0.3 初始化完成（提交 `3a169fe`，`data/` 不被追踪 ✅ 保密满足）
- 失误：探索脚本多个路径硬编码（Windows 绝对路径在算力机失效）→ 改相对/参数化；sheet 自带 depth 列冲突 → 转名后删除；pivot 后 258k 行（各层时间戳独立）→ 先 groupby(ts3h, depth) 聚合。均已在正式代码修复。
- 冒烟：✅ 通过（6/6 测试 + 真实数据 fast_dev_run + 30 epoch 训练）
- 交付物：核心代码 4 模块 + 2 CLI + 测试 + 3 文档 + Git 首次提交
- 状态：✅ 完成
- 下一步：实现 M3（GAT+贪心点位优化）/M4（预警分级）/M5（PCMCI+）；或按新需求修订 00-核心规则 架构约束（参数量、Stage 7 INT8 可放宽）

---

### [2026-06-08 11:50] Stage 0 · 任务 0.1 补充 —— 更新 README
- 目标：用户要求「readme  请你更新一下」，将 README 反映项目当前实际状态（0.1 已就位）
- 改动文件：仅 `README.md`
- 嵌入的三处更新：
  1. **第 2 节**：在“规则与活文件”表格后，加了一张“项目根工程文件清单”表，列出 `environment.yml` / `pyproject.toml` / `.gitignore` / `configs/` / `rams/` / `scripts/` / `tests/` / `data/{raw,interim,processed}/` 等
  2. **第 6 节**：表格上方加了一个 **“📍 当前进度（2026-06-08）：Stage 0 · 任务 0.1 ✅”** 小框
  3. **附录**：加了一张“项目根脚手架（0.1 已就位）”示意图，反映实际创建的目录与文件
- 失误：中间发生过一次**严重踩坑**——见第二节表格（一条新登）。原 README 在并行 replace 失败中被误清空。重建后内容 466 行，比原 399 行多 67 行（增量是 3 项更新 + 几个补充说明）。
- 冒烟：不涉及（文档更新）
- 交付物：README.md（466 行，含 3 项更新）
- 状态：✅ 完成
- 下一步：**停下等人确认**。可以开始 0.2 硬件配置，或按其他指示走。

---

### [2026-06-08 11:35] Stage 0 · 任务 0.1 —— 初始化项目骨架
- 目标：建项目骨架（目录树 + `__init__.py` + 配置文件 + 活文件），验证 `import rams` 跑通
- 改动文件：
  - 新建：`rams/__init__.py` 及 6 个子包 `__init__.py`（`rams/{data,models,training,inference,interpret,eval}/__init__.py`）、`tests/__init__.py`
  - 新建：`environment.yml`、`pyproject.toml`、`.gitignore`
  - 新建：项目根 `FLOW.md`、`LOG.md`（由 `.clinerules/` 复制，后续只改根版本）
  - 新建目录：`configs/{hardware,data,model,train,experiment}/`、`rams/{data,models,training,inference,interpret,eval}/`、`scripts/`、`notebooks/`、`tests/`、`data/{raw,interim,processed}/`、`docs/`
  - 移动：3 个 xlsx 从 `data/` 根目录移到 `data/raw/`（按 01-数据规范.md 落地）
- 执行命令与结果：
  - `New-Item -ItemType Directory ...`：全部 18 个目录创建成功
  - `Move-Item ... data\*.xlsx → data\raw\`：先失败（`data/raw` 还不存在），创好目录后重试成功
  - `python -c "import rams; print('rams import OK, version =', rams.__version__)"`：**输出 `rams import OK, version = 0.1.0`**
- 失误：见第二节表格（1 条）—— 移动 xlsx 第一次失败，根因是 `data/raw` 目录尚未创建；PowerShell 中文文件名乱码仅是控制台编码显示问题，不影响实际逻辑
- 冒烟：**通过**（不报错、`__version__` 正确读取、空包依赖最小）
- 交付物：见第一节表格（7 行新登记）
- 状态：✅ 完成
- 下一步：**停下等人确认**。确认后开始 0.2 硬件配置（建 `configs/hardware/{local_4060,rental_a100,h100}.yaml`，验证切换 `hardware=local_4060` 等能读出正确配置）。仍**不要**自动跑 0.2。

---

### 记录模板（复制这一段填写）
```
### [YYYY-MM-DD HH:MM] Stage X · 任务 X.Y —— 一句话标题
- 目标：本次要做什么
- 改动文件：列出新建/修改的文件
- 执行命令与结果：
  - `命令`：结果摘要（成功/失败、关键输出，不含原始数据数值）
- 失误：遇到的问题（同时登记到第二节表格），没有则写"无"
- 冒烟：是否通过（不报错/loss 有限/形状对），或本步不涉及
- 交付物：本次新产出（同时登记到第一节表格），没有则写"无"
- 状态：✅ 完成 / 🔄 进行中 / ⏸️ 阻塞（写原因）
- 下一步：建议（不自动执行）
```

---

## 如何使用本文件（给 Cline）
- **任务开始前**：读第一节（已有产物，避免重做）+ 第二节（已知坑，避免重犯）。
- **任务结束后**：① 在第三节追加一条记录；② 有新产出登记到第一节；③ 有踩坑登记到第二节。
- 报错信息照实记录（这是最有价值的部分），但**不要粘贴含真实数值的数据行**。
