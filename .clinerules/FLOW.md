# RAMS 项目 · 流程与进度（FLOW）

> **这是一个活文件。** Cline 每次任务开始前读它（确认当前步骤和验收标准），每次任务结束后更新它（改对应步骤状态 + 更新"当前位置"）。
> 状态图例：⬜ 待办　🔄 进行中　✅ 完成　⏸️ 阻塞（在 LOG.md 写明原因）
> 推进原则：一次一个任务；一个阶段全部 ✅ 后**停下等人确认**再进下一阶段；不跳阶段。
> 规则依据：`.clinerules/00-核心规则.md`（总则）、`.clinerules/01-数据规范.md`（数据契约）。

---

## 📍 当前位置

- **阶段**：Stage 0 启动准备
- **步骤**：任务 0.1 初始化项目骨架（⬜ 待办）
- **下一步**：完成 0.1 后汇报，等确认再做 0.2

---

## 进度总览

| 阶段 | 状态 | 出口验收 |
|---|---|---|
| Stage 0 启动准备 | ⬜ | `import rams` 跑通、硬件配置可读、本地 Git 就绪 |
| Stage 1 数据理解 | ⬜ | 标准数据集产出 + 字段契约锁定 |
| Stage 2 工程规划 | ⬜ | `python -m rams --help` 跑通、骨架可 import |
| Stage 3 模型实现 | ⬜ | 全链路 `fast_dev_run` 跑通 |
| Stage 4 验证分析 | ⬜ | 统计显著性 + 可解释性脚本就绪 |
| Stage 5 封装测试 | ⬜ | 测试 ≥80% + wheel 就绪 |
| Stage 6 报告撰写 | ⬜ | 报告骨架 + 图表脚本 + Demo |
| Stage 7 部署收尾 | ⬜ | ONNX INT8 + 性能基准 |

---

## Stage 0 · 启动准备
**目标**：搭骨架和配置，让 `import rams` 跑通。**不写任何模型逻辑。**

- ⬜ **0.1 初始化项目骨架**：建目录树（见 00 第 3 节）+ 各 `__init__.py` + `environment.yml`/`pyproject.toml`/`.gitignore`/`README.md`。`.gitignore` 含 `data/ *.xls* *.parquet __pycache__/ *.pt *.ckpt mlruns/ .venv/ outputs/`。
  - 验收：`python -c "import rams"` 不报错，目录结构完整
- ⬜ **0.2 硬件配置**：`configs/hardware/` 建三个 yaml（内容见文末附录）。
  - 验收：切换 `hardware=local_4060` / `rental_a100` 能正确读出对应 `device`/`batch_size`
- ⬜ **0.3 初始化 Git**：`git init`（不配远程），确认 `git status` 看不到 `data/`，首次提交 `chore: 初始化项目骨架`。
  - 验收：`git log` 有一条提交，`data/` 不被追踪

> 三个都 ✅ 后停下，等确认再进 Stage 1。

---

## Stage 1 · 数据理解
**目标**：一次性清洗 xls → 产出**标准数据集**，锁定字段契约。**严格按 `01-数据规范.md` 的契约执行。**

- ⬜ **1.1 确认原始文件**：`Get-ChildItem D:\coding\26AIendwork\data\raw`，记录真实文件名（不硬编码）。
- ⬜ **1.2 写清洗脚本 `scripts/build_dataset.py`**（唯一读 xls 的地方）：按 `01-数据规范.md` 第 4 节流程——20 sheet 合并（保留 depth）+ 气象 10min 最近时刻 join + 透光率=100 标记/插值 + 4 字段物理推算补齐 + 中文列名转英文 + 类型固化 + Pandera 校验。冒烟：只跑采样的少量行。
- ⬜ **1.3 产出标准数据集**：人工跑一次 `build_dataset.py` 生成 `data/processed/standard.parquet` 与 `norm_stats.json`（Cline 写好脚本 + 冒烟通过即停，由人跑全量）。0.5m 层结果须与匹配表逐行一致。
- ⬜ **1.4 字段契约**：`configs/data/schema.yaml`（中英映射/类型/范围/单位/source 标签）+ Pandera 校验脚本。
- ⬜ **1.5 数据画像**：只打印形状/列名/统计量（**不打印原始行**），可输出 notebook；确认深层透光率=100 占比。
  - 验收：标准数据集产出、Pandera 校验通过、画像无原始数值泄露

---

## Stage 2 · 工程规划
**目标**：骨架接口 + CLI + 实验配置就位（先不写实现）。

- ⬜ **2.1 接口签名占位**：各模块函数签名 + docstring 写清 I/O 形状。
- ⬜ **2.2 CLI 入口**：`scripts/` 下 `build_dataset/train/eval/predict/export/serve`，`python -m rams --help` 能跑。
- ⬜ **2.3 实验配置**：每模块 1 个 P0 实验的 Hydra yaml。
  - 验收：`python -m rams --help` 跑通、骨架可 import

---

## Stage 3 · 模型实现（最重，逐个冒烟）
**目标**：数据管线 → Backbone → 五头 → 训练编排，每个单独冒烟。

- ⬜ **3.1 数据管线**：`preprocessing.py` + `tensor_builder.py`（读 standard.parquet，按 (timestamp,depth) 透视成 `(B,T,D,C)`，风向 sin/cos 编码、加载时归一化）+ `datamodule.py`（时间 70/15/15 切分）。
- ⬜ **3.2 Backbone**：`ChannelEmbedding`→`TemporalBlock`→`VerticalBlock`→`FusionBlock`，**打印验证参数量 ≈1.9M**。
- ⬜ **3.3 五个头**：M1/M2/M3(GAT+贪心)/M4/M5。
- ⬜ **3.4 训练编排**：`trainer.py` 两阶段（联合预训练→冻结微调），**必带 `fast_dev_run`**，接本地 MLflow。
  - 每个子任务终点是冒烟通过 → 停 → 由人跑训练
  - 验收：全链路 `fast_dev_run` 跑通

---

## Stage 4 · 验证分析
- ⬜ 统计检验脚本（3 seed × Wilcoxon）⬜ SHAP/Captum ⬜ PCMCI+ ⬜ 共形预测（MAPIE）
- 多在人跑完训练后做；Cline 负责写脚本 + 用假结果/小样本冒烟。

## Stage 5 · 封装测试
- ⬜ 类型注解/docstring 补全 ⬜ `pytest` ≥80% ⬜ CLI 完整 ⬜ 打离线 wheel

## Stage 6 · 报告撰写
- ⬜ 图表脚本（英文标注、脱敏）⬜ Streamlit 演示 ⬜ 报告骨架

## Stage 7 · 部署收尾
- ⬜ ONNX 导出 ⬜ INT8 量化(<3MB) ⬜ FastAPI ⬜ Docker ⬜ 性能基准表

---

## 附录：`configs/hardware/` 初始内容（0.2 照此创建）

```yaml
# local_4060.yaml —— 本地 4060 8G，仅冒烟/小规模试跑
device: cuda
precision: 16-mixed
batch_size: 8
num_workers: 0
flash_attn: false
accumulate_grad_batches: 4
```
```yaml
# rental_a100.yaml —— 租用 A100，跑完整实验
device: cuda
precision: bf16-mixed
batch_size: 64
num_workers: 8
flash_attn: true
accumulate_grad_batches: 1
```
```yaml
# h100.yaml —— H100 恢复后用
device: cuda
precision: bf16-mixed
batch_size: 128
num_workers: 8
flash_attn: true
accumulate_grad_batches: 1
```

---

## 如何更新本文件（给 Cline）
- 开始某步：⬜→🔄，更新顶部"当前位置"。
- 完成某步：→✅；若是阶段最后一项，更新"进度总览"该阶段状态。
- 受阻：→⏸️，在 `LOG.md` 写清原因。
- 只改状态标记和"当前位置"，**不要删改步骤描述和验收标准**（要改流程先问人）。
