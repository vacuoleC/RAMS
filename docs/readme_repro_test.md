# README 复现验证：干净环境冒烟（H100）

- 日期：2026-08-11
- 目标：在算力机 sensecore（NVIDIA H100 80GB）的**干净目录**里，严格按 README Quick Start 命令逐步执行，验证能否复现（冒烟级跑通）。
- 干净目录：`/data/RAMS/readme_test/`
- 结论：**三步全部跑通**。无 README 未提到的隐藏依赖。发现 2 个 README 需修正处（`python` → `python3`；Quick Start 缺 `pip install -e .` 与 Python 版本约束说明）。

---

## 1. 环境准备

| 项 | 值 |
|---|---|
| 机器 | sensecore（ssh），Ubuntu，NVIDIA H100 80GB HBM3 |
| Python | `/usr/bin/python3` = **Python 3.8.10**（系统默认，**无 `python` 命令**） |
| PyTorch | torch 2.3.1+cu121，`cuda_available=True`，驱动 580.95.05 |
| pandas / numpy / pyarrow | 2.0.3 / 1.24.4 / 17.0.0 |
| openpyxl / scikit-learn / numba / pytest | 3.1.5 / 1.3.2 / 0.58.1 / 8.3.5 |

- 已安装且本流程**需要**的核心包齐全：numpy、pandas、pyarrow、openpyxl、sklearn、numba、torch 2.3.1+cu121。
- 未安装但本流程**不需要**：polars、pandera、lightning、mlflow、optuna、shap、mapie、onnx、scienceplots、cmocean、torch-geometric（torch_geometric 2.6.1 已装）。
- 步骤：`mkdir -p /data/RAMS/readme_test/data/raw`；把本地仓库 `rams/ scripts/ tests/ pyproject.toml environment.yml` scp 到干净目录（不拷 data/.git/治理文件）；把 `/data/RAMS/*.xlsx` 三个原始 xlsx 拷入 `data/raw/`（md5 与源一致）。代码文件 md5 与本地逐字节一致。

## 2. 每步执行结果

### (a) build_dataset（README：`python scripts/build_dataset.py --raw data/raw --out data/processed`）

执行（用 `python3`）：`python3 scripts/build_dataset.py --raw data/raw --out data/processed`

- 结果：**成功**，耗时 57s。
- 输出：`data/processed/standard.parquet`（(258542, 23)）+ `data/processed/norm_stats.json`（20 列）。
- 藻类观测 258,542 行，与 README「258,542 条藻类观测」完全一致。
- 命令/参数与 README 一致，默认 `--train-frac 0.7` 无需改。

### (b) train_v020 --fast-dev（README：`python scripts/train_v020.py --fast-dev`）

执行（用 `python3`）：`python3 scripts/train_v020.py --fast-dev`

- 结果：**成功**，耗时 48s，`device=cuda`。
- 输出：3 seed × 17 窗口全部跑完，逐窗口/逐视界/总体统计打印完整（无 `--out` 时只打印，不落盘）。
- 冒烟口径：`--fast-dev` 只把每次 fit 压到 2 epoch × 前 2 batch；**外层仍是 17 窗口 × 3 seed**（脚本默认）。约 48s。
- 输出形状：每窗口 n≈762，n_tr≈675，测试段 M1/M2/M4 评估正常。

### (c) eval_m4_warning --smoke（README：`python scripts/eval_m4_warning.py --smoke`）

执行（用 `python3`）：`python3 scripts/eval_m4_warning.py --smoke`

- 结果：**成功**，耗时 20s，`device=cuda`。
- 输出：日级表 1638 行（2021-03-01 → 2025-09-30）、BloomLabeler 5 个日级事件、阈值扫描（θ=0.3/0.5/0.6/0.7/0.8）、与 N 12 事件对齐明细、基线对照；写 `exp/mdl_m4_warning/results.json`（19 KB）。
- 冒烟口径：`--smoke` = 1 seed × 17 窗口 + fast_dev_run；评估期 [2023-03-01, 2025-05-19] 共 795 天，概率覆盖 787 天。
- 提示：冒烟模型仅 2 epoch，θ=0.5 召回 0.4（2/5）、命中事件提前量 19.5 天，数值不代表正式结果（README 未注明——见问题 4）。

## 3. README 复现是否可行

**可行（冒烟级）**。在干净目录、系统 Python 3.8 + 已装 torch 2.3.1+cu121 的前提下，README Quick Start 三条命令按序全部跑通：

```
python3 scripts/build_dataset.py --raw data/raw --out data/processed   # 57s 成功
python3 scripts/train_v020.py --fast-dev                               # 48s 成功（17窗×3seed，cuda）
python3 scripts/eval_m4_warning.py --smoke                             # 20s 成功（cuda）
```

前提：`data/raw/` 已有 3 个原始 xlsx；环境有 pandas/pyarrow/openpyxl + torch cu121。这些均属 README「安装」+「数据」两节的隐含前置，无隐藏依赖。

## 4. 发现的问题及 README 需修正处

1. **命令 `python` 在 Ubuntu 不存在**（只有 `python3`）。算力机 `which python` 为空；README Quick Start 三处 `python` 命令在 Linux 上直接执行会报 `command not found`。README 应改为 `python3`（或在安装节注明 Linux 下用 `python3`）。
2. **安装节 `pip install -e .` 在此机器不可行**：
   - `pyproject.toml` 写 `requires-python = ">=3.10,<3.12"`，系统 Python 3.8.10 不满足。
   - 系统 pip 20.0.2 过旧，PEP 660 可编辑安装无 `setup.py` 时报错（`File "setup.py" not found`）。
   - 但三条 Quick Start 脚本都 `sys.path.insert` 仓库根目录，**不依赖** `pip install -e .`；不装也能跑。README 的安装步骤与 Quick Start 实际要求脱节，需说明「editable 安装仅本地 dev 用，算力机可直接跑脚本」。
3. **Python 版本声明与实际不符**：README badge 写「Python 3.10+」、`environment.yml` 要求 3.10、`pyproject` 要求 `>=3.10,<3.12`；而算力机系统 Python 3.8 实测可跑通全部冒烟（代码用了 `from __future__ import annotations`，`str|Path` 注解在 3.8 安全；无 3.9+ 运行时语法）。README 的 Python 版本要求比实际必要更严，建议注明「3.8–3.11 均可」或把 pyproject 下限放宽到 3.8。
4. **冒烟命令未注明「不等于正式指标」**：`--fast-dev`/`--smoke` 只跑 2 epoch × 2 batch，外层仍是 17 窗口 × 3 seed（train）或 17 窗口（eval）。README 标注 `# smoke test` 但未说明输出数值不代表 README 顶部「关键成果」表（如 CRPS +22.1%、覆盖 0.766、提前量 14.5 天）——冒烟结果明显偏低（覆盖 ~0.16、θ=0.5 召回 0.4）。建议加一句「冒烟结果仅供链路验证，数值不代表正式结论」。

## 5. 结论

- README Quick Start 三条命令在算力机干净环境下**全部冒烟级跑通**，数据规模、窗口数、评估期与 README 描述一致。
- 复现无需 README 之外额外安装的依赖。
- README 需修正 2 处事实性问题（`python`→`python3`；`pip install -e .` 与 Python 3.10 约束与算力机 3.8 实际不符）和 2 处说明性改进（Python 版本下限可放宽；冒烟数值不代表性）。
