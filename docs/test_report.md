# mdl-test-package 模块结果（RAMS 0.2.0 · 测试与打包）

> 模块：`modules/mdl-test-package/module_design.yaml`（UID fa5dc76f-551c-453a-ac9b-a72863e6039b）
> 日期：2026-08-10 ｜ 全链路 pytest 通过 + 打包验证通过，交主会话评审
> 数据保密：全部测试用合成 / 聚合统计量，不打印任何原始数据行。

## 1. 做了什么

RAMS 0.2.0 落地收尾模块：全链路 pytest、补测试缺口、打包验证、冒烟记录。

- **跑全链路 pytest**：5 个测试文件 53 项全部通过，覆盖率 83%（原 31 项 / 71%）。
- **补测试缺口**：
  - `tests/test_eval_m4_warning.py`（新增，12 项）：M4 预警评估纯函数
    （`events_span` / `warning_episodes` / `evaluate_threshold` 的召回 / 提前量 / 误报逻辑）。
    此前 `scripts/eval_m4_warning.py` 的评估逻辑完全无测试。
  - `tests/test_preprocessing.py`（新增，10 项）：唯一原始数据接触点
    `rams/data/preprocessing.py`（xlsx → parquet，load_algae / load_meteo / align_and_clean /
    compute_norm_stats / build_dataset），全部用合成 xlsx，覆盖率 0% → 85%。
- **修一个真 bug（脚本被 import 时劫持 stdout）**：`scripts/eval_m4_warning.py` /
  `scripts/train_v020.py` / `rams/training/smoke_roll.py` 顶层
  `sys.stdout = io.TextIOWrapper(sys.stdout.buffer)` 会在被 pytest import 时包住并关闭
  capture 缓冲，导致整站测试 teardown 崩溃（`I/O operation on closed file`）。
  改为 `if __name__ == "__main__" and hasattr(sys.stdout, "buffer"):` 守卫，
  仅脚本运行时启用 UTF-8 控制台输出。修复后脚本 / 测试双路径均正常。
- **打包验证**：`python -m build` 成功产出 wheel + sdist，wheel 装入独立 target 后
  `import rams` 全链路 + 前向 + 两阶段训练均通过。
- **冒烟记录**：见第 4 节。

## 2. pytest 结果（全链路）

运行：`python -m pytest`（pyproject.toml 已含 `--cov=rams --cov-report=term-missing`）。

| 测试文件 | 覆盖 | 结果 |
|---|---|---|
| `tests/test_smoke.py` | 旧冒烟（RamsNet 形状 / TensorBuilder / Trainer fast_dev_run） | 10/10 通过 |
| `tests/test_daily_pipeline.py` | 日级管线（DailyTensorBuilder / BloomLabeler / 滚动窗口） | 12/12 通过 |
| `tests/test_model_integrate.py` | 模型整合（q9 / 损失 / 两阶段 / CRPS / M4 标签） | 9/9 通过 |
| `tests/test_eval_m4_warning.py` | **新增**：M4 预警评估纯函数 | 12/12 通过 |
| `tests/test_preprocessing.py` | **新增**：预处理（合成 xlsx 全链路） | 10/10 通过 |
| **合计** | | **53 passed** |

### 覆盖率（分支模式）

```
TOTAL  787 stmts  115 miss  174 branch  46 brpart  83%
```

| 文件 | 覆盖率 |
|---|---|
| `rams/data/preprocessing.py` | 85%（新增测试后 0% → 85%） |
| `rams/data/tensor_builder.py` | 89% |
| `rams/models/rams_net.py` | 95% |
| `rams/training/trainer.py` | 91% |
| `rams/data/models/training/__init__.py` | 100% |
| `rams/eval` / `rams/inference` / `rams/interpret` | 0%（空占位子包，仅 docstring，无代码） |
| `rams/training/smoke_roll.py` | 0%（独立冒烟脚本，非 import 目标；作为脚本实测通过） |

核心模块（数据 / 模型 / 训练 / 预处理）均 ≥85%；余下 0% 文件为空子包占位或独立冒烟脚本。

## 3. 打包结果

- 环境：系统 Python 3.14.6（有 pytest 9.1.1 / torch 2.13.0+cpu / numpy 2.5.2 / pandas 3.0.5 / pyarrow / openpyxl）。
- `python -m build --outdir dist` → **成功**：
  - `dist/rams-0.1.0-py3-none-any.whl`（46 KB）
  - `dist/rams-0.1.0.tar.gz`（64 KB）
  - 包内容：`rams/data`、`rams/models`、`rams/training`、`rams/eval`、`rams/inference`、
    `rams/interpret` + `scripts` 入口点（`rams-*` CLI）。
- wheel 安装验证：`pip install --no-deps --ignore-requires-python --target /tmp/rams_install dist/*.whl`
  → `import rams` 全链路 + `RamsNet` 前向（(2,27)/(2,2)/(2,4)）+ `Trainer.fit_two_stage`
  fast_dev_run（Stage1/Stage2 loss 有限）全部通过。
- **环境约束提示（交主会话）**：`pyproject.toml` 声明 `requires-python = ">=3.10,<3.12"`，
  而本机系统 Python 为 3.14 → `pip install` 默认拒绝（需 `--ignore-requires-python`，仅因
  wheel 是 `py3-none-any` 纯 Python）。算力机 conda 若为 3.11 则不受影响；如需在本机
  `pip install -e .` 或放开打包环境，需评估放宽到 `>=3.10`。
- **已知缺口（未修，交主会话定夺）**：`pyproject.toml` 声明的入口点
  `rams-eval` / `rams-predict` / `rams-export` / `rams-serve` 对应模块
  `scripts/eval.py` / `predict.py` / `export.py` / `serve.py` **不存在**（0.1.0 占位遗留）。
  `python -m build` 不校验入口点模块存在性，wheel 照常构建成功，但这 4 个 CLI
  装完后运行即失败。`rams-build-dataset` / `rams-train` 两个入口点存在且可用。
  是否补建或删除入口点声明，建议主会话与冻结设计对齐后再定。

## 4. 冒烟记录（本地，全部未打印原始数据行）

| 冒烟项 | 命令 | 结果 |
|---|---|---|
| 全链路 pytest | `python -m pytest` | **53 passed，83%** |
| 数据管线 CLI | `python rams/data/tensor_builder.py --parquet data/processed/standard.parquet` | X(1601,30,20,2) / X_flat(1601,30,29)，bloom 正例 114/1601（0.071），冒烟通过 |
| 训练冒烟 | `python rams/training/trainer.py` | 两阶段 loss 有限，evaluate RMSE/acc/coverage 有限 |
| 滚动冒烟 | `python rams/training/smoke_roll.py` | q9 形状 (87,9,7)，CRPS 3.354 / 持久化 3.588 / 技能 +6.5%，覆盖率 0.138，冒烟通过 |
| M4 预警评估冒烟 | `python scripts/eval_m4_warning.py --smoke` | 评估期 5 事件：θ=0.5 召回 3/5（N8 提前 13 天、N9 提前 26 天），基线召回 5/5 但提前仅 4 天 + 14 误报段；与模块文档口径一致 |
| 正式训练 fast-dev | `python scripts/train_v020.py --windows 1 --seeds 1 --fast-dev` | 全链路跑通：CRPS 3.275 / 技能 +8.7% / M4 acc 0.862 |
| `import rams` 全链路 | `python -c "import rams, rams.data, rams.models, rams.training, ..."` | 全部成功，`__version__ = 0.1.0` |
| 打包构建 | `python -m build --outdir dist` | wheel + sdist 构建成功 |
| wheel 安装验证 | 装入独立 target + import + 前向 + 两阶段训练 | 通过 |

## 5. 验收达标情况

| 验收项（冻结设计） | 状态 |
|---|---|
| pytest 全链路通过 | ✅ **53/53 通过**（覆盖 83%，无跳过无告警） |
| 冒烟测试记录 | ✅ 上表 8 项全通过，已归档本报告 |
| 打包可构建 | ✅ `python -m build` wheel + sdist 成功，wheel 安装 import + 训练验证通过 |

## 6. 结论

mdl-test-package 模块完成：全链路 pytest 53 项通过、核心模块覆盖率 83%（较基线 71% 提升），
预处理与 M4 预警评估两个零覆盖缺口已补齐，wheel 构建与安装验证通过，冒烟记录完整。
附带修复了脚本模块级 stdout 劫持 bug（不影响脚本运行路径，仅消除被 import 时对 capture 的破坏）。
唯一需主会话关注的是 `requires-python = "<3.12"` 与本机 Python 3.14 的打包环境约束。
