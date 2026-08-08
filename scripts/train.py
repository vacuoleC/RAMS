# -*- coding: utf-8 -*-
"""train CLI：训练 RamsNet（M1 预测 + M2 分层）

用法：
  # 冒烟（推荐先跑）
  python -m scripts.train --fast-dev-run

  # 正式训练
  python -m scripts.train --epochs 30 --parquet data/processed/standard.parquet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 RAMS 多任务模型")
    parser.add_argument("--parquet", default="data/processed/standard.parquet")
    parser.add_argument("--T", type=int, default=24, help="回看窗口")
    parser.add_argument("--H", type=int, default=8, help="预测步数（24h）")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-m2", type=float, default=3.0, help="M2 分层任务权重")
    parser.add_argument("--no-quantile", action="store_true", help="关闭分位数输出")
    parser.add_argument("--fast-dev-run", action="store_true", help="冒烟模式")
    args = parser.parse_args()

    from rams.data.tensor_builder import TensorBuilder, TensorConfig
    from rams.models.rams_net import RamsNet
    from rams.training.trainer import Trainer

    cfg = TensorConfig(T=args.T, H=args.H)
    ds = TensorBuilder(cfg).build(args.parquet)
    (X_tr, y_tr, s_tr), (X_va, y_va, s_va), (X_te, y_te, s_te) = (
        ds["train"], ds["val"], ds["test"])
    print(f"数据: train {X_tr.shape} val {X_va.shape} test {X_te.shape}", flush=True)

    model = RamsNet(
        feat_dim=ds["feat_dim"], horizon=args.H,
        hidden=args.hidden, quantile=not args.no_quantile,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}", flush=True)

    trainer = Trainer(model, lr=args.lr, w_m2=args.w_m2)
    trainer.fit(X_tr, y_tr, s_tr, X_va, y_va, s_va,
                epochs=args.epochs, batch_size=args.batch_size,
                fast_dev_run=args.fast_dev_run)

    if not args.fast_dev_run:
        res = trainer.evaluate(X_te, y_te, s_te, ds["y_sd"])
        print("\n=== 测试结果 ===", flush=True)
        print(f"  M1 RMSE = {res['rmse']:.4f}（原始浓度单位）", flush=True)
        print(f"  M2 acc  = {res['acc']:.4f}", flush=True)
        if "coverage" in res:
            print(f"  p10-p90 覆盖率 = {res['coverage']:.3f}", flush=True)
            print(f"  平均区间宽 = {res['interval_width']:.3f}", flush=True)


if __name__ == "__main__":
    main()
