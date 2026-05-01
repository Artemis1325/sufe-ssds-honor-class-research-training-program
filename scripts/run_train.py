import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse

from lapreg_lassonet.config import RunConfig, DataConfig, ModelConfig, TrainConfig
from lapreg_lassonet.train.trainer import train_one_run, run_multiseed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="datasets")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default="graph_lassonet")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_mlp", type=float, default=1e-3)
    p.add_argument("--lr_theta", type=float, default=None)
    p.add_argument("--standardize_x", action="store_true")

    p.add_argument("--lambda_l1", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--prox_interval", type=int, default=1)

    p.add_argument("--hidden_dims", type=str, default="20")  # e.g. "50,20"
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--multiseed", action="store_true")
    p.add_argument("--seeds", type=str, default="1-20")  # "1-20" or "1,2,3"
    p.add_argument("--graph_mode", type=str, default="fixed", choices=["fixed", "pearson_train"])
    p.add_argument("--pearson_k", type=int, default=10)
    p.add_argument("--pearson_abs_corr", dest="pearson_abs_corr", action="store_true")
    p.add_argument("--signed_corr", dest="pearson_abs_corr", action="store_false")
    p.set_defaults(pearson_abs_corr=True)
    return p.parse_args()


def parse_hidden_dims(s: str):
    s = s.strip()
    if not s:
        return (20,)
    parts = [int(x) for x in s.split(",")]
    return tuple(parts)


def parse_seeds(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()

    cfg = RunConfig(
        data=DataConfig(
            dataset="npy",
            data_dir=args.data_dir,
            graph_mode=args.graph_mode,
            pearson_k=args.pearson_k,
            pearson_abs_corr=args.pearson_abs_corr,
        ),
        model=ModelConfig(hidden_dims=parse_hidden_dims(args.hidden_dims), task="binary", threshold=args.threshold),
        train=TrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_mlp=args.lr_mlp,
            lr_theta=args.lr_theta,
            lambda_l1=args.lambda_l1,
            gamma=args.gamma,
            prox_interval=args.prox_interval,
            device=args.device,
            standardize_x=args.standardize_x,
        ),
        results_dir=args.results_dir,
        run_name=args.run_name,
    )

    if args.multiseed:
        seeds = parse_seeds(args.seeds)
        out = run_multiseed(cfg, seeds)
        print("Saved multiseed:", out["artifacts"]["json"])
    else:
        out = train_one_run(cfg)
        print("Saved:", out["artifacts"]["json"])
        print("Final:", out["final"])


if __name__ == "__main__":
    print("RUN_TRAIN_ENTRY_OK")
    main()
