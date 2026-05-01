import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import itertools
import os

from lapreg_lassonet.config import RunConfig, DataConfig, ModelConfig, TrainConfig
from lapreg_lassonet.train.trainer import train_one_run
from lapreg_lassonet.utils.io import save_json, results_subdir, timestamp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="datasets")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default="grid")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_mlp", type=float, default=1e-3)
    p.add_argument("--lr_theta", type=float, default=None)

    p.add_argument("--hidden_dims", type=str, default="20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--lambdas", type=str, default="0.01,0.05,0.1,0.2,0.5")
    p.add_argument("--gammas", type=str, default="0.0,0.001,0.01,0.05,0.1")
    return p.parse_args()


def parse_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_hidden_dims(s: str):
    return tuple(int(x) for x in s.split(",") if x.strip())


def main():
    args = parse_args()

    lambdas = parse_list(args.lambdas)
    gammas = parse_list(args.gammas)

    grid_results = []
    for lam, gam in itertools.product(lambdas, gammas):
        cfg = RunConfig(
            data=DataConfig(dataset="npy", data_dir=args.data_dir),
            model=ModelConfig(hidden_dims=parse_hidden_dims(args.hidden_dims), task="binary"),
            train=TrainConfig(
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr_mlp=args.lr_mlp,
                lr_theta=args.lr_theta,
                lambda_l1=lam,
                gamma=gam,
                prox_interval=1,
                device=args.device,
            ),
            results_dir=args.results_dir,
            run_name=args.run_name,
        )
        out = train_one_run(cfg)
        grid_results.append({
            "lambda_l1": lam,
            "gamma": gam,
            "final": out["final"],
            "artifacts": out["artifacts"],
        })
        print(f"[grid] lam={lam} gam={gam} -> acc={out['final']['acc']:.4f} auc={out['final']['auc']:.4f} nnz={out['final']['nnz_theta']} lap={out['final']['lap_energy']:.4f}")

    run_dir = results_subdir(args.results_dir, args.run_name)
    ts = timestamp()
    save_path = os.path.join(run_dir, f"grid_{ts}.json")
    save_json(save_path, {
        "lambdas": lambdas,
        "gammas": gammas,
        "results": grid_results,
    })
    print("Saved grid:", save_path)


if __name__ == "__main__":
    main()
