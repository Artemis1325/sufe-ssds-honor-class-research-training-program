import os
import sys
from statistics import mean, stdev

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from lapreg_lassonet.config import RunConfig, DataConfig, ModelConfig, TrainConfig
from lapreg_lassonet.train.trainer import train_one_run
from lapreg_lassonet.utils.io import save_json, results_subdir, timestamp


SEEDS = list(range(1, 21))

SETTINGS = [
    {"name": "mlp",            "lambda_l1": 0.0, "gamma": 0.0},
    {"name": "graph_mlp",      "lambda_l1": 0.0, "gamma": 0.001},
    {"name": "lassonet",       "lambda_l1": 0.1, "gamma": 0.0},
    {"name": "graph_lassonet", "lambda_l1": 0.1, "gamma": 0.001},
]


def safe_std(xs):
    return stdev(xs) if len(xs) >= 2 else 0.0


def main():
    results_dir = "results"
    run_name = "brca_ablation_tcga_20seed"
    data_dir = "datasets/tcga_brca"

    all_results = []

    for seed in SEEDS:
        print(f"\n===== seed {seed} =====")
        for s in SETTINGS:
            cfg = RunConfig(
                data=DataConfig(dataset="npy", data_dir=data_dir),
                model=ModelConfig(hidden_dims=(20,), task="binary"),
                train=TrainConfig(
                    seed=seed,
                    epochs=40,
                    batch_size=32,
                    lr_mlp=1e-3,
                    lambda_l1=s["lambda_l1"],
                    gamma=s["gamma"],
                    prox_interval=1,
                    device="cpu",
                ),
                results_dir=results_dir,
                run_name=f"{run_name}_seed{seed}_{s['name']}",
            )

            out = train_one_run(cfg)
            row = {
                "seed": seed,
                "model": s["name"],
                "lambda_l1": s["lambda_l1"],
                "gamma": s["gamma"],
                "final": out["final"],
                "artifacts": out["artifacts"],
            }
            all_results.append(row)

            f = out["final"]
            print(
                f"[ablation] seed={seed} model={s['name']}: "
                f"acc={f['acc']:.4f} auc={f['auc']:.4f} "
                f"nnz={f['nnz_theta']} lap={f['lap_energy']:.4f}"
            )

    metrics = ["acc", "auc", "nnz_theta", "lap_energy"]
    summary = {}

    for s in SETTINGS:
        model = s["name"]
        rows = [r for r in all_results if r["model"] == model]
        summary[model] = {
            "lambda_l1": s["lambda_l1"],
            "gamma": s["gamma"],
            "n_seeds": len(rows),
            "metrics": {},
        }
        for m in metrics:
            vals = [float(r["final"][m]) for r in rows]
            summary[model]["metrics"][m] = {
                "mean": mean(vals),
                "std": safe_std(vals),
                "values": vals,
            }

    print("\n===== summary =====")
    for model, info in summary.items():
        mm = info["metrics"]
        print(
            f"{model}: "
            f"acc={mm['acc']['mean']:.4f}±{mm['acc']['std']:.4f}, "
            f"auc={mm['auc']['mean']:.4f}±{mm['auc']['std']:.4f}, "
            f"nnz={mm['nnz_theta']['mean']:.2f}±{mm['nnz_theta']['std']:.2f}, "
            f"lap={mm['lap_energy']['mean']:.4f}±{mm['lap_energy']['std']:.4f}"
        )

    run_dir = results_subdir(results_dir, run_name)
    save_path = os.path.join(run_dir, f"ablation_{timestamp()}.json")
    save_json(save_path, {"seeds": SEEDS, "results": all_results, "summary": summary})
    print("\nSaved:", save_path)


if __name__ == "__main__":
    main()