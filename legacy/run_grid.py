# src/run_grid.py
"""
Run a quick grid of lambda_l1 to see sparsity vs accuracy tradeoff.
Saves combined run_grid_py_multiseed_results to run_grid_py_multiseed_results/graph_proto_grid_results_multiseed.json
and logs to console.
"""

import json
from run_graph_lassonet_proto import train_and_eval

lams = [0.01, 0.1, 1.0, 5.0, 10.0]
seeds = list(range(1, 21))   # 20 seeds

grid_results = {}

for lam in lams:
    print("\n=== RUN lambda_l1 =", lam, "===")
    seed_results = []

    for sd in seeds:
        print(f"  -- seed {sd}")
        hist = train_and_eval(
            seed=sd,
            hidden_dims=(20,),
            lr=1e-3,
            batch_size=32,
            epochs=20,
            lambda_l1=lam,
            gamma=0.01,
            prox_interval=1,
            adaptive_update_every=10
        )
        seed_results.append({
            "acc": hist["test_acc"][-1],
            "auc": hist["test_auc"][-1],
            "nnz": hist["n_nonzero_theta"][-1]
        })

    # compute mean/std
    mean_acc = sum(r["acc"] for r in seed_results) / len(seed_results)
    mean_auc = sum(r["auc"] for r in seed_results) / len(seed_results)
    mean_nnz = sum(r["nnz"] for r in seed_results) / len(seed_results)

    grid_results[str(lam)] = {
        "lambda": lam,
        "mean_acc": mean_acc,
        "mean_auc": mean_auc,
        "mean_nnz": mean_nnz,
        "all_results": seed_results,
        "seeds": seeds,
    }

# save
out_path = "../results/run_grid_py_multiseed_results/graph_proto_grid_results_multiseed.json"
with open(out_path, "w") as f:
    json.dump(grid_results, f, indent=2)

print("\nSaved multiseed grid run_grid_py_multiseed_results to", out_path)
