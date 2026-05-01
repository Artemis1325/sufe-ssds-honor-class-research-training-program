# src/grid_search_extL.py
"""
Grid search over lambda_l1 and gamma for external-L experiment.
Saves run_grid_py_multiseed_results to run_grid_extL_multiseed_results/extL_grid_results.json and prints progress.
Requires: train_and_eval_external(...) available in run_graph_with_external_L.py
"""

import os
import json
import time
import itertools
from run_graph_with_external_L import train_and_eval_external

os.makedirs("../results/run_grid_extL_multiseed_results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---- grid  ----
lambda_list = [0.01, 0.1, 1.0, 5.0]
gamma_list  = [0.0, 0.001, 0.01, 0.1] # include 0.0 as baseline
# ---------------------------

# reduce epochs for quick test if needed
quick_mode = False
if quick_mode:
    extra_kwargs = {"epochs": 20}
else:
    extra_kwargs = {"epochs": 40}

results = {}
total = len(lambda_list) * len(gamma_list)
count = 0

SEEDS = list(range(1, 21))   # 20 seeds

def multiseed_eval(lam, gam):
    """Run train_and_eval_external over multiple seeds and return mean AUC/NNZ."""
    acc_list = []
    auc_list = []
    nnz_list = []

    for sd in SEEDS:
        print(f"      Seed {sd}")
        hist, _theta = train_and_eval_external(
            seed=sd,
            hidden_dims=(20,),
            lr=1e-3,
            batch_size=32,
            lambda_l1=lam,
            gamma=gam,
            prox_interval=1,
            adaptive_update_every=10,
            **extra_kwargs
        )

        acc_list.append(hist["test_acc"][-1])
        auc_list.append(hist["test_auc"][-1])
        nnz_list.append(hist["n_nonzero_theta"][-1])

    import numpy as np
    return {
        "mean_acc": float(np.mean(acc_list)),
        "mean_auc": float(np.mean(auc_list)),
        "mean_nnz": float(np.mean(nnz_list)),
        "per_seed": {
            "acc": acc_list,
            "auc": auc_list,
            "nnz": nnz_list
        }
    }

for lam, gam in itertools.product(lambda_list, gamma_list):
    count += 1
    print(f"\n=== RUN {count}/{total}: lambda={lam}  gamma={gam} ===")
    start = time.time()

    ms_results = multiseed_eval(lam, gam)

    elapsed = time.time() - start

    key = f"{lam}__{gam}"
    results[key] = {
        "lambda": lam,
        "gamma": gam,
        # store multiseed averages (used for heatmaps)
        "mean_acc": ms_results["mean_acc"],
        "mean_auc": ms_results["mean_auc"],
        "mean_nnz": ms_results["mean_nnz"],
        # store per-seed values for completeness
        "per_seed": ms_results["per_seed"],
        "time_sec": elapsed
    }

    # save intermediate
    with open("../results/run_grid_extL_multiseed_results/extL_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved intermediate multiseed results ({len(results)} entries). Elapsed: {elapsed:.1f}s")

print("Grid completed. Final multiseed results saved to run_grid_extL_multiseed_results/extL_grid_results.json")
