# src/fine_grid_with_lapenergy.py
"""
Run a fine grid over lambda and gamma, save lap_energy as well as acc/auc/nnz.
Writes results to lapenergy_results/fine_extL_grid_results.json and saves theta arrays.
Now includes multiseed averaging (without modifying original logic structure).
"""
import json, itertools, time, os
import numpy as np
from run_graph_with_external_L import train_and_eval_external, load_data

os.makedirs("../results/lapenergy_results", exist_ok=True)

# fine grid
lambda_list = [0.05, 0.1, 0.2, 0.5]
gamma_list  = [0.0, 0.001, 0.01, 0.05]
seeds = list(range(1, 21))       # 20 seeds

results = {}
count = 0
total = len(lambda_list) * len(gamma_list)

for lam, gam in itertools.product(lambda_list, gamma_list):
    count += 1
    print(f"\n=== RUN {count}/{total}: lambda={lam}  gamma={gam} ===")

    per_seed_acc = []
    per_seed_auc = []
    per_seed_nnz = []
    per_seed_lap = []

    # Load L only once
    _, _, L_np = load_data()

    for sd in seeds:
        print(f"  -- seed {sd}")
        start = time.time()

        hist, theta = train_and_eval_external(
            seed=sd,
            hidden_dims=(20,),
            lr=1e-3,
            batch_size=32,
            epochs=40,
            lambda_l1=lam,
            gamma=gam,
            prox_interval=1,
            adaptive_update_every=10
        )
        elapsed = time.time() - start

        # collect last epoch values
        per_seed_acc.append(hist["test_acc"][-1])
        per_seed_auc.append(hist["test_auc"][-1])
        per_seed_nnz.append(hist["n_nonzero_theta"][-1])

        # lap energy
        lap_energy = float(theta @ (L_np @ theta))
        per_seed_lap.append(lap_energy)

    # ======= multiseed average =======
    mean_acc = float(np.mean(per_seed_acc))
    mean_auc = float(np.mean(per_seed_auc))
    mean_nnz = float(np.mean(per_seed_nnz))
    mean_lap = float(np.mean(per_seed_lap))

    key = f"{lam}__{gam}"
    results[key] = {
        "lambda": lam,
        "gamma": gam,
        "mean_acc": mean_acc,
        "mean_auc": mean_auc,
        "mean_nnz": mean_nnz,
        "mean_lap_energy": mean_lap,
        "per_seed": {
            "acc": per_seed_acc,
            "auc": per_seed_auc,
            "nnz": per_seed_nnz,
            "lap": per_seed_lap
        }
    }

    with open("../results/lapenergy_results/fine_extL_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved intermediate lapenergy_results for this grid point.")

print("Finished fine grid. Results saved to lapenergy_results/fine_extL_grid_results.json")
