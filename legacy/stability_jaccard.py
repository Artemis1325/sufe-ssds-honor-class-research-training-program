# src/stability_jaccard.py
"""
Run multiple seeds for fixed lambda & gamma, save selected feature sets and compute pairwise Jaccard.
"""
import numpy as np, json, os, itertools
from run_graph_with_external_L import train_and_eval_external, load_data

os.makedirs("../results/jaccard_results", exist_ok=True)

lambda_l1 = 0.1
gamma = 0.01
seeds = list(range(1, 21))

selected_sets = []
thetas = []
for s in seeds:
    print("RUN seed", s)
    hist, theta = train_and_eval_external(
        seed=s, hidden_dims=(20,), lr=1e-3, batch_size=32, epochs=40,
        lambda_l1=lambda_l1, gamma=gamma, prox_interval=1, adaptive_update_every=10
    )
    sel = set(np.where(np.abs(theta) > 1e-8)[0].tolist())
    selected_sets.append(sel)
    thetas.append(theta)
    # save per-seed
    np.save(f"../results/jaccard_results/selected_seed_{s}.npy", np.array(list(sel), dtype=int))
    np.save(f"../results/jaccard_results/theta_seed_{s}.npy", theta)

# compute pairwise Jaccard
def jacc(a,b):
    if len(a|b) == 0: return 1.0
    return len(a & b) / len(a | b)

pairs = list(itertools.combinations(range(len(seeds)), 2))
jacs = []
for i,j in pairs:
    ji = jacc(selected_sets[i], selected_sets[j])
    jacs.append(ji)

mean_jacc = float(np.mean(jacs))
median_jacc = float(np.median(jacs))

res = {
    "lambda": lambda_l1,
    "gamma": gamma,
    "seeds": seeds,
    "pairwise_jacc": {f"{seeds[i]}_{seeds[j]}": float(jacc(selected_sets[i], selected_sets[j])) for i,j in pairs},
    "mean_jacc": mean_jacc,
    "median_jacc": median_jacc,
    "selected_counts": {str(seeds[i]): len(selected_sets[i]) for i in range(len(seeds))}
}

with open("../results/jaccard_results/stability_jaccard_results.json", "w") as f:
    json.dump(res, f, indent=2)

print("Mean Jaccard:", mean_jacc)
print("Saved results to jaccard_results/stability_jaccard_results.json")
