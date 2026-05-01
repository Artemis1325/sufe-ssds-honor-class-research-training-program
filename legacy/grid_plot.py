import json
import numpy as np
import matplotlib.pyplot as plt

# 如果文件在项目根 results/ 下，改为 "results/graph_proto_grid_results_multiseed.json"
json_path = "../results/run_grid_py_multiseed_results/graph_proto_grid_results_multiseed.json"

with open(json_path, "r") as f:
    data = json.load(f)

# 提取并排序
lams = []
mean_acc = []
mean_auc = []
mean_nnz = []

for k, entry in data.items():
    lams.append(float(entry.get("lambda", k)))
    mean_acc.append(entry["mean_acc"])
    mean_auc.append(entry["mean_auc"])
    mean_nnz.append(entry["mean_nnz"])

# sort by lambda
order = np.argsort(lams)
lams = np.array(lams)[order]
mean_acc = np.array(mean_acc)[order]
mean_auc = np.array(mean_auc)[order]
mean_nnz = np.array(mean_nnz)[order]

# Desired x-ticks: 1e-2, 1e-1, 1e0, 1e1
xticks = [1e-2, 1e-1, 1e0, 1e1]
xtick_labels = [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"]

# -------- Plot 1: Performance (Accuracy & AUC) on log-x --------
plt.figure(figsize=(6.5,4))
plt.plot(lams, mean_acc, marker='o', linestyle='-', label='Accuracy')
plt.plot(lams, mean_auc, marker='s', linestyle='--', label='AUC')

plt.xscale('log')
plt.xticks(xticks, xtick_labels)
# set y limits suitable for perf near 1.0
ymin = min(min(mean_acc), min(mean_auc)) - 0.02
ymax = max(max(mean_acc), max(mean_auc)) + 0.01
# keep reasonable bounds
ymin = max(0.0, ymin)
ymax = min(1.0, ymax if ymax <= 1.05 else 1.05)
plt.ylim(ymin, ymax)

plt.xlabel("Lambda")
plt.ylabel("Performance")
plt.title("Performance vs Lambda (mean over seeds)")
plt.grid(True, which='both', axis='x', linestyle=':', alpha=0.6)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("/mnt/datasets/performance_vs_lambda_log.png", dpi=200)
plt.show()

# -------- Plot 2: Sparsity (nnz) on log-x --------
plt.figure(figsize=(6.5,4))
plt.plot(lams, mean_nnz, marker='o', linestyle='-', color='black')
plt.xscale('log')
plt.xticks(xticks, xtick_labels)

# set y limit based on observed nnz and input dim (if you know p, set p)
p_est = int(max(mean_nnz) + 1) if max(mean_nnz) > 0 else 30
ymax_nnz = max( max(mean_nnz)*1.1, 1.0 )
ymax_nnz = max(ymax_nnz, p_est)
plt.ylim(0, ymax_nnz)

plt.xlabel("Lambda")
plt.ylabel("Number of selected features (nnz)")
plt.title("Sparsity vs Lambda (mean over seeds)")
plt.grid(True, which='both', axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("/mnt/datasets/sparsity_vs_lambda_log.png", dpi=200)
plt.show()
