import json
import matplotlib.pyplot as plt

# === Load your uploaded result ===
path = "external_graph_multiseed.json"

with open(path, "r") as f:
    data = json.load(f)

mean_auc = data["mean_test_auc"]
mean_nnz = data["mean_nnz"]

epochs = list(range(1, len(mean_auc)+1))

# ============================
# 1. Training Dynamics (AUC)
# ============================

plt.figure(figsize=(7,5))
plt.plot(epochs, mean_auc, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("AUC (mean over seeds)")
plt.title("Training dynamics with Laplacian")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================
# 2. Feature Selection (NNZ)
# ============================

plt.figure(figsize=(7,5))
plt.plot(epochs, mean_nnz, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Number of selected features (mean NNZ)")
plt.title("Feature selection over training")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
