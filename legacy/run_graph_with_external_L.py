# src/run_graph_with_external_L.py
"""
Train Graph-LassoNet prototype but load external datasets+L from datasets/*.npy
Usage:
    python src\run_graph_with_external_L.py
Parameters to tune are at the bottom of the file (lambda_l1, gamma, epochs, etc.)
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score

os.makedirs("../results/run_graph_with_external_L_results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# reuse model/prox from earlier prototype
class GraphLassoNetProtoExt(nn.Module):
    def __init__(self, input_dim, hidden_dims=(50,)):
        super().__init__()
        self.input_dim = input_dim
        self.theta = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        linear_res = x @ self.theta
        mlp_out = self.mlp(x).squeeze(-1)
        return linear_res + mlp_out

def prox_l1_inplace(theta_tensor, lam):
    with torch.no_grad():
        theta = theta_tensor.data
        sign = torch.sign(theta)
        new = torch.clamp(torch.abs(theta) - lam, min=0.0) * sign
        theta.copy_(new)

def load_data():
    base = os.path.dirname(os.path.dirname(__file__))  # 回到项目根目录
    X = np.load(os.path.join(base, "datasets/X.npy"))
    y = np.load(os.path.join(base, "datasets/y.npy"))
    L = np.load(os.path.join(base, "datasets/L.npy"))
    return X, y, L


def train_and_eval_external(seed=42, hidden_dims=(20,), lr=1e-3, batch_size=32, epochs=40,
                            lambda_l1=0.1, gamma=0.01, prox_interval=1, adaptive_update_every=10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X, y, L_np = load_data()
    p = X.shape[1]

    # datasets split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    L_torch = torch.tensor(L_np, dtype=torch.float32)

    model = GraphLassoNetProtoExt(p, hidden_dims=hidden_dims)
    optimizer = optim.Adam([p for n,p in model.named_parameters() if n != "theta"], lr=lr)
    bce = nn.BCEWithLogitsLoss()

    n_train = X_train.shape[0]
    n_batches = max(1, n_train // batch_size)

    history = {"test_acc": [], "test_auc": [], "n_nonzero_theta": []}
    start_time = time.time()

    # finite diff check with random theta (sanity)
    rand_theta = np.random.randn(p)
    # analytic grad = 2 * L @ theta
    analytic = (2.0 * (L_np @ rand_theta))[:10]
    # finite diff on first 10 coords
    eps = 1e-6
    fd = []
    for i in range(10):
        e = np.zeros(p); e[i] = 1.0
        fplus = (rand_theta + eps*e).T @ (L_np @ (rand_theta + eps*e))
        fminus = (rand_theta - eps*e).T @ (L_np @ (rand_theta - eps*e))
        fd.append((fplus - fminus) / (2*eps))
    print("finite-diff analytic (first10):", analytic.tolist())
    print("finite-diff fd      (first10):", fd)

    for epoch in range(1, epochs+1):
        # simple epoch training
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss_cls = bce(logits, yb)
            theta = model.theta
            lap_pen = theta @ (L_torch @ theta)
            loss = loss_cls + gamma * lap_pen
            loss.backward()
            optimizer.step()

            # theta manual step
            if model.theta.grad is not None:
                grad_theta = model.theta.grad.data.clone()
            else:
                grad_theta = torch.zeros_like(model.theta.data)
            with torch.no_grad():
                model.theta.data.add_(-lr * grad_theta)

        # prox
        prox_l1_inplace(model.theta, lambda_l1 * prox_interval)

        # eval
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t)
            probs = torch.sigmoid(logits_test).numpy()
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            nnz = int((model.theta.abs() > 1e-8).sum().item())
            history["test_acc"].append(acc)
            history["test_auc"].append(auc)
            history["n_nonzero_theta"].append(nnz)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | test_acc={acc:.4f} test_auc={auc:.4f} nnz_theta={nnz}")

    elapsed = time.time() - start_time
    # --- compute final laplacian energy and save theta
    with torch.no_grad():
        theta_np = model.theta.detach().cpu().numpy().copy()
    lap_energy = float(theta_np @ (L_np @ theta_np))

    results = {
        "final_test_acc": history["test_acc"][-1],
        "final_test_auc": history["test_auc"][-1],
        "n_nonzero_theta": history["n_nonzero_theta"][-1],
        "elapsed_seconds": elapsed,
        "lambda_l1": lambda_l1,
        "gamma": gamma,
        "hidden_dims": hidden_dims,
        "lap_energy": lap_energy
    }
    ts = time.strftime("%Y%m%d-%H%M%S")
    results_path = f"../results/run_graph_with_external_L_results/external_graph_results_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # also save theta array and selected mask
    import numpy as _np
    theta_path = f"../results/run_graph_with_external_L_results/theta_{ts}.npy"
    sel_path = f"../results/run_graph_with_external_L_results/selected_{ts}.npy"
    _np.save(theta_path, theta_np)
    _np.save(sel_path, (abs(theta_np) > 1e-8).astype(int))

    print("=== Results ===")
    print(json.dumps(results, indent=2))
    print("Saved:", results_path, "theta:", theta_path, "selected:", sel_path)

    # return history and theta
    return history, theta_np

def multiseed_train_and_eval_external(
        seeds=list(range(1, 21)),
        hidden_dims=(20,),
        lr=1e-3,
        batch_size=32,
        epochs=40,
        lambda_l1=0.1,
        gamma=0.01,
        prox_interval=1,
        adaptive_update_every=10):

    all_histories = []
    all_thetas = []

    print("\n=== Running multiseed external Graph-LassoNet ===")
    print("Seeds:", seeds)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        hist, theta_np = train_and_eval_external(
            seed=seed,
            hidden_dims=hidden_dims,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            lambda_l1=lambda_l1,
            gamma=gamma,
            prox_interval=prox_interval,
            adaptive_update_every=adaptive_update_every
        )

        all_histories.append(hist)
        all_thetas.append(theta_np)

    # ============ compute mean curves ============
    # each hist has: test_acc[], test_auc[], n_nonzero_theta[]
    mean_acc = np.mean([h["test_acc"] for h in all_histories], axis=0).tolist()
    mean_auc = np.mean([h["test_auc"] for h in all_histories], axis=0).tolist()
    mean_nnz = np.mean([h["n_nonzero_theta"] for h in all_histories], axis=0).tolist()

    # ============ save ============
    os.makedirs("../results/run_graph_with_external_L_results", exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"../results/run_graph_with_external_L_results/external_graph_multiseed_{ts}.json"

    out = {
        "seeds": seeds,
        "mean_test_acc": mean_acc,
        "mean_test_auc": mean_auc,
        "mean_nnz": mean_nnz,
        "per_seed": [
            {
                "seed": s,
                "test_acc": all_histories[i]["test_acc"],
                "test_auc": all_histories[i]["test_auc"],
                "n_nonzero_theta": all_histories[i]["n_nonzero_theta"]
            }
            for i, s in enumerate(seeds)
        ]
    }

    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Multiseed results saved to:", save_path)
    return out


if __name__ == "__main__":
    # run multiseed version
    multiseed_train_and_eval_external(
        seeds=list(range(1, 21)),
        hidden_dims=(20,),
        lr=1e-3,
        batch_size=32,
        epochs=40,
        lambda_l1=0.1,
        gamma=0.01,
        prox_interval=1,
        adaptive_update_every=10
    )