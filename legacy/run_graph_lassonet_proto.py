# src/run_graph_lassonet_proto.py
"""
Prototype: Graph-Laplacian regularized Lasso-like neural net.
Saves run_graph_multiseed_results to run_graph_multiseed_results/ and logs to logs/.
Usage:
    conda activate lassonet-env
    python src\run_graph_lassonet_proto.py > logs\graph_proto_run.log
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import sparse

os.makedirs("../results/run_graph_multiseed_results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------
# Utilities: build Laplacian
# ----------------------------
def build_W_from_corr(X, threshold=0.2, tau=5.0):
    """
    Build non-negative similarity matrix W based on Pearson correlation of features.
    X: (n_samples, n_features)
    threshold: absolute correlation below which we drop edge
    tau: scaling parameter in exponential kernel
    Returns dense numpy W (p x p)
    """
    corr = np.corrcoef(X.T)  # p x p
    np.fill_diagonal(corr, 1.0)
    W = np.exp(tau * np.abs(corr)) - 1.0  # make positive, monotonic
    W[np.abs(corr) < threshold] = 0.0
    # symmetrize
    W = (W + W.T) / 2.0
    return W

def laplacian_from_W(W, normalized=False):
    # W: numpy p x p
    d = np.sum(W, axis=1)
    D = np.diag(d)
    L = D - W
    if normalized:
        # symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
        with np.errstate(divide='ignore'):
            d_sqrt_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_sqrt_inv)
        L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
        return L_sym
    return L

# ----------------------------
# Model
# ----------------------------
class GraphLassoNetProto(nn.Module):
    def __init__(self, input_dim, hidden_dims=(50,)):
        super().__init__()
        self.input_dim = input_dim
        # theta: residual linear weights (no bias for simplicity)
        # We'll treat theta as a parameter vector we can prox on.
        self.theta = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))
        # small MLP
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))  # binary classification logit output
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, p)
        linear_res = x @ self.theta  # (batch,)
        mlp_out = self.mlp(x).squeeze(-1)  # (batch,)
        return linear_res + mlp_out

# ----------------------------
# Proximal soft-threshold for theta (L1 prox)
# ----------------------------
def prox_l1_inplace(theta_tensor, lam):
    # theta_tensor: nn.Parameter tensor (in-place update)
    with torch.no_grad():
        # soft-thresholding
        theta = theta_tensor.data
        theta_sign = torch.sign(theta)
        theta_abs = torch.abs(theta)
        new = torch.clamp(theta_abs - lam, min=0.0) * theta_sign
        theta.copy_(new)

# ----------------------------
# Finite difference check for Laplacian gradient
# ----------------------------
def finite_diff_grad_check(theta_np, L_np, eps=1e-6):
    # compute analytic gradient of g(theta) = theta^T L theta = 2 L theta (if L symmetric)
    # We will compare finite-difference on a random direction
    p = theta_np.size
    theta = theta_np.astype(np.float64)
    L = L_np.astype(np.float64)
    analytic = 2.0 * (L @ theta)  # shape (p,)
    # do finite diff for first few coords
    fd = np.zeros_like(theta)
    for i in range(min(10, p)):
        e = np.zeros_like(theta); e[i] = 1.0
        f_plus = (theta + eps * e).T @ (L @ (theta + eps * e))
        f_minus = (theta - eps * e).T @ (L @ (theta - eps * e))
        fd[i] = (f_plus - f_minus) / (2 * eps)
    return analytic[:10], fd[:10]

# ----------------------------
# Training / main
# ----------------------------
def train_and_eval(seed=42,
                   hidden_dims=(30,),
                   lr=1e-3,
                   batch_size=64,
                   epochs=50,
                   lambda_l1=1e-3,
                   gamma=1e-2,
                   prox_interval=1,
                   adaptive_update_every=5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_breast_cancer()
    X, y = data.data, data.target
    p = X.shape[1]

    # build initial W and Laplacian (adaptive)
    W = build_W_from_corr(X, threshold=0.2, tau=3.0)
    L_np = laplacian_from_W(W, normalized=True)
    L_torch = torch.tensor(L_np, dtype=torch.float32, device=device)

    # datasets split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    model = GraphLassoNetProto(p, hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam([p for n, p in model.named_parameters() if n != "theta"], lr=lr)
    # Note: we'll handle theta updates manually (grad desc + prox)

    bce = nn.BCEWithLogitsLoss()

    n_train = X_train.shape[0]
    n_batches = max(1, n_train // batch_size)

    history = {"train_loss": [], "test_acc": [], "test_auc": [], "n_nonzero_theta": []}

    start_time = time.time()

    # finite-diff sanity check print (for Laplacian)
    print("Finite-diff check for Laplacian gradient (first 10 coords):")
    analytic, fd = finite_diff_grad_check(np.zeros(p), L_np)
    print("analytic:", analytic.tolist())
    print("fd     :", fd.tolist())

    for epoch in range(1, epochs + 1):
        model.train()
        # shuffle
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            # we will compute gradient w.r.t mlp parameters; theta will be handled after backward
            logits = model(xb)
            loss_cls = bce(logits, yb)
            # Laplacian penalty on theta (as scalar)
            theta = model.theta
            lap_pen = (theta @ (L_torch @ theta))  # scalar
            loss = loss_cls + gamma * lap_pen
            loss.backward()
            optimizer.step()

            # manual gradient step for theta (simple gradient descent on theta)
            # grad for theta from classification path
            if model.theta.grad is not None:
                grad_theta = model.theta.grad.data.clone()
            else:
                grad_theta = torch.zeros_like(model.theta.data)

            # include gradient from lap_pen (we've already included lap_pen in loss.backward,
            # but because theta was a parameter, its grad exists; to be explicit we can re-add)
            # Here we will do an explicit step and then the prox for L1
            with torch.no_grad():
                # simple SGD step on theta using grad stored in model.theta.grad
                step = - lr * grad_theta
                model.theta.data.add_(step)

        # proximal step for L1 (soft-threshold)
        prox_l1_inplace(model.theta, lambda_l1 * prox_interval)

        # optionally update W / L (adaptive) every few epochs
        if epoch % adaptive_update_every == 0 and epoch > 0:
            W = build_W_from_corr(X_train, threshold=0.15, tau=3.0)
            L_np = laplacian_from_W(W, normalized=True)
            L_torch = torch.tensor(L_np, dtype=torch.float32, device=device)
            print(f"Epoch {epoch}: updated Laplacian (adaptive weights).")

        # eval
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t)
            probs = torch.sigmoid(logits_test).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            nonzero = int((model.theta.abs() > 1e-6).sum().item())
            history["test_acc"].append(acc)
            history["test_auc"].append(auc)
            history["n_nonzero_theta"].append(nonzero)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | test_acc={acc:.4f} test_auc={auc:.4f} nnz_theta={nonzero}")
    return history

def train_and_eval_multiseed(
            seeds,
            hidden_dims=(30,),
            lr=1e-3,
            batch_size=64,
            epochs=50,
            lambda_l1=1e-3,
            gamma=1e-2,
            prox_interval=1,
            adaptive_update_every=5):

    all_acc = []
    all_auc = []
    all_nnz = []

    all_histories = {}

    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        hist = train_and_eval(
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
        all_histories[seed] = hist

        all_acc.append(hist["test_acc"][-1])
        all_auc.append(hist["test_auc"][-1])
        all_nnz.append(hist["n_nonzero_theta"][-1])

    results = {
        "seeds": seeds,
        "mean_acc": float(np.mean(all_acc)),
        "mean_auc": float(np.mean(all_auc)),
        "mean_nnz": float(np.mean(all_nnz)),
        "all_results": [
            {"seed": s,
             "acc": all_histories[s]["test_acc"][-1],
             "auc": all_histories[s]["test_auc"][-1],
             "nnz": all_histories[s]["n_nonzero_theta"][-1]}
             for s in seeds
        ]
    }

    ts = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"../results/run_graph_multiseed_results/graph_proto_multiseed_{ts}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== MULTI-SEED SUMMARY ===")
    print(json.dumps(results, indent=2))
    print("Saved:", save_path)

    return results


if __name__ == "__main__":
    train_and_eval_multiseed(
        seeds=list(range(1, 21)),
        hidden_dims=(20,),
        lr=1e-3,
        batch_size=32,
        epochs=40,
        lambda_l1=1e-2,
        gamma=1e-2,
        prox_interval=1,
        adaptive_update_every=10
    )
