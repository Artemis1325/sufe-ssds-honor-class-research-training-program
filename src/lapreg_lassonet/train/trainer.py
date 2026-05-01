from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from ..config import RunConfig
from ..models.graph_lassonet import GraphLassoNet, prox_l1_inplace
from ..eval.metrics import nnz as nnz_fn, lap_energy as lap_energy_fn, binary_metrics_from_logits
from ..utils.io import timestamp, results_subdir, save_json, save_npy
from ..data.datasets import load_npy_dataset
from ..data.graphs import pearson_knn_laplacian


def _to_torch_L(L: Any, device: str) -> Any:
    """
    Return:
      - torch.Tensor for dense
      - torch.sparse_coo_tensor for scipy sparse
    """
    if sp is not None and sp.issparse(L):
        L = L.tocoo()
        indices = torch.tensor(np.vstack([L.row, L.col]), dtype=torch.long)
        values = torch.tensor(L.data, dtype=torch.float32)
        shape = torch.Size(L.shape)
        Lt = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=device).coalesce()
        return Lt
    # dense numpy
    return torch.tensor(np.asarray(L), dtype=torch.float32, device=device)


def _lap_penalty(theta: torch.Tensor, L_t: Any) -> torch.Tensor:
    """
    theta^T L theta
    Works for:
      - dense torch.Tensor L_t
      - sparse torch.sparse_coo_tensor L_t
    """
    if isinstance(L_t, torch.Tensor) and L_t.is_sparse:
        v = torch.sparse.mm(L_t, theta.view(-1, 1)).view(-1)
        return torch.dot(theta, v)
    else:
        return theta @ (L_t @ theta)


def _load_dataset(cfg: RunConfig):
    if cfg.data.dataset == "npy":
        return load_npy_dataset(cfg.data.data_dir)
    raise ValueError(f"Unknown dataset={cfg.data.dataset}")


def _resolve_graph_for_split(cfg: RunConfig, X_train: np.ndarray, L_fixed: Any):
    graph_mode = str(cfg.data.graph_mode).lower()
    if graph_mode == "fixed":
        return L_fixed, {
            "graph_mode": "fixed",
            "graph_source": "precomputed_dataset_laplacian",
        }
    if graph_mode == "pearson_train":
        L_dyn, graph_info = pearson_knn_laplacian(
            X_train,
            k=cfg.data.pearson_k,
            abs_corr=cfg.data.pearson_abs_corr,
            normalized=cfg.data.pearson_normalized_laplacian,
        )
        return L_dyn, graph_info
    raise ValueError(f"Unknown graph_mode={cfg.data.graph_mode}")


def train_one_run(cfg: RunConfig) -> Dict[str, Any]:
    """
    One training run (one seed). Returns a dict with:
      metrics, history, artifact paths, config snapshot
    """
    device = torch.device(cfg.train.device)
    seed = int(cfg.train.seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y, L, meta = _load_dataset(cfg)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)

    # split
    stratify = y if cfg.data.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=seed, stratify=stratify
    )

    if getattr(cfg.train, "standardize_x", False):
        mean = X_train.mean(axis=0, dtype=np.float64)
        std = X_train.std(axis=0, dtype=np.float64)
        eps = float(getattr(cfg.train, "standardize_eps", 1e-8))
        std = np.where(std < eps, 1.0, std)
        X_train = ((X_train - mean) / std).astype(np.float32)
        X_test = ((X_test - mean) / std).astype(np.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_np = y_test.copy()

    L_run, graph_info = _resolve_graph_for_split(cfg, X_train, L)

    p = X_train.shape[1]
    L_t = _to_torch_L(L_run, cfg.train.device)

    model = GraphLassoNet(input_dim=p, hidden_dims=cfg.model.hidden_dims).to(device)
    bce = nn.BCEWithLogitsLoss()

    lr_theta = cfg.train.lr_mlp if cfg.train.lr_theta is None else cfg.train.lr_theta

    # IMPORTANT: optimizer only for MLP parameters (exclude theta)
    mlp_params = [param for name, param in model.named_parameters() if name != "theta"]
    optimizer = optim.Adam(mlp_params, lr=cfg.train.lr_mlp)

    n_train = X_train.shape[0]
    batch_size = int(cfg.train.batch_size)
    n_batches = max(1, int(np.ceil(n_train / batch_size)))

    history = {
        "test_acc": [],
        "test_auc": [],
        "nnz_theta": [],
    }

    start = time.time()

    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)

            loss_cls = bce(logits, yb)
            theta = model.theta
            lap = _lap_penalty(theta, L_t)
            loss = loss_cls + float(cfg.train.gamma) * lap

            loss.backward()
            optimizer.step()

            # explicit theta step (SGD)
            with torch.no_grad():
                if model.theta.grad is not None:
                    model.theta.data.add_(-float(lr_theta) * model.theta.grad.data)

        # proximal step every prox_interval epochs
        if cfg.train.prox_interval > 0 and (epoch % cfg.train.prox_interval == 0):
            prox_l1_inplace(model.theta, float(cfg.train.lambda_l1))

        # eval
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t).detach().cpu().numpy()
            m = binary_metrics_from_logits(y_test_np, logits_test, threshold=cfg.model.threshold)

            theta_np = model.theta.detach().cpu().numpy().copy()
            nnz_val = nnz_fn(theta_np, eps=cfg.train.theta_nnz_eps)

            history["test_acc"].append(m["acc"])
            history["test_auc"].append(m["auc"])
            history["nnz_theta"].append(nnz_val)

    elapsed = time.time() - start

    # final artifacts/metrics
    theta_np = model.theta.detach().cpu().numpy().copy()
    lap_e = lap_energy_fn(theta_np, L_run)
    nnz_val = nnz_fn(theta_np, eps=cfg.train.theta_nnz_eps)

    final_logits = model(X_test_t).detach().cpu().numpy()
    final_m = binary_metrics_from_logits(y_test_np, final_logits, threshold=cfg.model.threshold)

    run_dir = results_subdir(cfg.results_dir, cfg.run_name)
    ts = timestamp()

    out: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "meta": {**meta, "graph_info": graph_info},
        "seed": seed,
        "final": {
            "acc": final_m["acc"],
            "auc": final_m["auc"],
            "nnz_theta": nnz_val,
            "lap_energy": float(lap_e),
            "elapsed_seconds": float(elapsed),
        },
        "history": history,
        "artifacts": {},
    }

    # save outputs
    json_path = f"{run_dir}/run_{ts}_seed{seed}.json"
    save_json(json_path, out)
    out["artifacts"]["json"] = json_path

    if cfg.save_theta:
        theta_path = f"{run_dir}/theta_{ts}_seed{seed}.npy"
        save_npy(theta_path, theta_np)
        out["artifacts"]["theta"] = theta_path

    if cfg.save_selected:
        sel = (np.abs(theta_np) > cfg.train.theta_nnz_eps).astype(np.int32)
        sel_path = f"{run_dir}/selected_{ts}_seed{seed}.npy"
        save_npy(sel_path, sel)
        out["artifacts"]["selected"] = sel_path

    if cfg.save_history:
        # history already inside json; nothing extra needed
        pass

    return out


def run_multiseed(cfg: RunConfig, seeds: List[int]) -> Dict[str, Any]:
    """
    Run multiple seeds and aggregate mean curves.
    """
    per_seed = []
    for s in seeds:
        cfg_s = cfg
        cfg_s.train.seed = int(s)
        per_seed.append(train_one_run(cfg_s))

    # aggregate
    acc_curves = [r["history"]["test_acc"] for r in per_seed]
    auc_curves = [r["history"]["test_auc"] for r in per_seed]
    nnz_curves = [r["history"]["nnz_theta"] for r in per_seed]

    mean_acc = np.mean(np.asarray(acc_curves, dtype=float), axis=0).tolist()
    mean_auc = np.mean(np.asarray(auc_curves, dtype=float), axis=0).tolist()
    mean_nnz = np.mean(np.asarray(nnz_curves, dtype=float), axis=0).tolist()

    run_dir = results_subdir(cfg.results_dir, cfg.run_name)
    ts = timestamp()
    out = {
        "config": cfg.to_dict(),
        "seeds": seeds,
        "mean": {
            "test_acc": mean_acc,
            "test_auc": mean_auc,
            "nnz_theta": mean_nnz,
        },
        "per_seed": [
            {
                "seed": r["seed"],
                "final": r["final"],
                "artifacts": r["artifacts"],
                "history": r["history"],
            }
            for r in per_seed
        ],
    }

    json_path = f"{run_dir}/multiseed_{ts}.json"
    save_json(json_path, out)
    out["artifacts"] = {"json": json_path}
    return out


"""
#做外部实验时用下面这版
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from ..config import RunConfig
from ..models.graph_lassonet import GraphLassoNet, prox_l1_inplace
from ..eval.metrics import nnz as nnz_fn, lap_energy as lap_energy_fn, binary_metrics_from_logits
from ..utils.io import timestamp, results_subdir, save_json, save_npy
from ..data.datasets import load_npy_dataset

"""
"""
def _to_torch_L(L: Any, device: str) -> Any:
    """"""
    Return:
      - torch.Tensor for dense
      - torch.sparse_coo_tensor for scipy sparse
    """"""
    if sp is not None and sp.issparse(L):
        L = L.tocoo()
        indices = torch.tensor(np.vstack([L.row, L.col]), dtype=torch.long)
        values = torch.tensor(L.data, dtype=torch.float32)
        shape = torch.Size(L.shape)
        Lt = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=device).coalesce()
        return Lt
    # dense numpy
    return torch.tensor(np.asarray(L), dtype=torch.float32, device=device)


def _lap_penalty(theta: torch.Tensor, L_t: Any) -> torch.Tensor:
    """"""
    theta^T L theta
    Works for:
      - dense torch.Tensor L_t
      - sparse torch.sparse_coo_tensor L_t
    """"""
    if isinstance(L_t, torch.Tensor) and L_t.is_sparse:
        v = torch.sparse.mm(L_t, theta.view(-1, 1)).view(-1)
        return torch.dot(theta, v)
    else:
        return theta @ (L_t @ theta)


def _load_dataset(cfg: RunConfig):
    if cfg.data.dataset == "npy":
        return load_npy_dataset(cfg.data.data_dir)
    raise ValueError(f"Unknown dataset={cfg.data.dataset}")


def train_one_run(cfg: RunConfig) -> Dict[str, Any]:
    """"""
    One training run (one seed). Returns a dict with:
      metrics, history, artifact paths, config snapshot
    """"""
    device = torch.device(cfg.train.device)
    seed = int(cfg.train.seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y, L, meta = _load_dataset(cfg)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)

    # split
    stratify = y if cfg.data.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=seed, stratify=stratify
    )
    # standardize using training split only
    scaler_mean = None
    scaler_std = None
    if cfg.train.standardize_x:
        scaler_mean = X_train.mean(axis=0, dtype=np.float64)
        scaler_std = X_train.std(axis=0, dtype=np.float64)
        eps = float(cfg.train.standardize_eps)
        scaler_std = np.where(scaler_std < eps, 1.0, scaler_std)

        X_train = ((X_train - scaler_mean) / scaler_std).astype(np.float32)
        X_test = ((X_test - scaler_mean) / scaler_std).astype(np.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_np = y_test.copy()

    p = X_train.shape[1]
    L_t = _to_torch_L(L, cfg.train.device)

    model = GraphLassoNet(input_dim=p, hidden_dims=cfg.model.hidden_dims).to(device)
    bce = nn.BCEWithLogitsLoss()

    lr_theta = cfg.train.lr_mlp if cfg.train.lr_theta is None else cfg.train.lr_theta

    # IMPORTANT: optimizer only for MLP parameters (exclude theta)
    mlp_params = [param for name, param in model.named_parameters() if name != "theta"]
    optimizer = optim.Adam(mlp_params, lr=cfg.train.lr_mlp)

    n_train = X_train.shape[0]
    batch_size = int(cfg.train.batch_size)
    n_batches = max(1, int(np.ceil(n_train / batch_size)))

    history = {
        "test_acc": [],
        "test_auc": [],
        "nnz_theta": [],
    }

    start = time.time()

    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        perm = np.random.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            #optimizer.zero_grad()
            #logits = model(xb)
            optimizer.zero_grad()
            if model.theta.grad is not None:
                model.theta.grad = None
            logits = model(xb)

            loss_cls = bce(logits, yb)
            theta = model.theta
            lap = _lap_penalty(theta, L_t)
            loss = loss_cls + float(cfg.train.gamma) * lap

            loss.backward()
            optimizer.step()

            # explicit theta step (SGD)
            with torch.no_grad():
                if model.theta.grad is not None:
                    model.theta.data.add_(-float(lr_theta) * model.theta.grad.data)

        # proximal step every prox_interval epochs
        if cfg.train.prox_interval > 0 and (epoch % cfg.train.prox_interval == 0):
            prox_l1_inplace(model.theta, float(cfg.train.lambda_l1))

        # eval
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t).detach().cpu().numpy()
            m = binary_metrics_from_logits(y_test_np, logits_test, threshold=cfg.model.threshold)

            theta_np = model.theta.detach().cpu().numpy().copy()
            nnz_val = nnz_fn(theta_np, eps=cfg.train.theta_nnz_eps)

            history["test_acc"].append(m["acc"])
            history["test_auc"].append(m["auc"])
            history["nnz_theta"].append(nnz_val)

    elapsed = time.time() - start

    # final artifacts/metrics
    theta_np = model.theta.detach().cpu().numpy().copy()
    lap_e = lap_energy_fn(theta_np, L_run)
    nnz_val = nnz_fn(theta_np, eps=cfg.train.theta_nnz_eps)

    final_logits = model(X_test_t).detach().cpu().numpy()
    final_m = binary_metrics_from_logits(y_test_np, final_logits, threshold=cfg.model.threshold)

    run_dir = results_subdir(cfg.results_dir, cfg.run_name)
    ts = timestamp()

    out: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "meta": meta,
        "seed": seed,
        "final": {
            "acc": final_m["acc"],
            "auc": final_m["auc"],
            "nnz_theta": nnz_val,
            "lap_energy": float(lap_e),
            "elapsed_seconds": float(elapsed),
        },
        "history": history,
        "artifacts": {},
    }

    # save outputs
    json_path = f"{run_dir}/run_{ts}_seed{seed}.json"
    save_json(json_path, out)
    out["artifacts"]["json"] = json_path

    if cfg.save_theta:
        theta_path = f"{run_dir}/theta_{ts}_seed{seed}.npy"
        save_npy(theta_path, theta_np)
        out["artifacts"]["theta"] = theta_path

    if cfg.save_selected:
        sel = (np.abs(theta_np) > cfg.train.theta_nnz_eps).astype(np.int32)
        sel_path = f"{run_dir}/selected_{ts}_seed{seed}.npy"
        save_npy(sel_path, sel)
        out["artifacts"]["selected"] = sel_path

    if cfg.save_history:
        # history already inside json; nothing extra needed
        pass

    if cfg.save_model:
        model_path = f"{run_dir}/model_{ts}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": p,
                "hidden_dims": cfg.model.hidden_dims,
                "threshold": cfg.model.threshold,
                "seed": seed,
                "config": cfg.to_dict(),
            },
            model_path,
        )
        out["artifacts"]["model"] = model_path

    if cfg.save_scaler and cfg.train.standardize_x:
        scaler_path = f"{run_dir}/scaler_{ts}_seed{seed}.npz"
        np.savez(
            scaler_path,
            mean=scaler_mean.astype(np.float32),
            std=scaler_std.astype(np.float32),
        )
        out["artifacts"]["scaler"] = scaler_path

    split_path = f"{run_dir}/split_{ts}_seed{seed}.npz"
    np.savez(
        split_path,
        X_train_shape=np.asarray(X_train.shape),
        X_test_shape=np.asarray(X_test.shape),
        y_train=y_train,
        y_test=y_test,
    )
    out["artifacts"]["split"] = split_path

    return out


def run_multiseed(cfg: RunConfig, seeds: List[int]) -> Dict[str, Any]:
    """"""
    Run multiple seeds and aggregate mean curves.
    """"""
    per_seed = []
    for s in seeds:
        cfg_s = cfg
        cfg_s.train.seed = int(s)
        per_seed.append(train_one_run(cfg_s))

    # aggregate
    acc_curves = [r["history"]["test_acc"] for r in per_seed]
    auc_curves = [r["history"]["test_auc"] for r in per_seed]
    nnz_curves = [r["history"]["nnz_theta"] for r in per_seed]

    mean_acc = np.mean(np.asarray(acc_curves, dtype=float), axis=0).tolist()
    mean_auc = np.mean(np.asarray(auc_curves, dtype=float), axis=0).tolist()
    mean_nnz = np.mean(np.asarray(nnz_curves, dtype=float), axis=0).tolist()

    run_dir = results_subdir(cfg.results_dir, cfg.run_name)
    ts = timestamp()
    out = {
        "config": cfg.to_dict(),
        "seeds": seeds,
        "mean": {
            "test_acc": mean_acc,
            "test_auc": mean_auc,
            "nnz_theta": mean_nnz,
        },
        "per_seed": [
            {
                "seed": r["seed"],
                "final": r["final"],
                "artifacts": r["artifacts"],
                "history": r["history"],
            }
            for r in per_seed
        ],
    }

    json_path = f"{run_dir}/multiseed_{ts}.json"
    save_json(json_path, out)
    out["artifacts"] = {"json": json_path}
    return out
"""
