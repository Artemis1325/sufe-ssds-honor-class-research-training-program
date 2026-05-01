# src/run_lassonet_breast_cancer.py
# LassoNet baseline
import os
import numpy as np
import json
import time
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from lassonet import LassoNetClassifierCV

os.makedirs("../results/run_grid_py_multiseed_results", exist_ok=True)

def run_once(seed):
    """Run LassoNet one time with given seed."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # define model
    model = LassoNetClassifierCV(
        hidden_dims=(100,),
        path_multiplier=1.1,
        n_iters=100,
        patience=10,
        val_size=0.2
    )

    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    nnz = len(model.best_selected_)

    return {
        "acc": acc,
        "auc": auc,
        "nnz": nnz,
        "best_lambda": float(model.best_lambda_)
    }


def main():
    seeds = list(range(1, 21))  # 20 runs
    results = []

    start_time = time.time()

    for s in seeds:
        print(f"=== Running seed {s} ===")
        res = run_once(s)
        res["seed"] = s
        results.append(res)

    elapsed = time.time() - start_time

    # aggregate
    mean_acc = np.mean([r["acc"] for r in results])
    mean_auc = np.mean([r["auc"] for r in results])
    mean_nnz = np.mean([r["nnz"] for r in results])

    std_acc = np.std([r["acc"] for r in results])
    std_auc = np.std([r["auc"] for r in results])
    std_nnz = np.std([r["nnz"] for r in results])

    summary = {
        "mean_acc": float(mean_acc),
        "mean_auc": float(mean_auc),
        "mean_nnz": float(mean_nnz),
        "std_acc": float(std_acc),
        "std_auc": float(std_auc),
        "std_nnz": float(std_nnz),
        "seeds": seeds,
        "all_results": results,
        "elapsed_seconds": elapsed
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"../results/run_grid_py_multiseed_results/lassonet_multiseed_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()

