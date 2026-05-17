"""
Hyperparameter tuning untuk Banking77 Intent Router.
GridSearchCV mencari C optimal untuk LogisticRegression + CalibratedClassifierCV.

Cara pakai:
    python modelling_tuning.py
    python modelling_tuning.py --data_path banking77_preprocessing --cv 3

Output: best_params.json + MLflow run dengan semua hasil
"""

import argparse
import json
import os
import pickle
import warnings
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

from routing_utils import route_decision, estimate_cost_savings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="banking77_preprocessing")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))

    vectorizer_path = os.path.join(args.data_path, "vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        X_train = vectorizer.transform(train_df["text"])
        X_test = vectorizer.transform(test_df["text"])
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True)
        X_train = vectorizer.fit_transform(train_df["text"])
        X_test = vectorizer.transform(test_df["text"])

    y_train, y_test = train_df["label"].values, test_df["label"].values

    # Grid: LR + SVM
    lr_configs = [
        {"C": C, "solver": solver}
        for solver in ["lbfgs", "saga"]
        for C in [0.5, 1, 5, 10, 50, 100]
    ]
    svm_configs = [{"C": C} for C in [0.1, 0.5, 1, 5, 10, 50]]

    total = len(lr_configs) + len(svm_configs)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("banking77-intent-router-tuning")

    print(f"\n{'='*60}")
    print(f"  LR configs  : {len(lr_configs)}")
    print(f"  SVM configs : {len(svm_configs)}")
    print(f"  Total       : {total} kombinasi  (CV folds={args.cv})")
    print(f"{'='*60}\n")

    results = []
    best_f1 = 0
    best_config = None

    def run_combo(model_type, base_model, label, C, extra_params=None):
        nonlocal best_f1, best_config
        with mlflow.start_run(run_name=f"tune_{label}"):
            print(f"  [{model_type}] {label} ...", end=" ", flush=True)
            t0 = time.time()

            calib_method = "isotonic" if model_type == "LR" else "sigmoid"
            model = CalibratedClassifierCV(base_model, cv=args.cv, method=calib_method)

            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)
            cv_f1_mean = float(cv_scores.mean())
            cv_f1_std = float(cv_scores.std())

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            test_f1 = f1_score(y_test, y_pred, average="macro")
            test_acc = accuracy_score(y_test, y_pred)
            test_ll = log_loss(y_test, y_proba)

            max_conf = y_proba.max(axis=1)
            routing = pd.Series([route_decision(c) for c in max_conf])
            cost_stats = estimate_cost_savings(routing)

            elapsed = time.time() - t0

            mlflow.log_param("model_type", model_type)
            mlflow.log_param("C", C)
            if extra_params:
                for k, v in extra_params.items():
                    mlflow.log_param(k, v)
            mlflow.log_metric("cv_f1_macro_mean", cv_f1_mean)
            mlflow.log_metric("cv_f1_macro_std", cv_f1_std)
            mlflow.log_metric("test_f1_macro", test_f1)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_log_loss", test_ll)
            mlflow.log_metric("cost_savings_pct", cost_stats["savings_percent"])
            mlflow.log_metric("template_pct", cost_stats["template_pct"])
            mlflow.log_metric("training_time_s", elapsed)

            row = {
                "model": model_type,
                "C": C,
                **(extra_params or {}),
                "cv_f1_mean": round(cv_f1_mean, 4),
                "cv_f1_std": round(cv_f1_std, 4),
                "test_f1": round(test_f1, 4),
                "test_acc": round(test_acc, 4),
                "cost_savings": round(cost_stats["savings_percent"], 2),
                "template_pct": round(cost_stats["template_pct"], 2),
                "time_s": round(elapsed, 1),
            }
            results.append(row)

            if test_f1 > best_f1:
                best_f1 = test_f1
                best_config = row.copy()

            print(f"F1={test_f1:.4f}  acc={test_acc:.4f}  savings={cost_stats['savings_percent']}%  ({elapsed:.1f}s)")

    print("--- Logistic Regression ---")
    for cfg in lr_configs:
        base = LogisticRegression(C=cfg["C"], solver=cfg["solver"], max_iter=1000, class_weight="balanced")
        run_combo("LR", base, f"C{cfg['C']}_{cfg['solver']}", cfg["C"], {"solver": cfg["solver"]})

    print("\n--- LinearSVC ---")
    for cfg in svm_configs:
        base = LinearSVC(C=cfg["C"], max_iter=3000, class_weight="balanced", dual=True)
        run_combo("SVM", base, f"C{cfg['C']}", cfg["C"])

    # Tampilkan tabel hasil
    print(f"\n{'='*88}")
    print(f"{'Model':>5} {'C':>6} {'Solver':>7} {'CV-F1':>8} {'Test-F1':>8} {'Acc':>8} {'Savings%':>9} {'Tmpl%':>7}")
    print(f"{'-'*88}")
    for r in sorted(results, key=lambda x: x["test_f1"], reverse=True):
        is_best = r["C"] == best_config["C"] and r["model"] == best_config["model"]
        marker = " ← BEST" if is_best else ""
        solver_col = r.get("solver", "-")
        print(f"{r['model']:>5} {r['C']:>6} {solver_col:>7} {r['cv_f1_mean']:>8.4f} {r['test_f1']:>8.4f} "
              f"{r['test_acc']:>8.4f} {r['cost_savings']:>9.2f} {r['template_pct']:>7.2f}{marker}")
    print(f"{'='*88}")

    print(f"\n  Best config: model={best_config['model']}, C={best_config['C']}"
          + (f", solver={best_config.get('solver', '-')}" if "solver" in best_config else ""))
    print(f"  F1 Macro   : {best_config['test_f1']}")
    print(f"  Accuracy   : {best_config['test_acc']}")
    print(f"  Cost savings: {best_config['cost_savings']}%")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/best_params.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  Saved → artifacts/best_params.json")


if __name__ == "__main__":
    main()
