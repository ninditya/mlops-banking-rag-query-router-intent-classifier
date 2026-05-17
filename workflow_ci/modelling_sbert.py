"""
Sentence-Transformers experiment untuk Banking77.
Ganti TF-IDF dengan dense neural embeddings → target F1 ~0.92-0.94.

Setup:
    pip install sentence-transformers

Cara pakai:
    python modelling_sbert.py
    python modelling_sbert.py --data_path banking77_preprocessing --save_best

Jika --save_best: simpan encoder + model ke artifacts/ untuk dipakai inference.
"""

import argparse
import json
import os
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss

warnings.filterwarnings("ignore")

SBERT_MODEL = "all-MiniLM-L6-v2"  # 384 dim, ~90MB, fast + accurate

from routing_utils import route_decision, estimate_cost_savings


def encode_texts(encoder, texts: list, batch_size: int = 256) -> np.ndarray:
    print(f"  Encoding {len(texts):,} texts (batch={batch_size}) ...", flush=True)
    t0 = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time()-t0:.1f}s  shape={embeddings.shape}")
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="banking77_preprocessing")
    parser.add_argument("--save_best", action="store_true", help="Simpan model terbaik ke artifacts/")
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n  ERROR: sentence-transformers belum terinstall.")
        print("  Jalankan: pip install sentence-transformers\n")
        return

    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    print(f"\n{'='*60}")
    print(f"  Model encoder : {SBERT_MODEL}")
    print(f"  Train samples : {len(train_df):,}")
    print(f"  Test samples  : {len(test_df):,}")
    print(f"{'='*60}\n")

    print(f"  Loading encoder ...", flush=True)
    encoder = SentenceTransformer(SBERT_MODEL)

    X_train = encode_texts(encoder, train_df["text"].tolist())
    X_test = encode_texts(encoder, test_df["text"].tolist())

    configs = [
        {"model": "SVM", "C": 0.5},
        {"model": "SVM", "C": 1},
        {"model": "SVM", "C": 5},
        {"model": "SVM", "C": 10},
        {"model": "LR",  "C": 1,  "solver": "lbfgs"},
        {"model": "LR",  "C": 10, "solver": "lbfgs"},
        {"model": "LR",  "C": 50, "solver": "lbfgs"},
    ]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("banking77-sbert-tuning")

    print(f"\n  Testing {len(configs)} configs ...\n")

    results = []
    best_f1 = 0
    best_config = None
    best_model_obj = None

    for cfg in configs:
        label = f"SBERT+{cfg['model']}_C{cfg['C']}"
        if cfg["model"] == "LR":
            label += f"_{cfg.get('solver', 'lbfgs')}"

        with mlflow.start_run(run_name=label):
            print(f"  [{cfg['model']}] C={cfg['C']} ...", end=" ", flush=True)
            t0 = time.time()

            if cfg["model"] == "SVM":
                base = LinearSVC(C=cfg["C"], max_iter=3000, class_weight="balanced", dual=True)
                calib = "sigmoid"
            else:
                base = LogisticRegression(
                    C=cfg["C"], solver=cfg.get("solver", "lbfgs"),
                    max_iter=1000, class_weight="balanced"
                )
                calib = "isotonic"

            model = CalibratedClassifierCV(base, cv=args.cv, method=calib)
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

            mlflow.log_param("encoder", SBERT_MODEL)
            mlflow.log_param("model_type", cfg["model"])
            mlflow.log_param("C", cfg["C"])
            if "solver" in cfg:
                mlflow.log_param("solver", cfg["solver"])
            mlflow.log_metric("test_f1_macro", test_f1)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_log_loss", test_ll)
            mlflow.log_metric("cost_savings_pct", cost_stats["savings_percent"])
            mlflow.log_metric("template_pct", cost_stats["template_pct"])

            row = {
                "model": f"SBERT+{cfg['model']}",
                "C": cfg["C"],
                "test_f1": round(test_f1, 4),
                "test_acc": round(test_acc, 4),
                "cost_savings": cost_stats["savings_percent"],
                "template_pct": cost_stats["template_pct"],
                "time_s": round(elapsed, 1),
            }
            results.append(row)

            if test_f1 > best_f1:
                best_f1 = test_f1
                best_config = {**cfg, "test_f1": round(test_f1, 4), "test_acc": round(test_acc, 4),
                               "cost_savings": cost_stats["savings_percent"], "encoder": SBERT_MODEL}
                best_model_obj = model

            print(f"F1={test_f1:.4f}  acc={test_acc:.4f}  savings={cost_stats['savings_percent']}%  ({elapsed:.1f}s)")

    # Tabel hasil
    print(f"\n{'='*78}")
    print(f"{'Model':>12} {'C':>5} {'Test-F1':>8} {'Acc':>8} {'Savings%':>9} {'Tmpl%':>7}")
    print(f"{'-'*78}")
    for r in sorted(results, key=lambda x: x["test_f1"], reverse=True):
        is_best = r["test_f1"] == best_config["test_f1"]
        marker = " ← BEST" if is_best else ""
        print(f"{r['model']:>12} {r['C']:>5} {r['test_f1']:>8.4f} {r['test_acc']:>8.4f} "
              f"{r['cost_savings']:>9.2f} {r['template_pct']:>7.2f}{marker}")
    print(f"{'='*78}")

    print(f"\n  Best: {best_config['model']} C={best_config['C']}")
    print(f"  F1 Macro    : {best_config['test_f1']}")
    print(f"  Accuracy    : {best_config['test_acc']}")
    print(f"  Cost savings: {best_config['cost_savings']}%")

    # Bandingkan dengan TF-IDF + SVM baseline
    tfidf_f1 = 0.8843
    delta = best_config["test_f1"] - tfidf_f1
    print(f"\n  vs TF-IDF+SVM (baseline): {tfidf_f1} → {'+' if delta >= 0 else ''}{delta:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/best_params_sbert.json", "w") as f:
        json.dump(best_config, f, indent=2)

    if args.save_best and best_model_obj is not None:
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(best_model_obj, f)
        with open("artifacts/encoder_name.txt", "w") as f:
            f.write(SBERT_MODEL)
        print(f"\n  Saved artifacts/model.pkl + artifacts/encoder_name.txt")
        print(f"  Inference sekarang pakai SBERT encoder, bukan TF-IDF vectorizer.")


if __name__ == "__main__":
    main()
