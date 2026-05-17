"""
Production training script untuk Banking77 Intent Router.
Dipanggil oleh MLflow Project (ci.yml via `mlflow run .`).
Hyperparameter di-pass dari MLProject entry point.

Model: SBERT (all-MiniLM-L6-v2) + LinearSVC + CalibratedClassifierCV
"""

import argparse
import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    log_loss,
)

warnings.filterwarnings("ignore")

ENCODER_NAME = "all-MiniLM-L6-v2"

from routing_utils import route_decision, estimate_cost_savings, HIGH_THR, MID_THR


def plot_confusion_matrix(y_true, y_pred, output_path: str, top_n: int = 20):
    top_intents = pd.Series(y_true).value_counts().head(top_n).index
    mask = np.isin(y_true, top_intents)
    cm = confusion_matrix(np.array(y_true)[mask], np.array(y_pred)[mask], labels=top_intents)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=top_intents, yticklabels=top_intents)
    plt.title(f"Confusion Matrix (Top {top_n} Intents)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def plot_confidence_distribution(probas: np.ndarray, output_path: str):
    max_conf = probas.max(axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(max_conf, bins=50, color="steelblue", edgecolor="black")
    plt.axvline(HIGH_THR, color="green", linestyle="--", label=f"High threshold ({HIGH_THR})")
    plt.axvline(MID_THR, color="orange", linestyle="--", label=f"Mid threshold ({MID_THR})")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Max Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=5.0, help="Regularization parameter for LinearSVC")
    parser.add_argument("--data_path", type=str, default="banking77_preprocessing")
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))

    from sentence_transformers import SentenceTransformer
    print(f"  Encoding dengan {ENCODER_NAME} ...")
    encoder = SentenceTransformer(ENCODER_NAME)
    X_train = encoder.encode(train_df["text"].tolist(), normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    X_test = encoder.encode(test_df["text"].tolist(), normalize_embeddings=True, batch_size=256, show_progress_bar=True)

    y_train, y_test = train_df["label"].values, test_df["label"].values

    via_mlflow_run = bool(os.environ.get("MLFLOW_RUN_ID"))
    if not via_mlflow_run:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment("banking77-intent-router")
        mlflow.start_run(run_name="ci_training")

    try:
        base_model = LinearSVC(C=args.C, max_iter=3000, class_weight="balanced", dual=True)
        model = CalibratedClassifierCV(base_model, cv=3, method="sigmoid")
        model.fit(X_train, y_train)

        if not via_mlflow_run:
            mlflow.log_param("C", args.C)
        mlflow.log_param("encoder", ENCODER_NAME)
        mlflow.log_param("model_type", "LinearSVC")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y_train)))

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        ll = log_loss(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec)
        mlflow.log_metric("recall_macro", rec)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("log_loss", ll)

        max_conf = y_proba.max(axis=1)
        routing = pd.Series([route_decision(c) for c in max_conf])
        cost_stats = estimate_cost_savings(routing)
        for k, v in cost_stats.items():
            mlflow.log_metric(f"router_{k}", v)

        os.makedirs("artifacts", exist_ok=True)

        plot_confusion_matrix(y_test, y_pred, "artifacts/confusion_matrix.png")
        mlflow.log_artifact("artifacts/confusion_matrix.png")

        report = classification_report(y_test, y_pred, output_dict=True)
        with open("artifacts/classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("artifacts/classification_report.json")

        plot_confidence_distribution(y_proba, "artifacts/confidence_distribution.png")
        mlflow.log_artifact("artifacts/confidence_distribution.png")

        router_df = pd.DataFrame({
            "text": test_df["text"],
            "predicted_intent": y_pred,
            "confidence": max_conf,
            "routing_decision": routing,
        })
        router_df.to_csv("artifacts/router_decisions.csv", index=False)
        mlflow.log_artifact("artifacts/router_decisions.csv")

        input_example = X_test[:3]
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact("artifacts/model.pkl")

        with open("artifacts/encoder_name.txt", "w") as f:
            f.write(ENCODER_NAME)
        mlflow.log_artifact("artifacts/encoder_name.txt")

        run = mlflow.active_run()
        run_id_final = run.info.run_id if run else os.environ.get("MLFLOW_RUN_ID", "unknown")
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run_id_final)

        print(f"\n✅ Training done")
        print(f"   Encoder       : {ENCODER_NAME}")
        print(f"   F1 Macro      : {f1:.4f}")
        print(f"   Accuracy      : {acc:.4f}")
        print(f"   Cost savings  : {cost_stats['savings_percent']}%")
        print(f"   Run ID        : {run_id_final}")

    finally:
        if not via_mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
