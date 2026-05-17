"""
Modelling Banking77 Intent Router dengan hyperparameter tuning + DagsHub.

K2 Advance:
- MLflow tracking ke DagsHub (online)
- Manual logging (bukan autolog)
- Minimal 2 artefak tambahan di luar autolog:
    1. confusion_matrix.png
    2. classification_report.json
    3. router_decisions.csv (BONUS: angle unik proyek ini)
    4. confidence_distribution.png

Usage:
    python modelling_tuning.py
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ============================================================
# DAGSHUB SETUP (K2 Advance: tracking online)
# ============================================================
# Set ini di environment / .env / GitHub Secrets:
# MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
# MLFLOW_TRACKING_USERNAME=<username>
# MLFLOW_TRACKING_PASSWORD=<dagshub-token>

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("banking77-intent-router")


# ============================================================
# ROUTING LOGIC (Unique Angle: Cost-Aware Router)
# ============================================================
def route_decision(confidence: float, high_thr: float = 0.55, mid_thr: float = 0.30) -> str:
    """3-tier routing berdasarkan confidence classifier."""
    if confidence >= high_thr:
        return "template_handler"  # Cheap, ~10ms
    elif confidence >= mid_thr:
        return "rag_pipeline"      # Medium, ~500ms
    else:
        return "llm_escalation"    # Expensive, ~2s


def estimate_cost_savings(routing_decisions: pd.Series) -> dict:
    """Hitung estimasi cost saving vs baseline (semua ke LLM besar)."""
    # Estimasi cost relatif: template=0.001, rag=0.01, llm=0.05 USD per query
    cost_map = {"template_handler": 0.001, "rag_pipeline": 0.01, "llm_escalation": 0.05}
    actual_cost = routing_decisions.map(cost_map).sum()
    baseline_cost = len(routing_decisions) * cost_map["llm_escalation"]
    savings_pct = (1 - actual_cost / baseline_cost) * 100
    return {
        "actual_cost_usd": round(actual_cost, 4),
        "baseline_cost_usd": round(baseline_cost, 4),
        "savings_percent": round(savings_pct, 2),
        "template_pct": round((routing_decisions == "template_handler").mean() * 100, 2),
        "rag_pct": round((routing_decisions == "rag_pipeline").mean() * 100, 2),
        "llm_pct": round((routing_decisions == "llm_escalation").mean() * 100, 2),
    }


# ============================================================
# PLOTTING HELPERS (untuk artefak tambahan)
# ============================================================
def plot_confusion_matrix(y_true, y_pred, output_path: str, top_n: int = 20):
    """Plot confusion matrix untuk top-N intent paling sering (77 kelas terlalu padat)."""
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
    """Distribusi max confidence per prediksi → menjustifikasi threshold routing."""
    max_conf = probas.max(axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(max_conf, bins=50, color="steelblue", edgecolor="black")
    plt.axvline(0.85, color="green", linestyle="--", label="High threshold (0.85)")
    plt.axvline(0.6, color="orange", linestyle="--", label="Mid threshold (0.6)")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Max Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    # Load data hasil preprocessing
    train_df = pd.read_csv("banking77_preprocessing/train.csv")
    test_df = pd.read_csv("banking77_preprocessing/test.csv")

    # Re-fit vectorizer (atau load dari pickle)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train, y_test = train_df["label"].values, test_df["label"].values

    # ===== HYPERPARAMETER TUNING (K2 Skilled+) =====
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
    }

    with mlflow.start_run(run_name="logreg_tuning"):
        grid = GridSearchCV(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            param_grid=param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # ===== MANUAL LOGGING (K2 Advance: bukan autolog) =====
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y_train)))

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        # Standard metrics (yang biasanya tercover autolog)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        ll = log_loss(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec)
        mlflow.log_metric("recall_macro", rec)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("log_loss", ll)
        mlflow.log_metric("cv_best_score", grid.best_score_)

        # ===== ROUTING METRICS (unique angle) =====
        max_conf = y_proba.max(axis=1)
        routing = pd.Series([route_decision(c) for c in max_conf])
        cost_stats = estimate_cost_savings(routing)

        for k, v in cost_stats.items():
            mlflow.log_metric(f"router_{k}", v)

        # ===== ARTEFAK TAMBAHAN (K2 Advance: minimal 2) =====
        os.makedirs("artifacts", exist_ok=True)

        # 1. Confusion matrix
        plot_confusion_matrix(y_test, y_pred, "artifacts/confusion_matrix.png")
        mlflow.log_artifact("artifacts/confusion_matrix.png")

        # 2. Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        with open("artifacts/classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("artifacts/classification_report.json")

        # 3. Confidence distribution
        plot_confidence_distribution(y_proba, "artifacts/confidence_distribution.png")
        mlflow.log_artifact("artifacts/confidence_distribution.png")

        # 4. Router decisions CSV (bonus angle)
        router_df = pd.DataFrame({
            "text": test_df["text"],
            "predicted_intent": y_pred,
            "confidence": max_conf,
            "routing_decision": routing,
        })
        router_df.to_csv("artifacts/router_decisions.csv", index=False)
        mlflow.log_artifact("artifacts/router_decisions.csv")

        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        with open("artifacts/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("artifacts/vectorizer.pkl")

        print(f"✅ Training done — F1: {f1:.4f}, Cost savings: {cost_stats['savings_percent']}%")


if __name__ == "__main__":
    main()
