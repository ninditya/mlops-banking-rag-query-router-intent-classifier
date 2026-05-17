"""
Prometheus exporter untuk Banking Router serving.
Target K4 Advance: 12 metrik untuk Grafana monitoring.

Setup:
    1. pip install fastapi uvicorn prometheus-client scikit-learn
    2. Letakkan model.pkl + encoder_name.txt di folder artifacts/
       (atau set MODEL_DIR env var)
    3. python prometheus_exporter.py
       — Inference API  : http://localhost:8001/predict
       — Prometheus metrics : http://localhost:8000/metrics
"""

import os
import pickle
import time
import random
import logging
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    start_http_server,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "artifacts")

# ============================================================
# 12 METRIK UNTUK K4 ADVANCE (>= 10 metrics)
# ============================================================

REQUEST_COUNT = Counter(
    "router_requests_total",
    "Total inference requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "router_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_CONFIDENCE = Histogram(
    "router_prediction_confidence",
    "Distribution of max prediction confidence",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
)

ROUTING_DECISIONS = Counter(
    "router_routing_decisions_total",
    "Total routing decisions by tier",
    ["tier"],
)

INTENT_PREDICTIONS = Counter(
    "router_intent_predictions_total",
    "Predicted intent class distribution",
    ["intent"],
)

INPUT_TEXT_LENGTH = Histogram(
    "router_input_text_length_chars",
    "Length of input text in characters",
    buckets=(10, 25, 50, 100, 200, 500, 1000),
)

ACTIVE_REQUESTS = Gauge(
    "router_active_requests",
    "Number of in-flight requests",
)

ESTIMATED_COST_USD = Counter(
    "router_estimated_cost_usd_total",
    "Cumulative estimated inference cost in USD",
    ["tier"],
)

LOW_CONFIDENCE_RATE = Gauge(
    "router_low_confidence_rate",
    "Rolling rate of low-confidence predictions (proxy for drift)",
)

MODEL_INFO = Gauge(
    "router_model_info",
    "Model metadata",
    ["model_version", "model_type"],
)

PREDICTION_ERRORS = Counter(
    "router_prediction_errors_total",
    "Total prediction errors",
    ["error_type"],
)

THROUGHPUT_SUMMARY = Summary(
    "router_throughput_summary",
    "Summary of throughput",
)

COST_MAP = {"template_handler": 0.001, "rag_pipeline": 0.01, "llm_escalation": 0.05}

# deque dengan maxlen menggantikan list + pop(0) manual — O(1) vs O(n)
_recent_confidences: deque = deque(maxlen=100)


def route_decision(confidence: float) -> str:
    if confidence >= 0.55:
        return "template_handler"
    elif confidence >= 0.30:
        return "rag_pipeline"
    return "llm_escalation"


# ============================================================
# FASTAPI APP
# ============================================================
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    routing: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_http_server(8000)
    logger.info("Prometheus metrics exposed at :8000/metrics")

    # Coba load model real; fallback ke mock jika belum ada
    try:
        encoder_name_path = os.path.join(MODEL_DIR, "encoder_name.txt")
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        encoder_name = open(encoder_name_path).read().strip()
        from sentence_transformers import SentenceTransformer
        app.state.encoder = SentenceTransformer(encoder_name)
        with open(model_path, "rb") as f:
            app.state.model = pickle.load(f)
        logger.info(f"Model loaded: encoder={encoder_name}")
    except FileNotFoundError:
        app.state.model = None
        app.state.encoder = None
        logger.warning("Model tidak ditemukan — berjalan dalam MOCK mode")

    MODEL_INFO.labels(model_version="v1.0", model_type="SBERT+LinearSVC").set(1)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        INPUT_TEXT_LENGTH.observe(len(req.text))

        if app.state.model is not None:
            embedding = app.state.encoder.encode([req.text], normalize_embeddings=True)
            proba = app.state.model.predict_proba(embedding)[0]
            confidence = float(proba.max())
            intent = f"intent_{int(proba.argmax())}"
        else:
            # Mock mode untuk demo monitoring tanpa model
            confidence = random.uniform(0.3, 0.99)
            intent = f"intent_{random.randint(0, 76)}"

        PREDICTION_CONFIDENCE.observe(confidence)
        INTENT_PREDICTIONS.labels(intent=intent).inc()

        tier = route_decision(confidence)
        ROUTING_DECISIONS.labels(tier=tier).inc()
        ESTIMATED_COST_USD.labels(tier=tier).inc(COST_MAP[tier])

        _recent_confidences.append(confidence)
        low_rate = sum(1 for c in _recent_confidences if c < 0.6) / len(_recent_confidences)
        LOW_CONFIDENCE_RATE.set(low_rate)

        latency = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        THROUGHPUT_SUMMARY.observe(latency)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()

        return PredictResponse(
            intent=intent,
            confidence=confidence,
            routing=tier,
            latency_ms=latency * 1000,
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
