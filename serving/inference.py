"""
Inference service untuk Banking Router + RAG + LLM escalation.
Load model dari artifacts/, serve via FastAPI.

Setup:
    1. pip install fastapi uvicorn scikit-learn sentence-transformers faiss-cpu
    2. Pastikan artifacts/ berisi model.pkl + encoder_name.txt
    3. python inference.py

Env vars (opsional):
    MODEL_DIR        path ke artifacts/      (default: artifacts)
    LLM_PROVIDER     openai | groq | ollama | mock  (default: mock)
    OPENAI_API_KEY   sk-...
    OPENAI_MODEL     gpt-4o-mini             (default)
    OLLAMA_BASE_URL  http://localhost:11434  (default)
    OLLAMA_MODEL     llama3.2               (default)

Endpoint:
    GET  /          — Chatbot UI
    POST /predict   — {"text": "I lost my card"}
    GET  /health    — status model
"""

import os
import pickle
import time
import logging
from collections import deque
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, start_http_server,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Prometheus metrics ──────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "router_requests_total", "Total inference requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "router_request_latency_seconds", "Request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTION_CONFIDENCE = Histogram(
    "router_prediction_confidence", "Distribution of max prediction confidence",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
)
ROUTING_DECISIONS = Counter(
    "router_routing_decisions_total", "Total routing decisions by tier", ["tier"],
)
INTENT_PREDICTIONS = Counter(
    "router_intent_predictions_total", "Predicted intent class distribution", ["intent"],
)
INPUT_TEXT_LENGTH = Histogram(
    "router_input_text_length_chars", "Length of input text in characters",
    buckets=(10, 25, 50, 100, 200, 500, 1000),
)
ACTIVE_REQUESTS = Gauge("router_active_requests", "Number of in-flight requests")
ESTIMATED_COST_USD = Counter(
    "router_estimated_cost_usd_total", "Cumulative estimated inference cost in USD", ["tier"],
)
LOW_CONFIDENCE_RATE = Gauge(
    "router_low_confidence_rate", "Rolling rate of low-confidence predictions",
)
MODEL_INFO = Gauge(
    "router_model_info", "Model metadata", ["model_version", "model_type"],
)
PREDICTION_ERRORS = Counter(
    "router_prediction_errors_total", "Total prediction errors", ["error_type"],
)
THROUGHPUT_SUMMARY = Summary("router_throughput_summary", "Summary of throughput")

COST_MAP = {"template_handler": 0.001, "rag_pipeline": 0.010, "llm_escalation": 0.050}
_recent_confidences: deque = deque(maxlen=100)

MODEL_DIR = os.getenv("MODEL_DIR", "artifacts")
FAQ_PATH  = os.path.join(os.path.dirname(__file__), "banking_faq.json")

from llm_client import LLMClient
llm_client = LLMClient()

HIGH_THR = 0.55
MID_THR  = 0.30

# ── Conversational pre-filter (tier 0) ─────────────────────────────────────
_GREETINGS = {"hi", "hello", "hey", "halo", "hai", "good morning", "good afternoon",
              "good evening", "selamat pagi", "selamat siang", "selamat malam", "howdy"}
_FAREWELLS = {"bye", "goodbye", "bye bye", "see you", "ciao", "sampai jumpa", "dadah",
              "thanks", "thank you", "thanks you", "thank u", "thx", "ty",
              "terima kasih", "makasih", "tengkyu"}

_CONV_RESPONSES = {
    "greeting": "Hello! Welcome to Banking Support 🏦 How can I help you today?",
    "farewell": "You're welcome! Is there anything else I can help you with? Have a great day! 😊",
}

def detect_conversational(text: str) -> Optional[str]:
    normalized = text.lower().strip().rstrip("!.,? ")
    if normalized in _GREETINGS or any(normalized.startswith(g) for g in _GREETINGS):
        return "greeting"
    if normalized in _FAREWELLS or any(normalized.startswith(f) for f in _FAREWELLS):
        return "farewell"
    return None


def route_decision(confidence: float) -> str:
    if confidence >= HIGH_THR:
        return "template_handler"
    elif confidence >= MID_THR:
        return "rag_pipeline"
    return "llm_escalation"


class BankingRouter:
    def __init__(self):
        self.model   = None
        self.encoder = None
        self.rag     = None

    def load(self, model_dir: str):
        model_path        = os.path.join(model_dir, "model.pkl")
        encoder_name_path = os.path.join(model_dir, "encoder_name.txt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.pkl tidak ada di {model_dir}")
        if not os.path.exists(encoder_name_path):
            raise FileNotFoundError(f"encoder_name.txt tidak ada di {model_dir}")

        encoder_name = open(encoder_name_path).read().strip()

        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(encoder_name)
        logger.info(f"Encoder loaded: {encoder_name}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Classifier loaded: {type(self.model).__name__}")

        # Load RAG jika faiss tersedia dan FAQ ada
        if os.path.exists(FAQ_PATH):
            try:
                from rag import BankingRAG
                self.rag = BankingRAG(faq_path=FAQ_PATH, encoder=self.encoder)
                logger.info("RAG module loaded")
            except ImportError:
                logger.warning("faiss-cpu tidak terinstall — RAG disabled. pip install faiss-cpu")
        else:
            logger.warning(f"FAQ tidak ditemukan di {FAQ_PATH} — RAG disabled")

    def predict(self, text: str):
        embedding  = self.encoder.encode([text], normalize_embeddings=True)
        proba      = self.model.predict_proba(embedding)[0]
        confidence = float(proba.max())
        intent_idx = int(proba.argmax())
        return intent_idx, confidence


router_model = BankingRouter()


# ── Prometheus remote_write helpers (manual protobuf + snappy) ─────────────
import struct, math

def _varint(n: int) -> bytes:
    out = []
    while n > 0x7f:
        out.append((n & 0x7f) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)

def _pb_bytes(field: int, data: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(data)) + data

def _pb_str(field: int, s: str) -> bytes:
    return _pb_bytes(field, s.encode())

def _pb_double(field: int, v: float) -> bytes:
    return _varint((field << 3) | 1) + struct.pack("<d", v)

def _pb_int64(field: int, v: int) -> bytes:
    return _varint((field << 3) | 0) + _varint(v & 0xFFFFFFFFFFFFFFFF)

def _build_write_request(samples: list) -> bytes:
    ts_ms = int(time.time() * 1000)
    body = b""
    for s in samples:
        ts = b""
        for k, v in [("__name__", s["name"])] + sorted(s.get("labels", {}).items()):
            ts += _pb_bytes(1, _pb_str(1, k) + _pb_str(2, v))
        ts += _pb_bytes(2, _pb_double(1, s["value"]) + _pb_int64(2, s.get("timestamp", ts_ms)))
        body += _pb_bytes(1, ts)
    return body


def _collect_router_samples() -> list:
    result = []
    ts_ms = int(time.time() * 1000)
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if not sample.name.startswith("router_"):
                continue
            if sample.name.endswith("_created"):
                continue
            if math.isnan(sample.value) or math.isinf(sample.value):
                continue
            result.append({
                "name":      sample.name,
                "value":     float(sample.value),
                "labels":    dict(sample.labels),
                "timestamp": ts_ms,
            })
    return result


async def _grafana_push_loop(url: str, username: str, password: str, interval: int = 15):
    import asyncio, cramjam
    while True:
        await asyncio.sleep(interval)
        try:
            samples = _collect_router_samples()
            if not samples:
                continue
            payload    = _build_write_request(samples)
            compressed = bytes(cramjam.snappy.compress_raw(payload))
            headers = {
                "Content-Encoding":                   "snappy",
                "Content-Type":                       "application/x-protobuf",
                "X-Prometheus-Remote-Write-Version":  "0.1.0",
            }
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    url, content=compressed, headers=headers,
                    auth=(username, password), timeout=10.0,
                )
                r.raise_for_status()
                logger.debug(f"Grafana Cloud push OK — {len(samples)} series")
        except Exception as e:
            logger.warning(f"Grafana Cloud push failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    try:
        router_model.load(MODEL_DIR)
        MODEL_INFO.labels(model_version="v1.0", model_type="SBERT+LinearSVC").set(1)
    except FileNotFoundError as e:
        logger.warning(f"Model belum ada: {e}. Jalankan training dulu.")

    # Local dev: buka port terpisah untuk Prometheus scrape
    metrics_port = os.getenv("METRICS_PORT")
    if metrics_port:
        try:
            start_http_server(int(metrics_port))
            logger.info(f"Prometheus metrics exposed at :{metrics_port}/metrics")
        except OSError:
            logger.warning(f"Port {metrics_port} sudah dipakai — skip separate metrics server")
    else:
        logger.info("Metrics available at /metrics on main port")

    # Production: push ke Grafana Cloud jika env var di-set
    grafana_url  = os.getenv("GRAFANA_REMOTE_WRITE_URL")
    grafana_user = os.getenv("GRAFANA_USERNAME")
    grafana_pass = os.getenv("GRAFANA_PASSWORD")
    if grafana_url and grafana_user and grafana_pass:
        asyncio.create_task(_grafana_push_loop(grafana_url, grafana_user, grafana_pass))
        logger.info(f"Grafana Cloud push aktif → {grafana_url}")

    yield


app = FastAPI(title="Banking Intent Router", lifespan=lifespan)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def frontend():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class PredictRequest(BaseModel):
    text: str


class RAGSource(BaseModel):
    question: str
    score: float


class PredictResponse(BaseModel):
    intent_id:             int
    confidence:            float
    routing:               str
    latency_ms:            float
    rag_answer:            Optional[str]    = None
    rag_sources:           list[RAGSource]  = []
    rag_score:             Optional[float]  = None
    llm_answer:            Optional[str]    = None
    conversational_intent: Optional[str]    = None


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if router_model.model is None:
        raise HTTPException(status_code=503, detail="Model belum loaded. Jalankan training dulu.")

    ACTIVE_REQUESTS.inc()
    INPUT_TEXT_LENGTH.observe(len(req.text))
    start = time.time()

    try:
        # Tier 0: conversational pre-filter (bypass ML model)
        conv_intent = detect_conversational(req.text)
        if conv_intent:
            latency_ms = (time.time() - start) * 1000
            ROUTING_DECISIONS.labels(tier="conversational_handler").inc()
            ESTIMATED_COST_USD.labels(tier="conversational_handler").inc(0.0)
            return PredictResponse(
                intent_id=-1,
                confidence=1.0,
                routing="conversational_handler",
                latency_ms=latency_ms,
                rag_answer=_CONV_RESPONSES[conv_intent],
                conversational_intent=conv_intent,
            )

        intent_id, confidence = router_model.predict(req.text)
        routing = route_decision(confidence)

        PREDICTION_CONFIDENCE.observe(confidence)
        ROUTING_DECISIONS.labels(tier=routing).inc()
        INTENT_PREDICTIONS.labels(intent=f"intent_{intent_id}").inc()
        ESTIMATED_COST_USD.labels(tier=routing).inc(COST_MAP[routing])

        _recent_confidences.append(confidence)
        low_rate = sum(1 for c in _recent_confidences if c < 0.6) / len(_recent_confidences)
        LOW_CONFIDENCE_RATE.set(low_rate)

        rag_answer  = None
        rag_sources = []
        rag_score   = None
        llm_answer  = None

        if routing in ("template_handler", "rag_pipeline") and router_model.rag is not None:
            top_k       = 1 if routing == "template_handler" else 3
            result      = router_model.rag.answer(req.text, top_k=top_k)
            rag_answer  = result["answer"]
            rag_score   = result["top_score"]
            if routing == "rag_pipeline":
                rag_sources = [RAGSource(question=s["question"], score=s["score"]) for s in result["sources"]]

        elif routing == "llm_escalation":
            try:
                llm_answer = await llm_client.generate(req.text, intent_id, confidence)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                llm_answer = "Our team is looking into this. Please contact support for immediate assistance."

        latency = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        THROUGHPUT_SUMMARY.observe(latency)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()

        return PredictResponse(
            intent_id=intent_id,
            confidence=confidence,
            routing=routing,
            latency_ms=latency * 1000,
            rag_answer=rag_answer,
            rag_sources=rag_sources,
            rag_score=rag_score,
            llm_answer=llm_answer,
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  router_model.model is not None,
        "rag_loaded":    router_model.rag is not None,
        "llm_provider":  llm_client.provider,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
