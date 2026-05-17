# Banking Intent Router — Cost-Aware ML System

> Reduce banking chatbot LLM inference costs by ~96% using a confidence-based 4-tier routing system, without sacrificing answer quality.

[![Live Demo](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/ninditya/banking-intent-router)
[![Model](https://img.shields.io/badge/🤗%20Model-HuggingFace%20Hub-blue)](https://huggingface.co/ninditya/banking-router-model)

## Live Demo

| | Link |
|---|---|
| **Interactive demo** | [huggingface.co/spaces/ninditya/banking-intent-router](https://huggingface.co/spaces/ninditya/banking-intent-router) |
| **Model (SBERT + LinearSVC)** | [huggingface.co/ninditya/banking-router-model](https://huggingface.co/ninditya/banking-router-model) |

Type any banking question — the UI shows which tier handled it, the confidence score, latency, and the actual response.

---

## Problem

Chatbot banking yang mengirim **semua query ke LLM besar** membakar biaya 5–10x lebih dari yang seharusnya. Sebagian besar pertanyaan pelanggan sebenarnya repetitif dan bisa dijawab tanpa LLM sama sekali.

## Solution

Multi-class intent classifier (77 intents) yang menghasilkan **confidence score** sebagai sinyal routing:

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Conversational Pre-filter  │  ← Tier 0: greetings, farewells
│  (keyword match, <1ms)      │       $0.000/query
└─────────────┬───────────────┘
              │ (not conversational)
              ▼
┌─────────────────────────────┐
│   Intent Classifier         │
│   SBERT + LinearSVC         │
│   Banking77 (13K samples)   │
└─────────────┬───────────────┘
              │ confidence score
    ┌─────────┼──────────┐
    ▼         ▼          ▼
  ≥ 0.55   0.30–0.55   < 0.30
    │         │          │
Template    RAG         LLM
Handler   Pipeline   Escalation
$0.001    $0.010     $0.050
~10ms     ~500ms     ~2s
```

### Cost Comparison

| Scenario | Cost per 1000 queries | Latency |
|---|---|---|
| Baseline (all LLM) | $50.00 | ~2s |
| **This router** | **~$2.10** | **~50ms avg** |
| **Savings** | **~96%** | **~25x faster avg** |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   GitHub Actions CI                 │
│  push → preprocess → train → build Docker → push   │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      DagsHub      Docker Hub   GitHub
     (MLflow)      (Image)    (Artifacts)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│              Serving Stack (Docker)                 │
│  FastAPI (:8001) — inference + /metrics endpoint    │
│  Prometheus (:9090) + Grafana (:3000)               │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Modeling** — Sentence-Transformers (SBERT all-MiniLM-L6-v2) + LinearSVC, GridSearchCV
- **Experiment Tracking** — MLflow + DagsHub
- **CI/CD** — GitHub Actions
- **Serving** — FastAPI + Docker
- **LLM Fallback** — Groq (llama-3.1-8b-instant) / OpenAI / Ollama / Mock (configurable via env)
- **Monitoring** — Prometheus (12 custom metrics) + Grafana Cloud (push-based)

---

## Project Structure

```
├── eksperimen/
│   ├── Eksperimen_NindityaS.ipynb     # EDA + preprocessing manual
│   ├── automate_NindityaS.py          # Preprocessing pipeline
│   └── modelling_tuning.py            # Hyperparameter tuning + MLflow
│
├── workflow_ci/
│   ├── MLProject                      # MLflow project definition
│   ├── conda.yaml                     # Environment spec
│   ├── modelling.py                   # Production training (SBERT + LinearSVC)
│   ├── modelling_tuning.py            # Hyperparameter grid search (TF-IDF baseline)
│   ├── modelling_sbert.py             # SBERT tuning — found best: SVM C=5, F1=0.930
│   └── ci.yml                         # GitHub Actions workflow
│
├── serving/
│   ├── inference.py                   # FastAPI serving + chatbot UI
│   ├── rag.py                         # FAISS retrieval module
│   ├── banking_faq.json               # Knowledge base (59 Q&A)
│   └── static/index.html             # Chatbot frontend
│
├── monitoring/
│   ├── docker-compose.yml             # Prometheus + Grafana stack
│   ├── prometheus.yml                 # Scrape config
│   └── grafana_dashboard_cloud.json   # Dashboard for Grafana Cloud import
│
└── .github/workflows/
    ├── preprocessing.yml              # Auto-trigger preprocessing
    └── ci.yml                         # Auto-retrain + Docker push
```

---

## Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/ninditya/mlops-banking-rag-query-router-intent-classifier
cd Banking-RAG-Query-Router-Intent-Classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preprocessing

```bash
cd eksperimen
python automate_NindityaS.py --input huggingface --output banking77_preprocessing
```

### 3. Training (lokal)

```bash
cd workflow_ci
mlflow run . --env-manager=local
```

### 4. Serving + Monitoring Stack

```bash
# Terminal 1 — chatbot UI + inference + /metrics endpoint (port 8001)
cd serving
MODEL_DIR=../workflow_ci/artifacts python inference.py

# Terminal 2 — Prometheus + Grafana stack
cd monitoring
docker-compose up -d
```

Metrics tersedia di `http://localhost:8001/metrics` (Prometheus scrape target).
Akses Grafana di `http://localhost:3000`

### 5. Test Inference

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I lost my debit card"}'
```

Response:
```json
{
  "intent_id": 44,
  "confidence": 0.923,
  "routing": "template_handler",
  "latency_ms": 8.4,
  "rag_answer": "To report a lost or stolen card, please call our 24/7 helpline immediately...",
  "rag_sources": [],
  "rag_score": 0.91,
  "llm_answer": null,
  "conversational_intent": null
}
```

---

## Monitoring — 12 Prometheus Metrics

| # | Metric | Type | Panel |
|---|---|---|---|
| 1 | `router_requests_total` | Counter | Request Rate |
| 2 | `router_request_latency_seconds` | Histogram | P95/P99 Latency |
| 3 | `router_prediction_confidence` | Histogram | Confidence Distribution |
| 4 | `router_routing_decisions_total` | Counter | Tier Breakdown |
| 5 | `router_intent_predictions_total` | Counter | Top 10 Intents |
| 6 | `router_input_text_length_chars` | Histogram | Input Drift |
| 7 | `router_active_requests` | Gauge | Concurrent Load |
| 8 | `router_estimated_cost_usd_total` | Counter | Cumulative Cost |
| 9 | `router_low_confidence_rate` | Gauge | Drift Indicator |
| 10 | `router_model_info` | Gauge | Model Metadata |
| 11 | `router_prediction_errors_total` | Counter | Error Rate |
| 12 | `router_throughput_summary` | Summary | Throughput |

---

## Dataset

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) — Casanueva et al. (2020)
- 13.083 query pelanggan bank
- 77 intent categories
- English, real-world customer service utterances

---

## Results

| Metric | Score |
|---|---|
| F1 Macro | 0.9305 |
| Accuracy | 0.9285 |
| Avg Routing Latency | ~50ms |
| Estimated Cost Saving | ~96% |

---

*Dataset: Banking77 by PolyAI · Casanueva et al., ACL 2020*
# mlops-banking-rag-query-router-intent-classifier
