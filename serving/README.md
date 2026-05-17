---
title: Banking RAG Query Router
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
short_description: Banking chatbot router — SBERT + RAG + LLM, cuts costs 96%
---

# Banking Intent Router — Live Demo

Reduce banking chatbot LLM costs **~96%** using a confidence-based 3-tier routing system.

## How it works

```
User Query → SBERT encode → LinearSVC predict → confidence score
    ├── ≥ 0.55  → Template Handler  ($0.001/query, ~8ms)
    ├── 0.30–0.55 → RAG Pipeline   ($0.010/query, ~300ms)
    └── < 0.30  → LLM Escalation   ($0.050/query, ~2s)
```

## Tech Stack
- **Model**: SBERT (all-MiniLM-L6-v2) + LinearSVC — F1 Macro: **0.9305**
- **RAG**: FAISS + 59-entry banking knowledge base
- **LLM**: OpenAI / Ollama / Mock (configurable)
- **Serving**: FastAPI + Uvicorn

## Results

| Metric | Score |
|---|---|
| F1 Macro | 0.9305 |
| Accuracy | 0.9285 |
| Cost savings vs all-LLM | ~96% |
| Avg latency | ~50ms |

Dataset: [Banking77](https://huggingface.co/datasets/PolyAI/banking77) — 13,083 samples, 77 intents
