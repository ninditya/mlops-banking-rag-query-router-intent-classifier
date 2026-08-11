"""Evaluasi Ragas untuk tier retrieval (template_handler/rag_pipeline) --
load dataset/generated_eval_set.json yang dihasilkan dari pipeline
sungguhan serving/rag.py (lihat generate_dataset.py), bukan contoh hardcoded.

Metrik dipilih yang murni LLM-based (faithfulness, context_precision,
context_recall) supaya cukup satu LLM (Claude) -- tidak perlu embedding
provider terpisah (OpenAI/Voyage), karena Anthropic tidak punya API embedding.

llm_escalation tier (jawaban ungrounded) SENGAJA tidak dievaluasi di sini --
tier itu tidak menerima retrieval context sama sekali (lihat inference.py),
jadi context_precision/recall/faithfulness tidak relevan untuknya. Lihat
deepeval_tests/test_banking_pipeline.py untuk evaluasi tier tersebut.

Jalur koneksi otomatis pilih berdasarkan env var:
- OPENROUTER_API_KEY di-set -> lewat OpenRouter (slug "anthropic/<model>").
- Kalau tidak -> ANTHROPIC_API_KEY langsung ke Anthropic.

Jalankan: python ragas_eval/evaluate_rag.py
Ini memanggil API asli (berbayar), lewat OpenRouter atau Anthropic langsung.
"""

import json
import os
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import context_precision, context_recall, faithfulness

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "generated_eval_set.json"

if os.environ.get("OPENROUTER_API_KEY"):
    from langchain_openai import ChatOpenAI

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="anthropic/claude-opus-5",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    )
else:
    from langchain_anthropic import ChatAnthropic

    evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-opus-5"))


def load_dataset() -> Dataset:
    rows = json.loads(DATASET_PATH.read_text())
    data = {
        "question": [r["query"] for r in rows],
        "answer": [r["rag_answer"] for r in rows],
        "contexts": [r["retrieved_contexts"] for r in rows],
        "ground_truth": [r["ground_truth_answer"] for r in rows],
    }
    return Dataset.from_dict(data)


def main():
    dataset = load_dataset()
    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=evaluator_llm,
    )
    df = result.to_pandas()
    print(df)

    out_path = Path(__file__).resolve().parent / "ragas_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
