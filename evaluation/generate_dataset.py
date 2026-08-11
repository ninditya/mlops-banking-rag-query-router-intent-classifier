"""Generate dataset evaluasi dari pipeline RAG + LLM yang SUNGGUHAN (bukan
contoh hardcoded) -- menjalankan serving/rag.py dan serving/llm_client.py atas
daftar query di dataset/eval_queries.json (paraphrase dari banking_faq.json,
untuk menguji generalisasi retrieval, bukan exact-match).

Untuk tiap query, disimpan:
- retrieved_contexts : top-3 jawaban FAQ hasil FAISS retrieval (dipakai RAGAS)
- rag_answer          : jawaban ekstraktif tier template_handler/rag_pipeline
                        (top-1, tanpa generasi -- faithful by construction)
- llm_answer          : jawaban ungrounded tier llm_escalation (tanpa context,
                        dipakai untuk cek answer relevancy & safety)
- ground_truth_answer : jawaban asli dari banking_faq.json untuk id tersebut

Tidak butuh model.pkl (classifier) -- retrieval cukup pakai encoder SBERT.
LLM_PROVIDER default "mock" (gratis, tanpa API key). Set LLM_PROVIDER=openai
atau groq di .env untuk hasil llm_answer yang lebih representatif.

Jalankan: python generate_dataset.py
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "serving"))

from rag import BankingRAG  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

FAQ_PATH = ROOT / "serving" / "banking_faq.json"
QUERIES_PATH = Path(__file__).resolve().parent / "dataset" / "eval_queries.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset" / "generated_eval_set.json"


async def main():
    faq = json.loads(FAQ_PATH.read_text())
    faq_by_id = {item["id"]: item for item in faq}
    queries = json.loads(QUERIES_PATH.read_text())

    print("Loading encoder (all-MiniLM-L6-v2) ...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    rag = BankingRAG(faq_path=str(FAQ_PATH), encoder=encoder)
    llm_client = LLMClient()

    dataset = []
    for row in queries:
        query = row["query"]
        ground_truth_id = row["ground_truth_id"]
        ground_truth_answer = faq_by_id[ground_truth_id]["answer"]

        retrieval = rag.answer(query, top_k=3)
        retrieved_contexts = [s["answer"] for s in retrieval["sources"]]
        rag_answer = retrieval["answer"]
        predicted_id = retrieval["sources"][0]["id"] if retrieval["sources"] else None

        llm_answer = await llm_client.generate(query, intent_id=-1, confidence=0.0)

        dataset.append({
            "query": query,
            "ground_truth_id": ground_truth_id,
            "ground_truth_answer": ground_truth_answer,
            "predicted_id": predicted_id,
            "retrieval_correct": predicted_id == ground_truth_id,
            "retrieved_contexts": retrieved_contexts,
            "rag_answer": rag_answer,
            "llm_answer": llm_answer,
        })
        print(f"  [{ 'OK' if predicted_id == ground_truth_id else 'MISS'}] {query!r} -> {predicted_id}")

    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2))
    hits = sum(1 for d in dataset if d["retrieval_correct"])
    print(f"\nSaved {len(dataset)} entries -> {OUTPUT_PATH}")
    print(f"Retrieval top-1 accuracy: {hits}/{len(dataset)}")


if __name__ == "__main__":
    asyncio.run(main())
