# Evaluation — DeepEval + Ragas

Evaluasi kualitas RAG pipeline (`serving/rag.py`) dan LLM escalation tier
(`serving/llm_client.py`), pakai dua framework, judge-nya Claude:

- `deepeval_tests/` — DeepEval, gaya unit-test (pytest). Cek faithfulness
  jawaban ekstraktif, answer relevancy jawaban LLM, dan custom GEval untuk
  safety (mengarang detail kontak).
- `ragas_eval/` — Ragas, fokus retrieval quality (faithfulness,
  context precision, context recall) untuk tier `template_handler`/`rag_pipeline`.

Dataset-nya **bukan contoh hardcoded** — `generate_dataset.py` menjalankan
`serving/rag.py` + `serving/llm_client.py` yang sungguhan atas 12 query yang
di-paraphrase dari `banking_faq.json` (biar retrieval benar-benar diuji
generalisasinya, bukan exact-match).

## Setup

```bash
cd evaluation
pip install -r requirements.txt
# generate_dataset.py juga butuh faiss-cpu + sentence-transformers dari
# ../requirements.txt (sudah wajib ada kalau serving/ sudah pernah di-setup)
```

Judge (Claude) terpisah dari `LLM_PROVIDER` milik app — tidak perlu ubah
`.env` yang sudah ada. Pilih salah satu:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."       # opsi A — langsung ke Anthropic
export OPENROUTER_API_KEY="sk-or-..."       # opsi B — lewat OpenRouter
```

Kalau `OPENROUTER_API_KEY` ke-set, dia yang dipakai (model diakses lewat slug
OpenRouter `anthropic/claude-opus-5`). Kalau tidak, fallback ke
`ANTHROPIC_API_KEY` langsung ke Anthropic. Keduanya **berbayar** saat
eval/test dijalankan.

## 1. Generate dataset (gratis — tanpa judge)

```bash
python generate_dataset.py
```

Load encoder SBERT (`all-MiniLM-L6-v2`) + `banking_faq.json`, retrieve top-3
context untuk tiap query, ambil jawaban ekstraktif (`rag_answer`) dan jawaban
LLM ungrounded (`llm_answer`, lewat `LLM_PROVIDER` di root `.env` — default
`mock`, gratis). Hasil disimpan ke `dataset/generated_eval_set.json`
(sudah ada contoh hasil run di repo ini, generate ulang kalau `banking_faq.json`
berubah).

**Baseline yang terukur (run terakhir):** retrieval top-1 accuracy 10/12
(0.83) atas 12 query paraphrase. Dua miss: "new card hasn't shown up" ditarik
sebagai `activate_card` (harusnya `card_arrival`), dan "app asking me to
prove who I am" ditarik sebagai `why_verify` (harusnya `verify_identity`) —
keduanya secara semantik dekat, kasus ambigu yang wajar untuk retrieval
berbasis embedding.

## 2. Jalankan DeepEval

```bash
deepeval test run deepeval_tests/test_banking_pipeline.py
```

4 kelompok test:
1. `test_retrieval_top1_accuracy` — regresi cepat tanpa judge LLM.
2. `test_extractive_answer_faithfulness` (×12) — tier ekstraktif harus setia
   ke context, harus lulus.
3. `test_llm_escalation_answer_relevancy` (×12) — tier LLM harus relevan
   dengan pertanyaan.
4. `test_llm_escalation_no_fabricated_contact_details` (×12) — custom GEval.
   **Temuan nyata** (bukan test yang sengaja dibuat gagal): untuk query soal
   kartu diblokir, Groq (`llama-3.1-8b-instant`, provider default project ini)
   mengarang nomor telepon `1-800-BANK-123` yang tidak ada di manapun dalam
   `banking_faq.json`. Ini kelas regresi yang sama seperti test keamanan di
   `SQA-AI/deepeval-ragas-demo`: LLM tanpa grounding context bisa "percaya
   diri" mengarang sesuatu yang salah — di konteks banking ini risiko fraud
   nyata (nomor kontak palsu).

## 3. Jalankan Ragas

```bash
python ragas_eval/evaluate_rag.py
```

Cetak & simpan (`ragas_results.csv`) skor faithfulness, context precision,
dan context recall untuk 12 pasangan query/context/answer dari tier
retrieval. Tier `llm_escalation` sengaja tidak dievaluasi di sini karena
tidak menerima retrieval context sama sekali (lihat `serving/inference.py`).

## Kenapa cuma metrik LLM-based yang dipilih

DeepEval dan Ragas defaultnya banyak yang butuh embedding model (mis.
`AnswerRelevancyMetric` versi Ragas, `AnswerCorrectness`) — dan Anthropic
tidak punya API embedding, jadi itu perlu provider kedua (OpenAI/Voyage).
Supaya eval ini tetap satu-vendor untuk judge (cukup Claude), dipilih metrik
yang murni LLM-judged.

## Catatan versi (penting)

- Paket `ragas` di PyPI sekarang di-maintain di bawah nama org **Vibrant
  Labs** (`vibrantlabsai/ragas`), rebrand resmi dari
  `explodinggradients/ragas` — bukan package lain/palsu, sudah dicek.
- `ragas==0.4.3` masih meng-import
  `langchain_community.chat_models.vertexai` secara *unconditional* di level
  module, padahal submodule itu sudah dihapus di `langchain-community>=0.4`.
  Kalau tidak di-pin, `import ragas` langsung `ModuleNotFoundError`.
  `requirements.txt` di sini sudah mengunci `langchain-community==0.3.27` +
  `langchain-anthropic==0.3.22` (rangkaian langchain 0.3.x pra-1.0 yang
  saling kompatibel) supaya bebas dari bug ini. Kalau suatu saat upgrade
  `ragas`, cek dulu apakah bug ini sudah diperbaiki sebelum melepas pin-nya.
