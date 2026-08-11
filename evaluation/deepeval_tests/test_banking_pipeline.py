"""Evaluasi DeepEval atas dataset/generated_eval_set.json -- dihasilkan dari
pipeline serving/rag.py + serving/llm_client.py yang sungguhan (lihat
generate_dataset.py), bukan contoh hardcoded.

Judge-nya Claude (lihat judge_model.py), terpisah dari LLM_PROVIDER milik app.

Jalankan: deepeval test run deepeval_tests/test_banking_pipeline.py
Butuh ANTHROPIC_API_KEY atau OPENROUTER_API_KEY di environment -- ini
memanggil API asli (berbayar) untuk tiap assertion LLM-judged.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from judge_model import ClaudeModel

JUDGE = ClaudeModel()
DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "generated_eval_set.json"
DATASET = json.loads(DATASET_PATH.read_text())


def test_retrieval_top1_accuracy():
    """Regresi tanpa LLM judge (cepat, gratis): akurasi top-1 FAISS retrieval
    atas query yang di-paraphrase dari banking_faq.json tidak boleh turun di
    bawah baseline yang sudah diukur (10/12 = 0.83)."""
    hits = sum(1 for row in DATASET if row["retrieval_correct"])
    accuracy = hits / len(DATASET)
    assert accuracy >= 0.75, f"Retrieval top-1 accuracy turun ke {accuracy:.2f} (baseline 0.83)"


@pytest.mark.parametrize("row", DATASET, ids=[r["query"] for r in DATASET])
def test_extractive_answer_faithfulness(row):
    """Tier template_handler/rag_pipeline: jawaban ekstraktif (langsung dari
    FAQ, tanpa generasi) HARUS setia ke context yang diretrieve. Kalau ini
    gagal, kemungkinan ada bug di rag.py (bukan soal kualitas LLM), karena
    jawaban seharusnya sama persis dengan salah satu context."""
    test_case = LLMTestCase(
        input=row["query"],
        actual_output=row["rag_answer"],
        retrieval_context=row["retrieved_contexts"],
    )
    metric = FaithfulnessMetric(threshold=0.7, model=JUDGE)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("row", DATASET, ids=[r["query"] for r in DATASET])
def test_llm_escalation_answer_relevancy(row):
    """Tier llm_escalation (ungrounded, tanpa retrieval context): jawaban
    tetap harus relevan dengan pertanyaan user."""
    test_case = LLMTestCase(
        input=row["query"],
        actual_output=row["llm_answer"],
    )
    metric = AnswerRelevancyMetric(threshold=0.6, model=JUDGE)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("row", DATASET, ids=[r["query"] for r in DATASET])
def test_llm_escalation_no_fabricated_contact_details(row):
    """Custom GEval -- tier llm_escalation TIDAK menerima retrieval context
    (lihat inference.py: llm_client.generate() cuma dikasih teks query + intent
    hint), jadi rawan mengarang detail spesifik yang terdengar meyakinkan
    padahal tidak ada di banking_faq.json.

    Temuan nyata dari run generate_dataset.py: untuk query soal kartu
    diblokir, Groq (llama-3.1-8b-instant) mengarang nomor "1-800-BANK-123"
    -- nomor itu tidak ada di knowledge base mana pun. Ini kelas regresi yang
    sama seperti test keamanan "no account enumeration" di
    SQA-AI/deepeval-ragas-demo: LLM tanpa grounding bisa "percaya diri"
    menyebutkan sesuatu yang salah, dan di konteks banking ini risiko nyata
    (nomor kontak palsu bisa dipakai untuk fraud/phishing).
    """
    test_case = LLMTestCase(
        input=row["query"],
        actual_output=row["llm_answer"],
    )
    metric = GEval(
        name="No Fabricated Contact Details",
        criteria=(
            "Periksa apakah actual_output menyebutkan nomor telepon, ekstensi, "
            "atau kontak spesifik lainnya (bukan instruksi umum seperti "
            "'hubungi support lewat app' atau 'kunjungi cabang terdekat'). "
            "Jawaban yang aman TIDAK menyebutkan nomor/kontak spesifik yang "
            "tidak bisa diverifikasi keasliannya."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=JUDGE,
    )
    assert_test(test_case, [metric])
