"""
LLM client untuk Banking Router — llm_escalation tier.

Provider didukung (via LLM_PROVIDER env var):
  - "openai"  → OpenAI Chat API (butuh OPENAI_API_KEY)
  - "groq"    → Groq API gratis  (butuh GROQ_API_KEY)
  - "ollama"  → Ollama lokal     (butuh Ollama running di OLLAMA_BASE_URL)
  - "mock"    → Fallback tanpa API key (default)

Env vars:
  LLM_PROVIDER      openai | groq | ollama | mock  (default: mock)
  OPENAI_API_KEY    sk-...
  OPENAI_MODEL      gpt-4o-mini (default)
  GROQ_API_KEY      gsk_...
  GROQ_MODEL        llama-3.2-3b-instant (default)
  OLLAMA_BASE_URL   http://localhost:11434 (default)
  OLLAMA_MODEL      llama3.2 (default)
"""

import os
import logging
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Load .env dari root project (dua level di atas serving/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

# Banking77 intent index → human-readable label
BANKING77_INTENTS = {
    0:  "activate_my_card",
    1:  "age_limit",
    2:  "apple_pay_or_google_pay",
    3:  "atm_support",
    4:  "automatic_top_up",
    5:  "balance_not_updated_after_bank_transfer",
    6:  "balance_not_updated_after_cheque_or_cash_deposit",
    7:  "beneficiary_not_allowed",
    8:  "cancel_transfer",
    9:  "card_about_to_expire",
    10: "card_acceptance",
    11: "card_arrival",
    12: "card_blocked",
    13: "card_delivery_estimate",
    14: "card_linking",
    15: "card_not_working",
    16: "card_payment_fee_charged",
    17: "card_payment_not_recognised",
    18: "card_payment_wrong_exchange_rate",
    19: "card_swallowed",
    20: "cash_withdrawal_charge",
    21: "cash_withdrawal_not_recognised",
    22: "change_pin",
    23: "compromised_card",
    24: "contactless_not_working",
    25: "country_support",
    26: "declined_card_payment",
    27: "declined_cash_withdrawal",
    28: "declined_transfer",
    29: "direct_debit_payment_not_recognised",
    30: "disposable_card_limits",
    31: "edit_personal_details",
    32: "exchange_charge",
    33: "exchange_rate",
    34: "exchange_via_app",
    35: "extra_charge_on_statement",
    36: "failed_transfer",
    37: "fiat_currency_support",
    38: "freeze_account",
    39: "get_disposable_virtual_card",
    40: "get_physical_card",
    41: "getting_spare_card",
    42: "getting_virtual_card",
    43: "identity_document_wrongly_rejected",
    44: "lost_or_stolen_card",
    45: "lost_or_stolen_phone",
    46: "order_physical_card",
    47: "passcode_forgotten",
    48: "pending_card_payment",
    49: "pending_cash_withdrawal",
    50: "pending_top_up",
    51: "pending_transfer",
    52: "pin_blocked",
    53: "receiving_money",
    54: "refund_not_showing_up",
    55: "request_refund",
    56: "reverted_card_payment",
    57: "supported_cards_and_pensions",
    58: "terminate_account",
    59: "top_up_by_bank_transfer_charge",
    60: "top_up_failed",
    61: "top_up_limits",
    62: "top_up_reverted",
    63: "topping_up_by_card",
    64: "transaction_charged_twice",
    65: "transfer_fee_charged",
    66: "transfer_into_account",
    67: "transfer_limit",
    68: "transfer_not_received_by_recipient",
    69: "transfer_timing",
    70: "unable_to_verify_identity",
    71: "verify_my_identity",
    72: "verify_source_of_funds",
    73: "verify_top_up",
    74: "virtual_card_not_working",
    75: "visa_or_mastercard",
    76: "why_verify_identity",
}

SYSTEM_PROMPT = """You are a helpful banking customer support assistant.
The user has sent a query that requires a detailed or nuanced response beyond standard templates.
Answer clearly and concisely (2-4 sentences). Focus on actionable guidance.
Do not mention internal systems, confidence scores, or routing tiers.
If the query is ambiguous, ask one clarifying question."""


def _build_messages(user_text: str, intent_label: str, confidence: float) -> list[dict]:
    # Hanya pakai intent hint jika confidence cukup reliable (≥ 0.25)
    # Di bawah itu, intent prediksi terlalu noise dan bisa menyesatkan LLM
    if confidence >= 0.25:
        context_hint = f"[Context: query likely relates to '{intent_label.replace('_', ' ')}']\n"
    else:
        context_hint = ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": context_hint + user_text},
    ]


class LLMClient:
    def __init__(self):
        self.provider     = os.getenv("LLM_PROVIDER", "mock").lower()
        self.api_key      = os.getenv("OPENAI_API_KEY", "")
        self.oai_model    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.groq_key     = os.getenv("GROQ_API_KEY", "")
        self.groq_model   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.ollama_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

        if self.provider == "openai" and not self.api_key:
            logger.warning("LLM_PROVIDER=openai tapi OPENAI_API_KEY kosong — fallback ke mock")
            self.provider = "mock"
        if self.provider == "groq" and not self.groq_key:
            logger.warning("LLM_PROVIDER=groq tapi GROQ_API_KEY kosong — fallback ke mock")
            self.provider = "mock"

        logger.info(f"LLMClient ready — provider={self.provider}")

    async def generate(self, user_text: str, intent_id: int, confidence: float = 0.0) -> str:
        intent_label = BANKING77_INTENTS.get(intent_id, "general_banking")
        messages     = _build_messages(user_text, intent_label, confidence)

        if self.provider == "openai":
            return await self._call_openai(messages)
        elif self.provider == "groq":
            return await self._call_groq(messages)
        elif self.provider == "ollama":
            return await self._call_ollama(messages)
        else:
            return self._mock_response(intent_label)

    # ------------------------------------------------------------------
    async def _call_openai(self, messages: list[dict]) -> str:
        payload = {
            "model":       self.oai_model,
            "messages":    messages,
            "max_tokens":  200,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _call_groq(self, messages: list[dict]) -> str:
        payload = {
            "model":       self.groq_model,
            "messages":    messages,
            "max_tokens":  200,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type":  "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _call_ollama(self, messages: list[dict]) -> str:
        payload = {
            "model":    self.ollama_model,
            "messages": messages,
            "stream":   False,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _mock_response(intent_label: str) -> str:
        label = intent_label.replace("_", " ")
        return (
            f"Thank you for reaching out regarding your {label} concern. "
            "Our support team is reviewing your request and will provide a detailed "
            "response within 24 hours. For urgent matters, please call our 24/7 "
            "helpline or visit a branch near you."
        )
