"""Wrapper supaya DeepEval pakai Claude (bukan OpenAI default) sebagai judge LLM.

Judge ini terpisah dari LLM_PROVIDER milik aplikasi (serving/llm_client.py) --
tidak perlu ubah kode app, cukup set salah satu env var di bawah untuk
menjalankan evaluasi.

Otomatis pilih jalur koneksi berdasarkan env var yang tersedia:
- OPENROUTER_API_KEY di-set -> lewat OpenRouter (endpoint OpenAI-compatible,
  slug model "anthropic/<model>").
- Kalau tidak -> ANTHROPIC_API_KEY langsung ke Anthropic.
"""

import os

import anthropic
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ClaudeModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "claude-opus-5"):
        self.use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
        if self.use_openrouter:
            self.model_name = f"anthropic/{model_name}"
            self.client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            self.model_name = model_name
            self.client = anthropic.Anthropic()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        if self.use_openrouter:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def a_generate(self, prompt: str) -> str:
        if self.use_openrouter:
            async_client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            response = await async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        async_client = anthropic.AsyncAnthropic()
        response = await async_client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def get_model_name(self) -> str:
        return self.model_name
