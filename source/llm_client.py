"""
Unified LLM client supporting the allowed competition models:
- Google Gemini (gemini-*-flash-lite / flash)
- OpenAI (gpt-4.1-mini, gpt-5-mini)
- Anthropic (claude-haiku-4-5)
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PROVIDER = os.getenv("MODEL_PROVIDER", "google").lower()
MODEL_NAME = os.getenv("MODEL_NAME","gemini-3.1-flash-lite")
API_KEY = os.getenv("API_KEY", "")


def _encode_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime(path: str | Path) -> str:
    p = str(path).lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return "image/png"


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = (provider or PROVIDER).lower()
        self.model = model or MODEL_NAME
        self.api_key = api_key or API_KEY
        if not self.api_key:
            logger.warning("API_KEY is empty – LLM calls will fail")

        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        elif self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text. Optionally attach one image (vision)."""
        try:
            if self.provider == "google":
                return self._google(prompt, image_path, temperature, max_tokens)
            if self.provider == "openai":
                return self._openai(prompt, image_path, temperature, max_tokens)
            if self.provider == "anthropic":
                return self._anthropic(prompt, image_path, temperature, max_tokens)
        except Exception as e:
            logger.error("LLM call failed (%s/%s): %s", self.provider, self.model, e)
            raise
        return ""

    def _google(self, prompt, image_path, temperature, max_tokens):
        import google.generativeai as genai
        from PIL import Image

        parts: list[Any] = [prompt]
        if image_path:
            img = Image.open(image_path)
            parts.append(img)

        response = self._client.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    def _openai(self, prompt, image_path, temperature, max_tokens):
        messages = [{"role": "user", "content": []}]
        content = messages[0]["content"]
        content.append({"type": "text", "text": prompt})
        if image_path:
            b64 = _encode_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{_mime(image_path)};base64,{b64}",
                    },
                }
            )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _anthropic(self, prompt, image_path, temperature, max_tokens):
        content: list[dict] = []
        if image_path:
            b64 = _encode_image(image_path)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _mime(image_path),
                        "data": b64,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text if resp.content else ""