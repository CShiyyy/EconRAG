"""Cloud LLM client abstraction for Agent C.

Defines a protocol for cloud LLM interaction and provides a Gemini
implementation using the google.genai SDK. The provider is swappable
via CLOUD_PROVIDER config.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

import httpx
from google import genai
from google.genai import types as genai_types
from google.api_core import exceptions as google_exceptions

from pipeline.config import (
    CLOUD_MAX_TOKENS,
    CLOUD_RETRY_BASE_DELAY,
    CLOUD_RETRY_MAX_ATTEMPTS,
    CLOUD_TEMPERATURE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


class CloudAuthError(Exception):
    """Raised on authentication/permission errors — no retry."""


class CloudTimeoutError(Exception):
    """Raised when the cloud API times out after retries."""


@runtime_checkable
class CloudLLMClient(Protocol):
    """Protocol for cloud LLM providers."""

    async def generate(
        self,
        messages: list[dict],
        json_mode: bool = True,
        response_schema: dict | None = None,
    ) -> str:
        """Send messages to the cloud LLM and return raw response text.

        Args:
            messages: List of {"role": "system"|"user", "content": str} dicts.
            json_mode: If True, request JSON-formatted output.
            response_schema: Optional JSON Schema dict for schema-constrained
                generation. When provided, the provider will enforce the exact
                output structure rather than relying solely on prompt instructions.

        Returns:
            Raw response text from the model.

        Raises:
            CloudAuthError: On authentication/permission failures.
            CloudTimeoutError: On timeout after retries.
        """
        ...


class GeminiClient:
    """Gemini implementation of CloudLLMClient using google.genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._api_key = api_key or GEMINI_API_KEY
        self._model_name = model_name or GEMINI_MODEL
        self._temperature = temperature if temperature is not None else CLOUD_TEMPERATURE
        self._max_tokens = max_tokens if max_tokens is not None else CLOUD_MAX_TOKENS

        if not self._api_key:
            raise CloudAuthError("GEMINI_API_KEY is not set")

        self._client = genai.Client(api_key=self._api_key)

    async def generate(
        self,
        messages: list[dict],
        json_mode: bool = True,
        response_schema: dict | None = None,
    ) -> str:
        """Call Gemini with retry logic for rate limits and timeouts."""
        # Separate system instruction from user content
        system_instruction = None
        contents: list[genai_types.Content] = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=msg["content"])],
                    )
                )

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )
        if json_mode:
            config.response_mime_type = "application/json"
        if response_schema is not None:
            config.response_schema = response_schema

        last_error: Exception | None = None
        for attempt in range(CLOUD_RETRY_MAX_ATTEMPTS):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                return response.text

            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.TooManyRequests,
            ) as exc:
                last_error = exc
                delay = CLOUD_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    CLOUD_RETRY_MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)

            except (
                google_exceptions.PermissionDenied,
                google_exceptions.Unauthenticated,
            ) as exc:
                raise CloudAuthError(
                    f"Authentication failed: {exc}"
                ) from exc

            except (
                asyncio.TimeoutError,
                google_exceptions.DeadlineExceeded,
            ) as exc:
                if attempt == 0:
                    last_error = exc
                    logger.warning("Timeout (attempt 1), retrying once")
                    await asyncio.sleep(CLOUD_RETRY_BASE_DELAY)
                else:
                    raise CloudTimeoutError(
                        f"Timed out after {attempt + 1} attempts"
                    ) from exc

        raise CloudTimeoutError(
            f"Failed after {CLOUD_RETRY_MAX_ATTEMPTS} attempts: {last_error}"
        )


class OllamaClient:
    """Ollama implementation of CloudLLMClient for local LLM inference."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._base_url = base_url or OLLAMA_BASE_URL
        self._model_name = model_name or OLLAMA_MODEL
        self._temperature = temperature if temperature is not None else CLOUD_TEMPERATURE
        self._max_tokens = max_tokens if max_tokens is not None else CLOUD_MAX_TOKENS

    async def generate(
        self,
        messages: list[dict],
        json_mode: bool = True,
        response_schema: dict | None = None,
    ) -> str:
        """Call Ollama chat API and return the raw response text."""
        payload: dict = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if response_schema is not None:
            # Ollama structured outputs: pass schema dict directly as format
            payload["format"] = response_schema
        elif json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]


def create_cloud_client() -> CloudLLMClient:
    """Factory: read CLOUD_PROVIDER from config and return the appropriate client."""
    from pipeline.config import CLOUD_PROVIDER

    if CLOUD_PROVIDER == "ollama":
        return OllamaClient()

    if CLOUD_PROVIDER == "gemini":
        return GeminiClient()

    raise ValueError(f"Unsupported CLOUD_PROVIDER: {CLOUD_PROVIDER!r}")
