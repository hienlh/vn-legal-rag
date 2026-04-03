"""Basic LLM provider abstraction for vn_legal_rag.

Features:
- API key validation at init time
- Timeout and retry logic with exponential backoff
- Empty response validation
"""

import os
import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMEmptyResponseError(LLMError):
    """Raised when LLM returns empty response."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""
    pass


class BaseLLMProvider:
    """Base class for LLM providers."""

    def __init__(self, model: str, timeout: int = DEFAULT_TIMEOUT, **kwargs):
        self.model = model
        self.timeout = timeout
        self.kwargs = kwargs

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
        max_retries: int = MAX_RETRIES,
        **kwargs
    ) -> str:
        """Generate response from LLM with retry logic."""
        raise NotImplementedError

    def _validate_response(self, response: Optional[str]) -> str:
        """Validate LLM response is not empty."""
        if not response or not response.strip():
            raise LLMEmptyResponseError("LLM returned empty response")
        return response


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider with timeout and retry."""

    def __init__(self, model: str = "gemini-2.0-flash", timeout: int = DEFAULT_TIMEOUT, **kwargs):
        super().__init__(model, timeout, **kwargs)
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set. "
                    "Set it with: export GOOGLE_API_KEY='your-key'"
                )
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
        max_retries: int = MAX_RETRIES,
        **kwargs
    ) -> str:
        """Generate with retry and timeout."""
        timeout = timeout or self.timeout
        backoff = INITIAL_BACKOFF

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={"temperature": temperature},
                    request_options={"timeout": timeout}
                )
                return self._validate_response(response.text)
            except LLMEmptyResponseError:
                raise
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(x in error_str for x in ["rate", "limit", "timeout", "429", "503"])

                if attempt < max_retries - 1 and is_retryable:
                    logger.warning(f"Gemini attempt {attempt + 1} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise LLMError(f"Gemini API error after {attempt + 1} attempts: {e}")

        raise LLMError(f"Gemini failed after {max_retries} retries")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider with timeout and retry."""

    def __init__(self, model: str = "gpt-4o-mini", timeout: int = DEFAULT_TIMEOUT, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, timeout, **kwargs)
        try:
            from openai import OpenAI
            client_kwargs = {"timeout": timeout}
            if base_url:
                client_kwargs["base_url"] = base_url
                # Proxy doesn't need real API key
                client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY", "proxy-key")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable not set. "
                        "Set it with: export OPENAI_API_KEY='your-key'"
                    )
                client_kwargs["api_key"] = api_key
            self.client = OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
        max_retries: int = MAX_RETRIES,
        **kwargs
    ) -> str:
        """Generate with retry and timeout."""
        timeout = timeout or self.timeout
        backoff = INITIAL_BACKOFF

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=timeout,
                )
                content = response.choices[0].message.content
                return self._validate_response(content)
            except LLMEmptyResponseError:
                raise
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(x in error_str for x in ["rate", "limit", "timeout", "429", "503"])

                if attempt < max_retries - 1 and is_retryable:
                    logger.warning(f"OpenAI attempt {attempt + 1} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise LLMError(f"OpenAI API error after {attempt + 1} attempts: {e}")

        raise LLMError(f"OpenAI failed after {max_retries} retries")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with timeout and retry."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", timeout: int = DEFAULT_TIMEOUT, **kwargs):
        super().__init__(model, timeout, **kwargs)
        base_url = kwargs.get("base_url") or os.getenv("ANTHROPIC_BASE_URL") or None
        api_key = os.getenv("ANTHROPIC_API_KEY") or None
        # Treat empty string as unset
        if not api_key:
            api_key = "sk-ant-placeholder"
        if base_url is not None and not base_url:
            base_url = None
        try:
            from anthropic import Anthropic
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            # Prevent empty auth_token from being sent
            auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
            if not auth_token:
                os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            self.client = Anthropic(**client_kwargs)
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
        max_retries: int = MAX_RETRIES,
        **kwargs
    ) -> str:
        """Generate with retry and timeout."""
        timeout = timeout or self.timeout
        backoff = INITIAL_BACKOFF

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=timeout,
                )
                # Handle both parsed object and raw SSE string (Meridian proxy)
                if isinstance(response, str):
                    return self._parse_sse_response(response)
                content = response.content[0].text
                return self._validate_response(content)
            except LLMEmptyResponseError:
                raise
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(x in error_str for x in ["rate", "limit", "timeout", "429", "503", "overloaded"])

                if attempt < max_retries - 1 and is_retryable:
                    logger.warning(f"Anthropic attempt {attempt + 1} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise LLMError(f"Anthropic API error after {attempt + 1} attempts: {e}")

        raise LLMError(f"Anthropic failed after {max_retries} retries")

    def _parse_sse_response(self, raw: str) -> str:
        """Parse SSE stream response from Meridian proxy."""
        import json as _json
        text_parts = []
        for line in raw.split("\n"):
            if line.startswith("data: "):
                try:
                    data = _json.loads(line[6:])
                    if data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_parts.append(delta.get("text", ""))
                except _json.JSONDecodeError:
                    continue
        return self._validate_response("".join(text_parts))


def create_llm_provider(
    provider: str,
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Create LLM provider instance.

    Args:
        provider: Provider name (gemini, openai, anthropic)
        model: Model name (optional)
        **kwargs: Additional provider-specific arguments

    Returns:
        LLM provider instance
    """
    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": OpenAIProvider,  # Uses OpenAI-compatible proxy via base_url
    }

    provider_class = providers.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider}")

    if model:
        kwargs["model"] = model

    return provider_class(**kwargs)
