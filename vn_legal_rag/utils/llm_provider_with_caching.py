"""
Simplified LLM provider wrapper with SQLite response caching.

Supports: openai, anthropic, gemini
"""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LLMProvider:
    """
    Unified LLM provider with response caching.

    Supports:
    - openai (gpt-4, gpt-3.5-turbo, etc.)
    - anthropic (claude-3-sonnet, claude-3-opus, etc.)
    - gemini (gemini-pro, gemini-1.5-pro, etc.)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        cache_db: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name (openai, anthropic, gemini)
            model: Model name
            cache_db: Path to SQLite cache database
            **kwargs: Additional provider-specific arguments
        """
        self.provider = provider.lower()
        self.model = model
        self.config = kwargs

        # Initialize client
        self.client = self._init_client()

        # Initialize cache
        if cache_db:
            self.cache_db = Path(cache_db)
            self._init_cache_db()
        else:
            self.cache_db = None

    def _init_client(self) -> Any:
        """Initialize provider client."""
        base_url = self.config.get("base_url")

        if self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
                client_kwargs = {"api_key": api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url
                return OpenAI(**client_kwargs)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
                # When using proxy, allow dummy API key (proxy handles auth)
                if base_url and not api_key:
                    api_key = "dummy"
                client_kwargs = {"api_key": api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url
                return Anthropic(**client_kwargs)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                genai.configure(api_key=api_key)
                return genai
            except ImportError:
                raise ImportError("Google GenAI package not installed. Run: pip install google-generativeai")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        if not self.cache_db:
            return

        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                provider TEXT,
                model TEXT,
                prompt TEXT,
                response TEXT,
                timestamp INTEGER,
                response_time_ms INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters.

        Compatible with semantica's LLMCacheManager format.
        """
        # Filter relevant params (same as semantica)
        relevant_params = {
            k: v for k, v in sorted(kwargs.items())
            if k in ('temperature', 'max_tokens', 'top_p', 'top_k', 'system_prompt')
            and v is not None
        }

        # Match semantica's cache key format
        key_data = {
            "provider": self.provider.lower(),
            "model": self.model.lower(),
            "prompt": prompt,
            "params": relevant_params,
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if exists."""
        if not self.cache_db or not self.cache_db.exists():
            return None

        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def _cache_response(self, cache_key: str, prompt: str, response: str, response_time_ms: int):
        """Cache LLM response using semantica-compatible schema."""
        if not self.cache_db:
            return

        import hashlib
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
        params_hash = hashlib.sha256(b'').hexdigest()[:16]  # Empty params
        now = time.time()
        expires_at = now + 86400 * 365  # 1 year expiry

        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO llm_cache
            (cache_key, provider, model, prompt_hash, params_hash, prompt_text, response, created_at, expires_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cache_key,
            self.provider,
            self.model,
            prompt_hash,
            params_hash,
            prompt,
            response,
            now,
            expires_at,
            0,
        ))
        conn.commit()
        conn.close()

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        # Check cache
        cache_key = self._get_cache_key(prompt, **kwargs)
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

        # Generate
        start_time = time.time()
        response = self._generate_impl(prompt, **kwargs)
        response_time_ms = int((time.time() - start_time) * 1000)

        # Cache
        self._cache_response(cache_key, prompt, response, response_time_ms)

        return response

    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """Provider-specific generation implementation."""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2000)

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            create_kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Use streaming for proxies (like claude-max-proxy)
            if self.config.get("base_url"):
                collected_text = []
                with self.client.messages.stream(**create_kwargs) as stream:
                    for text in stream.text_stream:
                        collected_text.append(text)
                return "".join(collected_text)
            else:
                response = self.client.messages.create(**create_kwargs)
                return response.content[0].text

        elif self.provider == "gemini":
            model = self.client.GenerativeModel(self.model)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            return response.text

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_sse_stream(self, sse_data: str) -> str:
        """Parse SSE stream from proxy to extract text content.

        SSE format:
        event: content_block_delta
        data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

        Args:
            sse_data: Raw SSE stream string

        Returns:
            Extracted text content
        """
        text_parts = []
        for line in sse_data.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                try:
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    data = json.loads(data_str)
                    # Extract text from content_block_delta events
                    if data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_parts.append(delta.get("text", ""))
                    # Also handle message_delta for stop_reason
                    elif data.get("type") == "message_delta":
                        pass  # Just marks end of message
                except json.JSONDecodeError:
                    continue
        return "".join(text_parts)

    def generate_json(self, prompt: str, **kwargs) -> Union[Dict, list]:
        """
        Generate JSON response.

        Args:
            prompt: Input prompt (should request JSON output)
            **kwargs: Additional generation parameters

        Returns:
            Parsed JSON object or array
        """
        response = self.generate(prompt, **kwargs)

        # Parse JSON from response
        text = response.strip()

        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            blocks = text.split("```")
            for block in blocks:
                block = block.strip()
                if (block.startswith("{") and block.endswith("}")) or \
                   (block.startswith("[") and block.endswith("]")):
                    text = block
                    break

        # Try to parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Find JSON boundaries
            start_obj = text.find("{")
            start_list = text.find("[")

            if start_obj >= 0 and (start_list < 0 or start_obj < start_list):
                # Object
                end = text.rfind("}")
                if end > start_obj:
                    return json.loads(text[start_obj:end + 1])
            elif start_list >= 0:
                # Array
                end = text.rfind("]")
                if end > start_list:
                    return json.loads(text[start_list:end + 1])

            raise ValueError(f"Could not parse JSON from response: {text[:200]}")

    def clear_cache(self):
        """Clear all cached responses."""
        if not self.cache_db or not self.cache_db.exists():
            return

        conn = sqlite3.connect(str(self.cache_db))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM llm_cache")
        conn.commit()
        conn.close()


# Factory function
def create_llm_provider(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    cache_db: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """
    Create LLM provider instance.

    Args:
        provider: Provider name (openai, anthropic, gemini)
        model: Model name
        cache_db: Path to SQLite cache database
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    return LLMProvider(provider=provider, model=model, cache_db=cache_db, **kwargs)
