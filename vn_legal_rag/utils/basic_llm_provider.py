"""Basic LLM provider abstraction for vn_legal_rag."""

from typing import Any, Dict, Optional


class BaseLLMProvider:
    """Base class for LLM providers."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def generate(self, prompt: str, temperature: float = 0.1, **kwargs) -> str:
        """Generate response from LLM."""
        raise NotImplementedError


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""

    def __init__(self, model: str = "gemini-2.0-flash", **kwargs):
        super().__init__(model, **kwargs)
        try:
            import google.generativeai as genai
            import os
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    def generate(self, prompt: str, temperature: float = 0.1, **kwargs) -> str:
        response = self.client.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        return response.text


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider."""

    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(model, **kwargs)
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def generate(self, prompt: str, temperature: float = 0.1, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content


def create_llm_provider(
    provider: str,
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Create LLM provider instance.

    Args:
        provider: Provider name (gemini, openai)
        model: Model name (optional)
        **kwargs: Additional provider-specific arguments

    Returns:
        LLM provider instance
    """
    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
    }

    provider_class = providers.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider}")

    if model:
        kwargs["model"] = model

    return provider_class(**kwargs)
