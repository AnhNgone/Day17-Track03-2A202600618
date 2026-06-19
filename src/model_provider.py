from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.7
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""
    value = value.lower().strip()

    mapping = {
        "openai": "openai",
        "gpt": "openai",

        "gemini": "gemini",
        "google": "gemini",
    }

    if value not in mapping:
        raise ValueError(f"Unsupported provider: {value}")

    return mapping[value]


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
