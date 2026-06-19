from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    confidence_threshold: float = 0.5
    model: ProviderConfig | None = None
    judge_model: ProviderConfig | None = None


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    state_dir = root / "state"
    state_dir.mkdir(exist_ok=True)

    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    api_key = os.getenv("OPENAI_API_KEY")

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=0.7,
        api_key=api_key,
    )

    judge_model = model 

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=1000,
        compact_keep_messages=6,
        model=model,
        judge_model=judge_model,
    )