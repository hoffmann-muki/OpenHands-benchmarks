from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import LLM


DEFAULT_LLM_MODEL = "openrouter/qwen/qwen3-coder-next"
DEFAULT_LLM_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_LLM_TEMPERATURE = 0.1


def load_llm_config(
    config_path: str | Path | None,
    *,
    default_model: str | None = None,
) -> LLM:
    if config_path is None:
        if default_model is None:
            raise ValueError("An LLM config file is required")
        api_key = os.getenv(DEFAULT_LLM_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"{DEFAULT_LLM_API_KEY_ENV} is required for the default "
                f"{default_model} model"
            )
        return LLM(
            model=default_model,
            api_key=SecretStr(api_key),
            temperature=DEFAULT_LLM_TEMPERATURE,
        )

    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"LLM config file {config_path} does not exist")

    with config_path.open("r", encoding="utf-8") as f:
        llm_config = f.read()

    return LLM.model_validate_json(llm_config)
