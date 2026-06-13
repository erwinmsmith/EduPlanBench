from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def get_llm_settings() -> dict[str, str]:
    load_dotenv()
    provider = os.environ.get("EDUPLAN_LLM_PROVIDER", "deepseek")
    api_key = (
        os.environ.get("EDUPLAN_LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    base_url = os.environ.get("EDUPLAN_LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("EDUPLAN_LLM_MODEL", "deepseek-chat")
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def build_external_llm_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env: dict[str, str] = dict(base_env or os.environ)
    settings = get_llm_settings()
    api_key = settings["api_key"]
    env["EDUPLAN_LLM_PROVIDER"] = settings["provider"]
    env["EDUPLAN_LLM_BASE_URL"] = settings["base_url"]
    env["EDUPLAN_LLM_MODEL"] = settings["model"]
    if api_key:
        env["EDUPLAN_LLM_API_KEY"] = api_key
        env.setdefault("DEEPSEEK_API_KEY", api_key)
        # Many third-party agents use OpenAI-compatible clients and only look
        # for OpenAI-style variable names.
        env.setdefault("OPENAI_API_KEY", api_key)
    env.setdefault("OPENAI_BASE_URL", settings["base_url"])
    env.setdefault("OPENAI_API_BASE", settings["base_url"])
    env.setdefault("MODEL_NAME", settings["model"])
    env.setdefault("OPENAI_MODEL", settings["model"])
    return env
