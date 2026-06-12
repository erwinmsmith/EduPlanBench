from __future__ import annotations

import json
import os
import time
import http.client
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from eduplanbench.core.env import load_dotenv


@dataclass(slots=True)
class OpenAICompatibleClient:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: int = 60
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        load_dotenv()
        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("EDUPLAN_LLM_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.environ.get("EDUPLAN_LLM_MODEL", "deepseek-chat"),
        )

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("missing LLM API key; set DEEPSEEK_API_KEY or OPENAI_API_KEY")
        response_text = self._chat(prompt)
        return _extract_json(response_text)

    def _chat(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.RemoteDisconnected, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last_error}")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise
