from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eduplanbench.agents import bridge_policy
from eduplanbench.core.env import get_llm_settings
from eduplanbench.llm.openai_compatible import OpenAICompatibleClient


class EduPlanBridge:
    def __init__(self, agent: str) -> None:
        self.agent = bridge_policy.normalize_agent(agent)
        self.task: dict[str, Any] | None = None
        self.client: OpenAICompatibleClient | None = None
        self.history: list[dict[str, Any]] = []

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        event = request.get("event")
        if event == "reset":
            self.task = request.get("task")
            self.history = []
            return {}
        if event == "reflect":
            return {"reflection": f"{self.agent} bridge completed EduPlanBench episode reflection."}
        if event != "act":
            return {}

        observation = request["observation"]
        task = request.get("task") or self.task or {}
        fallback = bridge_policy.fallback_action(self.agent, task, observation)
        llm_action = self._llm_action(task, observation, fallback)
        response = llm_action if llm_action is not None else fallback
        response = bridge_policy.enforce_bridge_policy(
            response,
            task,
            observation,
            fallback,
            self.history,
            self.agent,
        )
        self.history.append(
            {
                "step": observation.get("step"),
                "action_type": response.get("action_type"),
                "resource_id": response.get("resource_id"),
                "target_concepts": response.get("target_concepts", []),
            }
        )
        self.history = self.history[-20:]
        return response

    def _llm_action(
        self,
        task: dict[str, Any],
        observation: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any] | None:
        settings = get_llm_settings()
        mode = os.environ.get("EDUPLAN_EXTERNAL_BRIDGE_USE_LLM", "auto").lower()
        if mode in {"0", "false", "no", "off", "never"}:
            bridge_policy.mark_fallback(fallback, self.agent, "llm_disabled")
            return None
        if not settings["api_key"]:
            bridge_policy.mark_fallback(fallback, self.agent, "missing_api_key")
            return None
        try:
            if self.client is None:
                self.client = OpenAICompatibleClient.from_env()
                self.client.timeout = int(os.environ.get("EDUPLAN_EXTERNAL_BRIDGE_TIMEOUT", "60"))
                self.client.max_retries = int(os.environ.get("EDUPLAN_EXTERNAL_BRIDGE_RETRIES", "1"))
            prompt = bridge_policy.build_agent_prompt(self.agent, task, observation, fallback, self.history)
            payload = self.client.complete_json(prompt)
            return bridge_policy.normalize_llm_action(payload, observation, fallback, self.agent)
        except Exception as exc:
            bridge_policy.mark_fallback(fallback, self.agent, f"llm_error:{exc}")
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description="EduPlanBench JSON bridge for registered external agent systems.")
    parser.add_argument("--agent", default=os.environ.get("EDUPLAN_EXTERNAL_AGENT_NAME", "external"))
    args = parser.parse_args()

    bridge = EduPlanBridge(args.agent)
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
            response = bridge.handle(request)
        except Exception as exc:
            response = {
                "action_type": "diagnostic_quiz",
                "target_concepts": [],
                "rationale": f"Bridge recovered from error: {exc}",
                "payload": {"bridge_error": str(exc), "agent": args.agent},
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
