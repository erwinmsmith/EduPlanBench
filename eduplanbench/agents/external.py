from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eduplanbench.core.env import build_external_llm_env, get_llm_settings
from eduplanbench.core.schema import Action, Observation, TaskInstance, to_plain


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "external_agents.json"


@dataclass(slots=True)
class ExternalAgentSpec:
    name: str
    display_name: str
    repo_url: str
    repo_path: Path
    enabled: bool
    protocol: str
    command: list[str]
    cwd: Path
    endpoint: str = ""
    timeout_seconds: int = 300
    notes: str = ""


def load_external_agent_specs(config_path: Path | None = None) -> dict[str, ExternalAgentSpec]:
    path = config_path or Path(os.environ.get("EDUPLAN_EXTERNAL_AGENTS_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = {}
    for name, raw in payload.get("agents", {}).items():
        repo_path = _resolve_path(str(raw.get("repo_path", "")))
        cwd = _resolve_template(str(raw.get("cwd") or "{repo_path}"), repo_path=repo_path)
        command = [_resolve_template(str(item), repo_path=repo_path) for item in raw.get("command", [])]
        specs[name.lower()] = ExternalAgentSpec(
            name=name.lower(),
            display_name=str(raw.get("display_name") or name),
            repo_url=str(raw.get("repo_url") or ""),
            repo_path=repo_path,
            enabled=bool(raw.get("enabled", False)),
            protocol=str(raw.get("protocol") or "stdio_json"),
            command=command,
            cwd=Path(cwd),
            endpoint=str(raw.get("endpoint") or ""),
            timeout_seconds=int(raw.get("timeout_seconds", 300)),
            notes=str(raw.get("notes") or ""),
        )
    return specs


def is_external_agent(name: str) -> bool:
    normalized = _normalize_external_name(name)
    return normalized in load_external_agent_specs()


def external_agent_status() -> list[dict[str, Any]]:
    rows = []
    for spec in load_external_agent_specs().values():
        rows.append(
            {
                "name": spec.name,
                "display_name": spec.display_name,
                "enabled": spec.enabled,
                "protocol": spec.protocol,
                "repo_path": str(spec.repo_path),
                "repo_exists": (spec.repo_path / ".git").exists(),
                "command": " ".join(spec.command),
                "notes": spec.notes,
            }
        )
    return rows


class ExternalAgentAdapter:
    """Adapter for third-party agents exposed through a JSON bridge.

    The adapter does not reimplement the third-party algorithm. It only sends
    EduPlanBench state to an external process/server and normalizes the returned
    JSON into the benchmark Action schema.
    """

    system_name = "external"

    def __init__(self, name: str, *, config_path: Path | None = None) -> None:
        self.name = _normalize_external_name(name)
        specs = load_external_agent_specs(config_path)
        if self.name not in specs:
            available = ", ".join(sorted(specs)) or "<none>"
            raise ValueError(f"unknown external agent: {name}; available external agents: {available}")
        self.spec = specs[self.name]
        if not self.spec.enabled:
            raise RuntimeError(
                f"external agent '{self.name}' is registered but disabled. "
                f"Clone/setup its repo, implement its JSON bridge, then set enabled=true in {DEFAULT_CONFIG}. "
                f"Notes: {self.spec.notes}"
            )
        if self.spec.protocol not in {"stdio_json", "http_json"}:
            raise ValueError(f"unsupported external agent protocol for {self.name}: {self.spec.protocol}")
        self.task: TaskInstance | None = None

    def reset(self, task: TaskInstance) -> None:
        self.task = task
        self._send({"event": "reset", "agent": self.name, "llm": _public_llm_settings(), "task": to_plain(task)}, required=False)

    def act(self, observation: Observation) -> Action:
        request = {
            "event": "act",
            "agent": self.name,
            "llm": _public_llm_settings(),
            "task": to_plain(self.task) if self.task else None,
            "observation": to_plain(observation),
            "action_schema": {
                "action_type": sorted(Action.VALID_ACTIONS),
                "resource_id": "required for resource recommendation actions; must be one of candidate_resources.resource_id",
                "target_concepts": "list of concept ids/names",
                "rationale": "short evidence-grounded rationale",
                "plan_update": "optional plan update",
                "payload": "optional diagnostic/debug fields",
            },
        }
        payload = self._send(request, required=True)
        action = _action_from_payload(payload, observation)
        valid, _ = action.validate_for(observation.candidate_resources)
        if not valid and observation.candidate_resources and action.action_type not in {"diagnostic_quiz", "update_plan", "diagnose_misconception", "wait_or_reduce_load"}:
            resource = observation.candidate_resources[0]
            action.resource_id = resource.resource_id
            action.target_concepts = action.target_concepts or resource.concepts
            action.payload["fallback_normalized"] = True
        action.payload.setdefault("external_agent", self.name)
        return action

    def reflect(self, trace) -> str:
        payload = self._send({"event": "reflect", "agent": self.name, "llm": _public_llm_settings(), "trace": to_plain(trace)}, required=False)
        return str(payload.get("reflection", "")) if payload else ""

    def _send(self, request: dict[str, Any], *, required: bool) -> dict[str, Any]:
        if self.spec.protocol == "stdio_json":
            return self._send_stdio(request, required=required)
        if self.spec.protocol == "http_json":
            return self._send_http(request, required=required)
        raise ValueError(self.spec.protocol)

    def _send_stdio(self, request: dict[str, Any], *, required: bool) -> dict[str, Any]:
        if not self.spec.command:
            if required:
                raise RuntimeError(f"external agent '{self.name}' has no command configured")
            return {}
        env = build_external_llm_env(os.environ.copy())
        env["EDUPLAN_EXTERNAL_AGENT_NAME"] = self.name
        env["EDUPLAN_EXTERNAL_REPO_PATH"] = str(self.spec.repo_path)
        completed = subprocess.run(
            self.spec.command,
            input=json.dumps(request, ensure_ascii=False),
            capture_output=True,
            text=True,
            cwd=self.spec.cwd,
            env=env,
            timeout=self.spec.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            if required:
                raise RuntimeError(
                    f"external agent '{self.name}' failed with code {completed.returncode}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            return {}
        stdout = completed.stdout.strip()
        if not stdout:
            return {}
        return json.loads(stdout)

    def _send_http(self, request: dict[str, Any], *, required: bool) -> dict[str, Any]:
        if not self.spec.endpoint:
            if required:
                raise RuntimeError(f"external agent '{self.name}' has no endpoint configured")
            return {}
        body = json.dumps(request, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.spec.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.spec.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _action_from_payload(payload: dict[str, Any], observation: Observation) -> Action:
    return Action(
        action_type=str(payload.get("action_type", "diagnostic_quiz")),
        resource_id=payload.get("resource_id"),
        target_concepts=list(payload.get("target_concepts") or observation.goal.target_concepts),
        rationale=str(payload.get("rationale") or payload.get("reasoning_summary") or ""),
        plan_update=str(payload.get("plan_update", "")),
        payload=dict(payload.get("payload") or payload),
    )


def _public_llm_settings() -> dict[str, str]:
    settings = get_llm_settings()
    return {
        "provider": settings["provider"],
        "base_url": settings["base_url"],
        "model": settings["model"],
    }


def _normalize_external_name(name: str) -> str:
    normalized = name.lower()
    if normalized.startswith("external:"):
        normalized = normalized.split(":", 1)[1]
    return normalized.replace("-", "_")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_template(value: str, *, repo_path: Path) -> str:
    return value.format(repo_root=PROJECT_ROOT, repo_path=repo_path, python=os.environ.get("PYTHON", "python"))
