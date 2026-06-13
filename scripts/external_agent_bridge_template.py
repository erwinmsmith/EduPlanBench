from __future__ import annotations

import json
import os
import sys


def main() -> None:
    request = json.loads(sys.stdin.read())
    # EduPlanBench forwards one unified OpenAI-compatible LLM config to stdio
    # bridges. Third-party clients can read these without benchmark-specific code.
    llm_config = {
        "api_key": os.environ.get("EDUPLAN_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        "base_url": os.environ.get("EDUPLAN_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        "model": os.environ.get("EDUPLAN_LLM_MODEL") or os.environ.get("OPENAI_MODEL"),
    }
    event = request.get("event")
    if event in {"reset", "reflect"}:
        print("{}")
        return
    if event != "act":
        raise ValueError(f"unknown event: {event}")

    observation = request["observation"]

    # Replace this block with calls into a third-party agent implementation.
    # The bridge must return one EduPlanBench Action JSON object.
    resources = observation.get("candidate_resources", [])
    if resources:
        resource = resources[0]
        action = {
            "action_type": "recommend_exercise" if resource.get("type") == "exercise" else "recommend_explanation",
            "resource_id": resource["resource_id"],
            "target_concepts": resource.get("concepts", observation["goal"]["target_concepts"]),
            "rationale": "Bridge template fallback action. Replace with the external agent decision.",
            "payload": {"bridge_template": True, "llm_model": llm_config["model"]},
        }
    else:
        action = {
            "action_type": "diagnostic_quiz",
            "target_concepts": observation["goal"]["target_concepts"],
            "rationale": "Bridge template fallback action. Replace with the external agent decision.",
            "payload": {"bridge_template": True, "llm_model": llm_config["model"]},
        }
    print(json.dumps(action, ensure_ascii=False))


if __name__ == "__main__":
    main()
