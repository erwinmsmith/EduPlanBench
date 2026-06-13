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


RESOURCE_ACTIONS = {
    "recommend_exercise",
    "recommend_explanation",
    "recommend_review",
    "recommend_lecture_text",
    "recommend_problem",
    "recommend_easier_problem",
    "recommend_similar_problem",
}


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


class EduPlanBridge:
    def __init__(self, agent: str) -> None:
        self.agent = normalize_agent(agent)
        self.task: dict[str, Any] | None = None

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        event = request.get("event")
        if event == "reset":
            self.task = request.get("task")
            return {}
        if event == "reflect":
            return {"reflection": f"{self.agent} bridge completed EduPlanBench episode reflection."}
        if event != "act":
            return {}
        observation = request["observation"]
        task = request.get("task") or self.task or {}
        if self.agent == "llm_pddl":
            return llm_pddl_action(task, observation)
        if self.agent == "lats":
            return lats_action(task, observation)
        if self.agent == "plan_and_act":
            return plan_and_act_action(task, observation)
        if self.agent == "reactree":
            return reactree_action(task, observation)
        if self.agent == "hiagent":
            return hiagent_action(task, observation)
        return lats_action(task, observation)


def llm_pddl_action(task: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    target = target_concepts(observation)
    mastery = observation.get("estimated_mastery", {})
    step = int(observation.get("step", 0))
    resources = observation.get("candidate_resources", [])
    weak = weak_concepts(observation, threshold=0.5)
    events = observation.get("metadata", {}).get("dynamic_events", [])

    if step == 0 or recent_event_requires_replan(observation):
        return action(
            "update_plan",
            target,
            "LLM+P bridge compiles learner state into a symbolic learning plan before acting.",
            plan_update=pddl_like_plan(target, weak, events),
            payload={"bridge_strategy": "llm_pddl", "operator": "make_plan"},
        )
    if minimum_mastery(mastery, target) < 0.3:
        return action("diagnostic_quiz", target, "LLM+P bridge resolves uncertain initial facts with a diagnostic operator.", payload={"bridge_strategy": "llm_pddl"})

    preferred = ["exercise", "problem"] if step % 4 else ["explanation", "lecture_text"]
    resource = select_resource(resources, weak or target, mastery, preferred_types=preferred, difficulty_bias=0.15)
    if not resource:
        return action("diagnostic_quiz", target, "No applicable operator resource is visible.", payload={"bridge_strategy": "llm_pddl"})
    kind = "recommend_exercise" if resource_type(resource) in {"exercise", "problem"} else "recommend_explanation"
    return resource_action(kind, resource, "LLM+P bridge executes the next applicable plan operator.", {"bridge_strategy": "llm_pddl", "operator": kind})


def lats_action(task: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    target = target_concepts(observation)
    mastery = observation.get("estimated_mastery", {})
    resources = observation.get("candidate_resources", [])
    candidates: list[dict[str, Any]] = []

    if minimum_mastery(mastery, target) < 0.28:
        candidates.append(action("diagnostic_quiz", target, "LATS candidate: diagnose uncertain learner state.", payload={"value": 0.58}))
    candidates.append(action("update_plan", target, "LATS candidate: refresh search root plan.", plan_update=search_plan_text(observation), payload={"value": 0.35}))
    for resource in resources[:12]:
        for kind in action_types_for_resource(resource):
            score = score_resource(resource, target, mastery, observation)
            if kind == "recommend_review" and feedback_has(observation, "incorrect"):
                score += 0.18
            if kind == "recommend_explanation" and minimum_mastery(mastery, resource.get("concepts") or target) < 0.45:
                score += 0.12
            candidates.append(resource_action(kind, resource, "LATS candidate action from tree expansion.", {"value": score, "bridge_strategy": "lats"}))
    best = max(candidates, key=lambda item: float(item.get("payload", {}).get("value", 0.0))) if candidates else action("diagnostic_quiz", target, "LATS fallback diagnostic.")
    best["rationale"] = "LATS bridge selected the highest-value action after candidate expansion and rollout-style scoring."
    best.setdefault("payload", {})["bridge_strategy"] = "lats"
    return best


def plan_and_act_action(task: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    target = target_concepts(observation)
    mastery = observation.get("estimated_mastery", {})
    resources = observation.get("candidate_resources", [])
    step = int(observation.get("step", 0))
    if step == 0 or recent_event_requires_replan(observation):
        return action(
            "update_plan",
            target,
            "Plan-and-Act bridge first creates or repairs a high-level plan before execution.",
            plan_update=plan_and_act_plan(observation),
            payload={"bridge_strategy": "plan_and_act", "phase": "planner"},
        )
    if feedback_has(observation, "incorrect") or minimum_mastery(mastery, target) < 0.42:
        resource = select_resource(resources, weak_concepts(observation, threshold=0.55) or target, mastery, preferred_types=["explanation", "lecture_text", "exercise"], difficulty_bias=0.05)
        if resource:
            kind = "recommend_review" if resource_type(resource) in {"exercise", "problem"} else "recommend_explanation"
            return resource_action(kind, resource, "Plan-and-Act bridge executor chooses remediation under the current plan.", {"bridge_strategy": "plan_and_act", "phase": "executor"})
    resource = select_resource(resources, target, mastery, preferred_types=["exercise", "problem"], difficulty_bias=0.18)
    if resource:
        return resource_action("recommend_exercise", resource, "Plan-and-Act bridge executor advances the active plan with a target exercise.", {"bridge_strategy": "plan_and_act", "phase": "executor"})
    return action("diagnostic_quiz", target, "Plan-and-Act bridge needs more state before executing.", payload={"bridge_strategy": "plan_and_act"})


def reactree_action(task: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    target = target_concepts(observation)
    mastery = observation.get("estimated_mastery", {})
    resources = observation.get("candidate_resources", [])
    step = int(observation.get("step", 0))
    weak = weak_concepts(observation, threshold=0.58)
    if step == 0:
        return action(
            "update_plan",
            target,
            "ReAcTree bridge decomposes the learning objective into diagnostic, remediation, practice, and fallback nodes.",
            plan_update=reactree_plan(target, weak),
            payload={"bridge_strategy": "reactree", "node": "root_sequence"},
        )
    if feedback_has(observation, "incorrect") or feedback_has(observation, "forgotten"):
        resource = select_resource(resources, weak or target, mastery, preferred_types=["explanation", "lecture_text", "exercise"], difficulty_bias=-0.05)
        if resource:
            return resource_action("recommend_review", resource, "ReAcTree bridge activates fallback remediation node after negative feedback.", {"bridge_strategy": "reactree", "node": "fallback_review"})
    if minimum_mastery(mastery, target) < 0.34:
        return action("diagnostic_quiz", target, "ReAcTree bridge expands a diagnostic child because the root state is uncertain.", payload={"bridge_strategy": "reactree", "node": "diagnostic"})
    resource = select_resource(resources, weak or target, mastery, preferred_types=["exercise", "problem", "explanation"], difficulty_bias=0.12)
    if resource:
        kind = "recommend_exercise" if resource_type(resource) in {"exercise", "problem"} else "recommend_explanation"
        return resource_action(kind, resource, "ReAcTree bridge executes the active child node in the hierarchy.", {"bridge_strategy": "reactree", "node": "active_child"})
    return action("wait_or_reduce_load", target, "ReAcTree bridge reduces load when no child resource is available.", payload={"bridge_strategy": "reactree"})


def hiagent_action(task: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    target = target_concepts(observation)
    mastery = observation.get("estimated_mastery", {})
    resources = observation.get("candidate_resources", [])
    memory = build_memory_summary(observation)
    weak = weak_concepts(observation, threshold=0.55)
    if int(observation.get("step", 0)) == 0:
        return action(
            "update_plan",
            target,
            "HiAgent bridge initializes hierarchical working memory before choosing resources.",
            plan_update=hiagent_plan(target, memory),
            payload={"bridge_strategy": "hiagent", "memory": memory},
        )
    if feedback_has(observation, "time budget reduced"):
        resource = select_resource(resources, target, mastery, preferred_types=["exercise", "problem"], difficulty_bias=-0.05)
        if resource:
            return resource_action("recommend_easier_problem", resource, "HiAgent bridge compresses the plan after time-budget memory update.", {"bridge_strategy": "hiagent", "memory": memory})
        return action("wait_or_reduce_load", target, "HiAgent bridge reduces load after time-budget memory update.", payload={"bridge_strategy": "hiagent", "memory": memory})
    if feedback_has(observation, "incorrect") or weak:
        resource = select_resource(resources, weak or target, mastery, preferred_types=["explanation", "lecture_text", "exercise"], difficulty_bias=0.0)
        if resource:
            kind = "recommend_review" if resource_type(resource) in {"exercise", "problem"} else "recommend_explanation"
            return resource_action(kind, resource, "HiAgent bridge uses working memory to address the current weak concept.", {"bridge_strategy": "hiagent", "memory": memory})
    resource = select_resource(resources, target, mastery, preferred_types=["exercise", "problem"], difficulty_bias=0.16)
    if resource:
        return resource_action("recommend_exercise", resource, "HiAgent bridge selects from memory-aligned target resources.", {"bridge_strategy": "hiagent", "memory": memory})
    return action("diagnostic_quiz", target, "HiAgent bridge asks for diagnostic memory refresh.", payload={"bridge_strategy": "hiagent", "memory": memory})


def action(action_type: str, concepts: list[str], rationale: str, *, plan_update: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "resource_id": None,
        "target_concepts": concepts,
        "rationale": rationale,
        "plan_update": plan_update,
        "payload": payload or {},
    }


def resource_action(action_type: str, resource: dict[str, Any], rationale: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    concepts = list(resource.get("concepts") or [])
    return {
        "action_type": action_type,
        "resource_id": resource.get("resource_id"),
        "target_concepts": concepts,
        "rationale": rationale,
        "plan_update": "",
        "payload": payload or {},
    }


def target_concepts(observation: dict[str, Any]) -> list[str]:
    goal = observation.get("goal", {})
    return list(goal.get("target_concepts") or [])


def minimum_mastery(mastery: dict[str, Any], concepts: list[str]) -> float:
    if not concepts:
        return 0.0
    return min(float(mastery.get(concept, 0.3)) for concept in concepts)


def weak_concepts(observation: dict[str, Any], *, threshold: float) -> list[str]:
    mastery = observation.get("estimated_mastery", {})
    target = target_concepts(observation)
    concepts = target or list(mastery)
    return [concept for concept in concepts if float(mastery.get(concept, 0.0)) < threshold]


def feedback_has(observation: dict[str, Any], needle: str) -> bool:
    needle_l = needle.lower()
    return any(needle_l in str(item).lower() for item in observation.get("recent_feedback", []))


def recent_event_requires_replan(observation: dict[str, Any]) -> bool:
    if feedback_has(observation, "resource became unavailable") or feedback_has(observation, "time budget reduced"):
        return True
    events = observation.get("metadata", {}).get("dynamic_events", [])
    return bool(events and int(observation.get("step", 0)) in {int(item.get("step", -999)) for item in events})


def resource_type(resource: dict[str, Any]) -> str:
    return str(resource.get("type") or "").lower()


def action_types_for_resource(resource: dict[str, Any]) -> list[str]:
    rtype = resource_type(resource)
    if rtype in {"exercise", "problem"}:
        return ["recommend_exercise", "recommend_review"]
    if rtype in {"lecture_text", "explanation"}:
        return ["recommend_explanation"]
    return ["recommend_explanation", "recommend_review"]


def select_resource(
    resources: list[dict[str, Any]],
    concepts: list[str],
    mastery: dict[str, Any],
    *,
    preferred_types: list[str],
    difficulty_bias: float,
) -> dict[str, Any] | None:
    if not resources:
        return None
    preferred = {item.lower() for item in preferred_types}
    scored = []
    for resource in resources:
        score = score_resource(resource, concepts, mastery, {"recent_feedback": []}, difficulty_bias=difficulty_bias)
        if resource_type(resource) in preferred:
            score += 0.25
        scored.append((score, resource))
    return max(scored, key=lambda item: item[0])[1]


def score_resource(
    resource: dict[str, Any],
    concepts: list[str],
    mastery: dict[str, Any],
    observation: dict[str, Any],
    *,
    difficulty_bias: float = 0.12,
) -> float:
    r_concepts = list(resource.get("concepts") or [])
    overlap = len(set(r_concepts).intersection(concepts or r_concepts))
    avg = sum(float(mastery.get(concept, 0.35)) for concept in r_concepts or concepts or ["_"]) / max(1, len(r_concepts or concepts or ["_"]))
    difficulty = float(resource.get("difficulty", 0.5))
    desired = min(0.85, max(0.15, avg + difficulty_bias))
    score = 0.35 * overlap + 0.4 * (1.0 - abs(difficulty - desired))
    if feedback_has(observation, "incorrect") and difficulty > avg + 0.25:
        score -= 0.25
    return score


def pddl_like_plan(target: list[str], weak: list[str], events: list[dict[str, Any]]) -> str:
    weak_text = ", ".join(weak or target)
    event_text = ", ".join(str(item.get("type", "")) for item in events) or "none"
    return (
        "(:goal improve-target-mastery)\n"
        f"(:objects concepts={','.join(target)} weak={weak_text})\n"
        f"(:observed-events {event_text})\n"
        "plan: diagnose-state -> review-weak-concepts -> explain-target -> practice-target -> retention-review"
    )


def search_plan_text(observation: dict[str, Any]) -> str:
    return "search tree root: expand diagnostic, remediation, practice, and review actions; select by expected mastery gain and risk."


def plan_and_act_plan(observation: dict[str, Any]) -> str:
    return "planner: diagnose readiness; executor: remediate current weakness; executor: assign target practice; replan on incorrect feedback or resource/time perturbation."


def reactree_plan(target: list[str], weak: list[str]) -> str:
    return f"root(sequence): diagnostic -> fallback(review {weak or target}) -> active_child(practice {target}) -> retention_check."


def hiagent_plan(target: list[str], memory: dict[str, Any]) -> str:
    return f"memory hierarchy: goal={target}; weak={memory.get('weak_concepts')}; feedback={memory.get('recent_feedback_labels')}; plan=refresh memory -> remediate -> practice -> retain."


def build_memory_summary(observation: dict[str, Any]) -> dict[str, Any]:
    feedback = [str(item).lower() for item in observation.get("recent_feedback", [])]
    labels = []
    for label in ["incorrect", "correct", "forgotten", "resource became unavailable", "time budget reduced"]:
        if any(label in item for item in feedback):
            labels.append(label)
    return {
        "weak_concepts": weak_concepts(observation, threshold=0.55),
        "recent_feedback_labels": labels,
        "step": observation.get("step", 0),
    }


def normalize_agent(name: str) -> str:
    name = name.lower()
    if name.startswith("external:"):
        name = name.split(":", 1)[1]
    return name.replace("-", "_")


if __name__ == "__main__":
    main()
