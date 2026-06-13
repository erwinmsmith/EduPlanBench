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

from eduplanbench.core.env import get_llm_settings
from eduplanbench.core.schema import Action
from eduplanbench.llm.openai_compatible import OpenAICompatibleClient


RESOURCE_ACTIONS = {
    "recommend_exercise",
    "recommend_explanation",
    "recommend_review",
    "recommend_lecture_text",
    "recommend_problem",
    "recommend_easier_problem",
    "recommend_similar_problem",
}

NON_LEARNING_ACTIONS = {
    "diagnostic_quiz",
    "recommend_diagnostic",
    "diagnose_misconception",
    "update_plan",
    "update_path",
    "wait_or_reduce_load",
    "summarize_knowledge",
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
        fallback = self._fallback_action(task, observation)
        llm_action = self._llm_action(task, observation, fallback)
        response = llm_action if llm_action is not None else fallback
        response = enforce_bridge_policy(response, task, observation, fallback, self.history, self.agent)
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

    def _fallback_action(self, task: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
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

    def _llm_action(self, task: dict[str, Any], observation: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any] | None:
        settings = get_llm_settings()
        mode = os.environ.get("EDUPLAN_EXTERNAL_BRIDGE_USE_LLM", "auto").lower()
        if mode in {"0", "false", "no", "off", "never"}:
            mark_fallback(fallback, self.agent, "llm_disabled")
            return None
        if not settings["api_key"]:
            mark_fallback(fallback, self.agent, "missing_api_key")
            return None
        try:
            if self.client is None:
                self.client = OpenAICompatibleClient.from_env()
                self.client.timeout = int(os.environ.get("EDUPLAN_EXTERNAL_BRIDGE_TIMEOUT", "60"))
                self.client.max_retries = int(os.environ.get("EDUPLAN_EXTERNAL_BRIDGE_RETRIES", "1"))
            prompt = build_agent_prompt(self.agent, task, observation, fallback, self.history)
            payload = self.client.complete_json(prompt)
            return normalize_llm_action(payload, observation, fallback, self.agent)
        except Exception as exc:
            mark_fallback(fallback, self.agent, f"llm_error:{exc}")
            return None


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


def build_agent_prompt(agent: str, task: dict[str, Any], observation: dict[str, Any], fallback: dict[str, Any], history: list[dict[str, Any]] | None = None) -> str:
    family = prompt_family(agent)
    history = history or []
    context = compact_context(task, observation, fallback, history)
    return (
        f"{family['role']}\n\n"
        "You are controlling an EduPlanBench learning-planning episode. "
        "The task is long-horizon: you must use learner feedback, visible mastery, dynamic events, and candidate resources to choose the next step. "
        "Do not solve the math problem for the learner unless the action is an explanation/review and the hint remains pedagogically appropriate.\n\n"
        f"{family['method']}\n\n"
        f"{track_policy(task, observation, history)}\n\n"
        "Return exactly one strict JSON object with this schema:\n"
        "{\n"
        '  "action_type": "one of recommend_exercise, recommend_explanation, recommend_review, diagnostic_quiz, update_plan, diagnose_misconception, wait_or_reduce_load, recommend_easier_problem, recommend_similar_problem, summarize_knowledge",\n'
        '  "resource_id": "required for resource recommendation actions; must be one of candidate_resources[].resource_id, otherwise null",\n'
        '  "target_concepts": ["concept ids/names"],\n'
        '  "rationale": "one concise sentence grounded in observation/feedback/resources",\n'
        '  "plan_update": "non-empty only when action_type is update_plan or when replanning is necessary",\n'
        '  "payload": {"prompt_family": "...", "diagnosis": "... optional", "expected_effect": "... optional"}\n'
        "}\n\n"
        "Legality constraints:\n"
        "- If action_type recommends a resource, choose an existing resource_id from candidate_resources only.\n"
        "- Never output null resource_id for recommend_exercise/recommend_explanation/recommend_review/recommend_easier_problem/recommend_similar_problem.\n"
        "- diagnostic_quiz, diagnose_misconception, update_plan, and wait_or_reduce_load do not directly increase mastery; do not repeat them when resources are available.\n"
        "- Avoid repeating the same resource more than twice unless recent feedback explicitly says it helped.\n"
        "- If recent feedback shows incorrect/forgotten/overload, prefer diagnostic, review, easier problem, or replan over harder practice.\n"
        "- If a resource is unavailable or time budget changed, update the plan or select an alternative resource.\n"
        "- Keep the rationale short; do not include hidden chain-of-thought.\n\n"
        f"EduPlanBench context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )


def track_policy(task: dict[str, Any], observation: dict[str, Any], history: list[dict[str, Any]]) -> str:
    track = str(task.get("track") or observation.get("metadata", {}).get("track") or "")
    repeated_non_learning = recent_non_learning_count(history)
    common = (
        f"Recent non-learning action streak: {repeated_non_learning}. "
        "If the streak is >=2 and a valid candidate resource exists, choose a resource action now."
    )
    if track == "track1_text_math":
        return (
            "Track-specific policy for Track 1 Text-Math: diagnose the misconception early, then act on it. "
            "After one diagnosis/diagnostic step, use Eedi/MathDial/Misstep candidate resources for recommend_explanation, recommend_review, "
            "recommend_easier_problem, or recommend_similar_problem. Repeating diagnose_misconception without remediation is a failure mode. "
            f"{common}"
        )
    if track == "track2_mooc_planning":
        return (
            "Track-specific policy for Track 2 MOOC Planning: this is prerequisite/resource path planning, not misconception tutoring. "
            "Do not use diagnose_misconception. Use diagnostic_quiz at most once, then recommend prerequisite resources before target resources. "
            "Minimize prerequisite violations, plan drift, repeated resources, and over-budget wandering. "
            f"{common}"
        )
    if track == "track3_kt_simulator":
        return (
            "Track-specific policy for Track 3 KT Simulator: actions must affect the simulator through target-concept exercises, review, or explanations. "
            "Use diagnostic_quiz sparingly to improve visible estimates, then recommend exercise/review resources near learner readiness. "
            "Avoid repeated diagnostics and repeated identical resources. "
            f"{common}"
        )
    return common


def prompt_family(agent: str) -> dict[str, str]:
    if agent == "llm_pddl":
        return {
            "role": "Prompt family: LLM+P for EduPlanBench.",
            "method": (
                "Represent the learner state as symbolic facts and choose the next applicable planning operator. "
                "Use operators such as diagnose_state, repair_misconception, review_prerequisite, explain_target, practice_target, retention_review, and replan_after_event. "
                "Prefer update_plan when no valid operator sequence exists for the current dynamic constraints."
            ),
        }
    if agent == "lats":
        return {
            "role": "Prompt family: LATS for EduPlanBench.",
            "method": (
                "Mentally expand a small tree of candidate actions, estimate value by expected mastery gain, risk, validity, and long-horizon progress, then return only the best action. "
                "Do not output the search tree; summarize only the selected action rationale."
            ),
        }
    if agent == "plan_and_act":
        return {
            "role": "Prompt family: Plan-and-Act for EduPlanBench.",
            "method": (
                "Separate high-level planning from execution. If the plan is missing, stale, or contradicted by feedback/events, output update_plan. "
                "Otherwise act as the executor and choose the next resource/action aligned with the current plan."
            ),
        }
    if agent == "reactree":
        return {
            "role": "Prompt family: ReAcTree for EduPlanBench.",
            "method": (
                "Decompose the goal into a hierarchical control tree: root sequence, diagnostic child, remediation fallback, target-practice child, retention child. "
                "Choose the active node based on feedback; use fallback nodes after errors or overload."
            ),
        }
    if agent == "hiagent":
        return {
            "role": "Prompt family: HiAgent for EduPlanBench.",
            "method": (
                "Maintain hierarchical working memory: goal memory, weak-concept memory, recent-feedback memory, resource memory, and constraint memory. "
                "Choose actions that update or use memory to avoid repeating failures and adapt under long-horizon perturbations. "
                "If observation.step is 0, initialize working memory with action_type update_plan and do not recommend a resource yet."
            ),
        }
    return {
        "role": "Prompt family: generic external planner for EduPlanBench.",
        "method": "Choose the next action that maximizes long-horizon learning progress while respecting resource and action constraints.",
    }


def compact_context(task: dict[str, Any], observation: dict[str, Any], fallback: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    resources = []
    for resource in observation.get("candidate_resources", [])[:15]:
        resources.append(
            {
                "resource_id": resource.get("resource_id"),
                "type": resource.get("type"),
                "title": clip(str(resource.get("title", "")), 120),
                "concepts": list(resource.get("concepts") or []),
                "difficulty": resource.get("difficulty", 0.5),
                "text": clip(str(resource.get("text", "")), 260),
            }
        )
    return {
        "task": {
            "task_id": task.get("task_id") or observation.get("task_id"),
            "track": task.get("track") or observation.get("metadata", {}).get("track"),
            "domain": task.get("domain"),
            "horizon": task.get("horizon"),
            "constraints": task.get("constraints", {}),
            "metadata": {
                "task_type": task.get("metadata", {}).get("task_type"),
                "difficulty": task.get("metadata", {}).get("difficulty"),
                "misconception": task.get("metadata", {}).get("misconception"),
            },
        },
        "observation": {
            "step": observation.get("step"),
            "goal": observation.get("goal", {}),
            "learner_summary": clip(str(observation.get("learner_summary", "")), 500),
            "estimated_mastery": observation.get("estimated_mastery", {}),
            "recent_feedback": [clip(str(item), 220) for item in observation.get("recent_feedback", [])[-5:]],
            "available_actions": observation.get("available_actions", []),
            "current_plan": clip(str(observation.get("current_plan", "")), 600),
            "metadata": {
                "active_horizon": observation.get("metadata", {}).get("active_horizon"),
                "dynamic_events": observation.get("metadata", {}).get("dynamic_events", [])[-5:],
                "unavailable_resources": observation.get("metadata", {}).get("unavailable_resources", []),
                "prerequisites": observation.get("metadata", {}).get("prerequisites", []),
            },
            "candidate_resources": resources,
            "action_history": history_summary(history),
        },
        "deterministic_fallback_action": fallback,
    }


def history_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    resource_counts: dict[str, int] = {}
    for item in history:
        action_type = str(item.get("action_type") or "")
        resource_id = str(item.get("resource_id") or "")
        if action_type:
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        if resource_id:
            resource_counts[resource_id] = resource_counts.get(resource_id, 0) + 1
    return {
        "recent": history[-6:],
        "action_counts": action_counts,
        "resource_counts": resource_counts,
        "non_learning_streak": recent_non_learning_count(history),
    }


def normalize_llm_action(payload: dict[str, Any], observation: dict[str, Any], fallback: dict[str, Any], agent: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        mark_fallback(fallback, agent, "llm_non_object")
        return fallback
    candidate_ids = {resource.get("resource_id") for resource in observation.get("candidate_resources", [])}
    action_type = str(payload.get("action_type") or fallback.get("action_type") or "diagnostic_quiz")
    if action_type not in Action.VALID_ACTIONS:
        mark_fallback(fallback, agent, f"llm_invalid_action_type:{action_type}")
        return fallback
    resource_id = payload.get("resource_id")
    repaired_resource = False
    if action_type in RESOURCE_ACTIONS and resource_id not in candidate_ids:
        fallback_resource_id = fallback.get("resource_id")
        if fallback.get("action_type") in RESOURCE_ACTIONS and fallback_resource_id in candidate_ids:
            resource_id = fallback_resource_id
            repaired_resource = True
        else:
            mark_fallback(fallback, agent, f"llm_invalid_resource:{resource_id}")
            return fallback
    result = {
        "action_type": action_type,
        "resource_id": resource_id if resource_id else None,
        "target_concepts": list(payload.get("target_concepts") or fallback.get("target_concepts") or target_concepts(observation)),
        "rationale": str(payload.get("rationale") or fallback.get("rationale") or ""),
        "plan_update": str(payload.get("plan_update") or ""),
        "payload": dict(payload.get("payload") or {}),
    }
    result["payload"].setdefault("bridge_strategy", agent)
    result["payload"]["prompt_family"] = agent
    result["payload"]["llm_used"] = True
    if repaired_resource:
        result["payload"]["llm_resource_repaired"] = True
    return result


def enforce_bridge_policy(
    payload: dict[str, Any],
    task: dict[str, Any],
    observation: dict[str, Any],
    fallback: dict[str, Any],
    history: list[dict[str, Any]],
    agent: str,
) -> dict[str, Any]:
    track = str(task.get("track") or observation.get("metadata", {}).get("track") or "")
    action_type = str(payload.get("action_type") or "")
    repair_reason = ""
    if track == "track2_mooc_planning" and action_type == "diagnose_misconception":
        repair_reason = "track2_no_misconception_diagnosis"
    elif action_type in NON_LEARNING_ACTIONS and recent_non_learning_count(history) >= 2 and observation.get("candidate_resources"):
        repair_reason = "repeated_non_learning_action"
    elif payload.get("resource_id") and repeated_resource_count(history, str(payload.get("resource_id"))) >= 2:
        repair_reason = "repeated_resource"
    if not repair_reason:
        return payload
    repaired = policy_resource_action(task, observation, agent, repair_reason)
    if repaired is not None:
        return repaired
    payload.setdefault("payload", {})["policy_warning"] = repair_reason
    return payload


def policy_resource_action(task: dict[str, Any], observation: dict[str, Any], agent: str, reason: str) -> dict[str, Any] | None:
    track = str(task.get("track") or observation.get("metadata", {}).get("track") or "")
    mastery = observation.get("estimated_mastery", {})
    resources = observation.get("candidate_resources", [])
    target = target_concepts(observation)
    if track == "track2_mooc_planning":
        prereqs = list(observation.get("metadata", {}).get("prerequisites") or task.get("constraints", {}).get("prerequisites") or [])
        weak_prereqs = [concept for concept in prereqs if float(mastery.get(concept, 0.0)) < 0.6]
        concepts = weak_prereqs or prereqs[:1] or target
        resource = select_resource(resources, concepts, mastery, preferred_types=["lecture_text", "explanation", "exercise", "problem"], difficulty_bias=0.0)
        if resource:
            kind = "recommend_review" if resource_type(resource) in {"exercise", "problem"} and concepts != target else "recommend_explanation"
            return resource_action(kind, resource, f"Bridge repaired policy violation ({reason}) by selecting a prerequisite/path resource.", {"bridge_strategy": agent, "policy_repair": reason})
    elif track == "track1_text_math":
        resource = select_resource(resources, weak_concepts(observation, threshold=0.6) or target, mastery, preferred_types=["explanation", "lecture_text", "exercise", "problem"], difficulty_bias=-0.05)
        if resource:
            kind = "recommend_review" if resource_type(resource) in {"exercise", "problem"} else "recommend_explanation"
            return resource_action(kind, resource, f"Bridge repaired repeated diagnosis ({reason}) with misconception remediation.", {"bridge_strategy": agent, "policy_repair": reason})
    elif track == "track3_kt_simulator":
        resource = select_resource(resources, target, mastery, preferred_types=["exercise", "problem", "explanation"], difficulty_bias=0.12)
        if resource:
            kind = "recommend_exercise" if resource_type(resource) in {"exercise", "problem"} else "recommend_explanation"
            return resource_action(kind, resource, f"Bridge repaired policy violation ({reason}) with a simulator-effective resource.", {"bridge_strategy": agent, "policy_repair": reason})
    return None


def mark_fallback(action_payload: dict[str, Any], agent: str, reason: str) -> None:
    payload = action_payload.setdefault("payload", {})
    payload.setdefault("bridge_strategy", agent)
    payload.setdefault("prompt_family", agent)
    payload["llm_used"] = False
    payload["fallback_reason"] = reason


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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


def recent_non_learning_count(history: list[dict[str, Any]]) -> int:
    count = 0
    for item in reversed(history):
        if str(item.get("action_type") or "") not in NON_LEARNING_ACTIONS:
            break
        count += 1
    return count


def repeated_resource_count(history: list[dict[str, Any]], resource_id: str) -> int:
    count = 0
    for item in reversed(history):
        if str(item.get("resource_id") or "") != resource_id:
            break
        count += 1
    return count


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
