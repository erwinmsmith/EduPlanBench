from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from eduplanbench.agents import bridge_policy
from eduplanbench.core.schema import Action, Observation, Resource, TaskInstance, to_plain
from eduplanbench.llm.openai_compatible import OpenAICompatibleClient


class Agent(ABC):
    system_name = "agent"

    def reset(self, task: TaskInstance) -> None:
        self.task = task

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError

    def reflect(self, trace) -> str:
        return ""


class KTRecommenderAgent(Agent):
    system_name = "kt_recommender"

    def act(self, observation: Observation) -> Action:
        target = observation.goal.target_concepts[0]
        mastery = observation.estimated_mastery.get(target, 0.0)
        candidates = _rank_resources(observation.candidate_resources, observation.estimated_mastery, observation.goal.target_concepts)
        if mastery < 0.3:
            return Action(
                action_type="diagnostic_quiz",
                target_concepts=observation.goal.target_concepts,
                rationale="Visible mastery is very low; diagnose before assigning more load.",
            )
        if mastery < 0.45 and candidates:
            resource = candidates[0]
            action_type = "recommend_review" if resource.type in {"exercise", "explanation", "lecture_text"} else "recommend_explanation"
            return Action(
                action_type=action_type,
                resource_id=resource.resource_id,
                target_concepts=resource.concepts,
                rationale="Target mastery is low; use prerequisite-aligned review.",
            )
        if candidates:
            resource = candidates[0]
            return Action(
                action_type="recommend_exercise" if resource.type == "exercise" else "recommend_explanation",
                resource_id=resource.resource_id,
                target_concepts=resource.concepts,
                rationale="Select resource closest to visible mastery and target concepts.",
            )
        return Action(action_type="update_plan", target_concepts=observation.goal.target_concepts, rationale="No resources available.")


class StaticOneShotAgent(Agent):
    """Non-LLM one-shot baseline."""

    system_name = "static_one_shot"

    def act(self, observation: Observation) -> Action:
        resources = observation.candidate_resources
        if observation.step == 0:
            return Action(
                action_type="update_plan",
                target_concepts=observation.goal.target_concepts,
                rationale="Create a full learning path before recommending resources.",
                plan_update="diagnose misconception -> review prerequisite -> explanation -> target exercise -> retention review",
            )
        if resources:
            resource = resources[min(observation.step - 1, len(resources) - 1)]
            return Action(
                action_type="recommend_exercise" if resource.type == "exercise" else "recommend_explanation",
                resource_id=resource.resource_id,
                target_concepts=resource.concepts,
                rationale="Follow the initial static plan without adapting to feedback.",
            )
        return Action(action_type="diagnostic_quiz", target_concepts=observation.goal.target_concepts)


@dataclass
class ToolState:
    resource_cache: list[dict[str, Any]] = field(default_factory=list)
    learner_memory: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    plan: str = ""
    action_history: list[dict[str, Any]] = field(default_factory=list)


class LLMPlannerAgent(Agent):
    system_name = "llm_planner"

    def __init__(self, client: OpenAICompatibleClient | None = None) -> None:
        self.client = client or OpenAICompatibleClient.from_env()
        self.state = ToolState()

    def reset(self, task: TaskInstance) -> None:
        super().reset(task)
        self.state = ToolState()

    def act(self, observation: Observation) -> Action:
        self._update_tools(observation)
        task_dict = to_plain(self.task)
        observation_dict = to_plain(observation)
        fallback = bridge_policy.fallback_action(self.system_name, task_dict, observation_dict)
        prompt = bridge_policy.build_agent_prompt(self.system_name, task_dict, observation_dict, fallback, self.state.action_history)
        payload = self.client.complete_json(prompt)
        action_payload = bridge_policy.normalize_llm_action(payload, observation_dict, fallback, self.system_name)
        action_payload = bridge_policy.enforce_bridge_policy(
            action_payload,
            task_dict,
            observation_dict,
            fallback,
            self.state.action_history,
            self.system_name,
        )
        action = self._action_from_payload(action_payload, observation)
        valid, _ = action.validate_for(observation.candidate_resources)
        if not valid and observation.candidate_resources and action.action_type not in {"diagnostic_quiz", "update_plan", "diagnose_misconception", "wait_or_reduce_load"}:
            resource = observation.candidate_resources[0]
            action.resource_id = resource.resource_id
            action.target_concepts = action.target_concepts or resource.concepts
            action.payload["fallback_normalized"] = True
        if action.plan_update:
            self.state.plan = action.plan_update
        self.state.action_history.append(
            {
                "step": observation.step,
                "action_type": action.action_type,
                "resource_id": action.resource_id,
                "target_concepts": action.target_concepts,
            }
        )
        self.state.action_history = self.state.action_history[-20:]
        return action

    def _update_tools(self, observation: Observation) -> None:
        self.state.learner_memory.extend(observation.recent_feedback)
        self.state.learner_memory = self.state.learner_memory[-12:]
        self.state.resource_cache = [
            {
                "resource_id": resource.resource_id,
                "type": resource.type,
                "title": resource.title,
                "concepts": resource.concepts,
                "difficulty": resource.difficulty,
                "text": resource.text[:500],
            }
            for resource in observation.candidate_resources[:15]
        ]

    def _base_context(self, observation: Observation) -> dict[str, Any]:
        return {
            "task_id": observation.task_id,
            "step": observation.step,
            "goal": {
                "target_concepts": observation.goal.target_concepts,
                "target_mastery": observation.goal.target_mastery,
                "horizon": observation.goal.horizon,
                "description": observation.goal.description,
            },
            "learner_summary": observation.learner_summary,
            "estimated_mastery": observation.estimated_mastery,
            "recent_feedback": observation.recent_feedback,
            "available_actions": observation.available_actions,
            "candidate_resources": self.state.resource_cache,
            "current_plan": observation.current_plan or self.state.plan,
            "tools": {
                "search_resources(query)": "Use candidate_resources already retrieved for this observation.",
                "get_learner_summary()": observation.learner_summary,
                "get_concept_graph(concept)": "Use target_concepts, resource concepts, and prerequisites in metadata.",
                "recommend(action)": "Return a valid JSON action.",
                "update_plan()": "Use plan_update field.",
                "reflect()": "Use rationale field and memory.",
            },
        }

    def _prompt(self, observation: Observation) -> str:
        return _json_prompt(
            "You are an EduPlanBench planner. Choose one valid next learning action.",
            self._base_context(observation),
        )

    def _action_from_payload(self, payload: dict[str, Any], observation: Observation) -> Action:
        resource = _resource_by_id(observation.candidate_resources, payload.get("resource_id"))
        target_concepts = list(payload.get("target_concepts") or (resource.concepts if resource else observation.goal.target_concepts))
        return Action(
            action_type=str(payload.get("action_type", "diagnostic_quiz")),
            resource_id=payload.get("resource_id"),
            target_concepts=target_concepts,
            rationale=str(payload.get("rationale") or payload.get("reasoning_summary") or ""),
            plan_update=str(payload.get("plan_update", "")),
            payload=payload,
        )


class ReActPlannerAgent(LLMPlannerAgent):
    system_name = "react"

    def _prompt(self, observation: Observation) -> str:
        context = self._base_context(observation)
        context["react_protocol"] = [
            "Observe learner state and recent feedback.",
            "Think about prerequisite, cognitive load, target progress, and whether a tool/action is needed.",
            "Act with exactly one JSON action.",
        ]
        return _json_prompt(
            "You are a ReAct Planner. Use observation -> concise thought summary -> action. Do not reveal long hidden reasoning.",
            context,
        )


class OneShotPlannerAgent(LLMPlannerAgent):
    system_name = "one_shot"

    def act(self, observation: Observation) -> Action:
        return super().act(observation)

    def _prompt(self, observation: Observation) -> str:
        context = self._base_context(observation)
        context["instruction"] = "Create a full horizon plan in plan_update and choose the first action. This is your only planning call."
        return _json_prompt("You are a One-shot Planner.", context)


class StepByStepPlannerAgent(LLMPlannerAgent):
    system_name = "step_by_step"

    def _prompt(self, observation: Observation) -> str:
        context = self._base_context(observation)
        context["instruction"] = "Plan only the next step. Use current feedback, mastery, and candidate resources."
        return _json_prompt("You are a Step-by-step Planner.", context)


class CoTPlannerAgent(LLMPlannerAgent):
    system_name = "cot"

    def _prompt(self, observation: Observation) -> str:
        context = self._base_context(observation)
        context["instruction"] = (
            "Use careful internal reasoning about prerequisites, learner memory, cognitive load, retention, and drift. "
            "Return only a concise reasoning_summary/rationale and the final action JSON."
        )
        return _json_prompt("You are a CoT Planner. Keep hidden reasoning private; output concise rationale only.", context)


class OracleAgent(KTRecommenderAgent):
    system_name = "oracle"


class RandomAgent(Agent):
    system_name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: Observation) -> Action:
        if observation.candidate_resources and self.rng.random() < 0.75:
            resource = self.rng.choice(observation.candidate_resources)
            return Action(
                action_type=self.rng.choice(["recommend_exercise", "recommend_explanation", "recommend_review"]),
                resource_id=resource.resource_id,
                target_concepts=resource.concepts,
                rationale="Random baseline.",
            )
        return Action(
            action_type=self.rng.choice(["diagnostic_quiz", "update_plan"]),
            target_concepts=observation.goal.target_concepts,
            rationale="Random baseline.",
        )


class PrerequisiteRuleAgent(KTRecommenderAgent):
    system_name = "prerequisite_rule"

    def act(self, observation: Observation) -> Action:
        prereqs = observation.metadata.get("prerequisites") or []
        weak = [p for p in prereqs if observation.estimated_mastery.get(p, 0.0) < 0.6]
        if weak:
            resources = [r for r in observation.candidate_resources if set(r.concepts).intersection(weak)]
            if resources:
                resource = sorted(resources, key=lambda r: r.difficulty)[0]
                return Action(
                    action_type="recommend_review",
                    resource_id=resource.resource_id,
                    target_concepts=resource.concepts,
                    rationale="Rule baseline reviews unmet prerequisites first.",
                )
        return super().act(observation)


class DifficultyRuleAgent(KTRecommenderAgent):
    system_name = "difficulty_rule"

    def act(self, observation: Observation) -> Action:
        resources = _rank_resources(observation.candidate_resources, observation.estimated_mastery, observation.goal.target_concepts)
        if resources:
            resource = resources[0]
            return Action(
                action_type="recommend_exercise" if resource.type == "exercise" else "recommend_explanation",
                resource_id=resource.resource_id,
                target_concepts=resource.concepts,
                rationale="Rule baseline chooses difficulty closest to readiness.",
            )
        return super().act(observation)


AGENT_REGISTRY: dict[str, type[Agent]] = {
    "kt_recommender": KTRecommenderAgent,
    "static_one_shot": StaticOneShotAgent,
    "static": StaticOneShotAgent,
    "oracle": OracleAgent,
    "random": RandomAgent,
    "prerequisite_rule": PrerequisiteRuleAgent,
    "difficulty_rule": DifficultyRuleAgent,
    "oracle_prerequisite": PrerequisiteRuleAgent,
    "oracle_simulator": OracleAgent,
}

LLM_AGENT_REGISTRY: dict[str, type[LLMPlannerAgent]] = {
    "react": ReActPlannerAgent,
    "react_planner": ReActPlannerAgent,
    "one_shot": OneShotPlannerAgent,
    "one-shot": OneShotPlannerAgent,
    "one_shot_planner": OneShotPlannerAgent,
    "step_by_step": StepByStepPlannerAgent,
    "step-by-step": StepByStepPlannerAgent,
    "step_by_step_planner": StepByStepPlannerAgent,
    "cot": CoTPlannerAgent,
    "cot_planner": CoTPlannerAgent,
}


def create_agent(name: str, *, llm: str | None = None) -> Agent:
    normalized = name.lower()
    from eduplanbench.agents.external import ExternalAgentAdapter, is_external_agent

    if normalized.startswith("external:") or is_external_agent(normalized):
        return ExternalAgentAdapter(normalized)
    if normalized in LLM_AGENT_REGISTRY:
        cls = LLM_AGENT_REGISTRY.get(normalized, ReActPlannerAgent)
        return cls()
    if normalized in AGENT_REGISTRY:
        return AGENT_REGISTRY[normalized]()
    raise ValueError(f"unknown agent: {name}")


def _rank_resources(resources: list[Resource], mastery: dict[str, float], targets: list[str]) -> list[Resource]:
    def score(resource: Resource) -> tuple[float, float, float]:
        overlap = len(set(resource.concepts).intersection(targets))
        avg = sum(mastery.get(concept, 0.4) for concept in resource.concepts) / max(1, len(resource.concepts))
        return (-overlap, abs(resource.difficulty - min(avg + 0.15, 0.8)), resource.difficulty)

    return sorted(resources, key=score)


def _resource_by_id(resources: list[Resource], resource_id: Any) -> Resource | None:
    if not resource_id:
        return None
    return next((resource for resource in resources if resource.resource_id == resource_id), None)


def _json_prompt(role: str, context: dict[str, Any]) -> str:
    schema = {
        "action_type": "one of recommend_exercise, recommend_explanation, recommend_review, diagnostic_quiz, update_plan, recommend_lecture_text, recommend_problem, recommend_easier_problem, recommend_similar_problem, diagnose_misconception, wait_or_reduce_load, summarize_knowledge",
        "resource_id": "required for resource recommendation actions, otherwise null",
        "target_concepts": ["concept id/name strings"],
        "rationale": "short evidence-grounded rationale that cites learner state or feedback",
        "plan_update": "optional plan update",
        "diagnosis": "optional misconception diagnosis",
        "hint": "optional tutoring hint; do not directly reveal final answer unless necessary",
    }
    return (
        f"{role}\n"
        "Return ONLY valid JSON. No markdown.\n"
        f"Required schema: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Context: {json.dumps(context, ensure_ascii=False)}"
    )
