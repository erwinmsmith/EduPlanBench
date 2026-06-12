from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from eduplanbench.core.schema import Action, Resource, TaskInstance


@dataclass(slots=True)
class BKTParams:
    prior: float = 0.3
    learn: float = 0.12
    slip: float = 0.1
    guess: float = 0.2
    forget: float = 0.01
    difficulty_sensitivity: float = 0.25


@dataclass(slots=True)
class RuleBKTStudentSimulator:
    task: TaskInstance
    seed: int = 0
    params: BKTParams = field(default_factory=BKTParams)
    true_mastery: dict[str, float] = field(default_factory=dict)
    visible_mastery: dict[str, float] = field(default_factory=dict)
    dropout_risk: float = 0.0
    confusion: float = 0.0
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        initial = self.task.learner_profile.estimated_mastery or {}
        concepts = set(initial) | set(self.task.goal.target_concepts)
        for resource in self.task.resource_pool:
            concepts.update(resource.concepts)
        for concept in concepts:
            base = float(initial.get(concept, self.params.prior))
            self.visible_mastery[concept] = _clamp(base)
            self.true_mastery[concept] = _clamp(base + self.rng.uniform(-0.05, 0.05))

    def apply(self, action: Action, resource: Resource | None, step: int) -> dict[str, Any]:
        if action.action_type in {"recommend_exercise", "recommend_problem"}:
            return self._exercise(resource)
        if action.action_type in {"recommend_explanation", "recommend_lecture_text", "summarize_knowledge"}:
            return self._instruction(action, resource, gain=0.035)
        if action.action_type == "recommend_review":
            return self._instruction(action, resource, gain=0.055, retention=True)
        if action.action_type in {"diagnostic_quiz", "recommend_diagnostic"}:
            return self._diagnostic(action)
        if action.action_type in {"update_plan", "update_path", "diagnose_misconception", "wait_or_reduce_load"}:
            self.confusion = _clamp(self.confusion - 0.03)
            self.dropout_risk = _clamp(self.dropout_risk - 0.02)
            return {
                "student_feedback": {"completion": True, "feedback_text": "The learner used the planning step to reorganize effort."},
                "mastery_gain": 0.0,
                "elapsed_time": 20,
                "correct": None,
            }
        return {"student_feedback": {"completion": False, "feedback_text": "No learning effect."}, "mastery_gain": 0.0}

    def _exercise(self, resource: Resource | None) -> dict[str, Any]:
        concepts = resource.concepts if resource else self.task.goal.target_concepts
        difficulty = resource.difficulty if resource else 0.5
        avg_mastery = sum(self.true_mastery.get(c, self.params.prior) for c in concepts) / max(1, len(concepts))
        empirical = None
        if resource:
            response = str(resource.metadata.get("response", ""))
            is_correct = resource.metadata.get("is_correct")
            if response in {"0", "1"}:
                empirical = response == "1"
            elif isinstance(is_correct, bool):
                empirical = is_correct
        p_correct = avg_mastery * (1 - self.params.slip) + (1 - avg_mastery) * self.params.guess
        p_correct -= max(0.0, difficulty - avg_mastery) * self.params.difficulty_sensitivity
        correct = empirical if empirical is not None else self.rng.random() < _clamp(p_correct)
        total_gain = 0.0
        for concept in concepts:
            before = self.true_mastery.get(concept, self.params.prior)
            posterior = self._posterior(before, correct)
            after = posterior + (1 - posterior) * self.params.learn
            self.true_mastery[concept] = _clamp(after)
            self.visible_mastery[concept] = _clamp(self.visible_mastery.get(concept, before) * 0.65 + after * 0.35)
            total_gain += self.true_mastery[concept] - before
        overload = max(0.0, difficulty - avg_mastery - 0.2)
        self.confusion = _clamp(self.confusion + overload * 0.2 - (0.03 if correct else 0.0))
        self.dropout_risk = _clamp(self.dropout_risk + overload * 0.12)
        elapsed = int(45 + difficulty * 80 + self.confusion * 80 + self.rng.randint(0, 20))
        return {
            "student_feedback": {
                "correct": correct,
                "elapsed_time": elapsed,
                "completion": True,
                "confusion_signal": _level(self.confusion),
                "feedback_text": "correct" if correct else "incorrect; learner may need prerequisite review",
                "empirical_feedback": empirical is not None,
            },
            "mastery_gain": total_gain,
            "elapsed_time": elapsed,
            "correct": correct,
        }

    def _instruction(self, action: Action, resource: Resource | None, *, gain: float, retention: bool = False) -> dict[str, Any]:
        concepts = action.target_concepts or (resource.concepts if resource else self.task.goal.target_concepts)
        total_gain = 0.0
        for concept in concepts:
            before = self.true_mastery.get(concept, self.params.prior)
            multiplier = 1.2 if retention else 1.0
            after = before + (1 - before) * gain * multiplier
            self.true_mastery[concept] = _clamp(after)
            self.visible_mastery[concept] = _clamp(self.visible_mastery.get(concept, before) * 0.8 + after * 0.2)
            total_gain += self.true_mastery[concept] - before
        self.confusion = _clamp(self.confusion - 0.04)
        return {
            "student_feedback": {
                "completion": True,
                "elapsed_time": 60,
                "confusion_signal": _level(self.confusion),
                "feedback_text": "learner completed review" if retention else "learner read explanation",
            },
            "mastery_gain": total_gain,
            "elapsed_time": 60,
            "correct": None,
        }

    def _diagnostic(self, action: Action) -> dict[str, Any]:
        concepts = action.target_concepts or self.task.goal.target_concepts
        for concept in concepts:
            true = self.true_mastery.get(concept, self.params.prior)
            self.visible_mastery[concept] = _clamp(true + self.rng.uniform(-0.03, 0.03))
        return {
            "student_feedback": {
                "completion": True,
                "elapsed_time": 45,
                "feedback_text": "diagnostic estimate updated",
                "confusion_signal": _level(self.confusion),
            },
            "mastery_gain": 0.0,
            "elapsed_time": 45,
            "correct": None,
        }

    def _posterior(self, prior: float, correct: bool) -> float:
        if correct:
            numerator = prior * (1 - self.params.slip)
            denominator = numerator + (1 - prior) * self.params.guess
        else:
            numerator = prior * self.params.slip
            denominator = numerator + (1 - prior) * (1 - self.params.guess)
        return numerator / denominator if denominator else prior


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _level(value: float) -> str:
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "medium"
    return "low"
