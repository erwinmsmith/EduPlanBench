from __future__ import annotations

from typing import Any

from eduplanbench.core.schema import Action, Observation, StepResult, TaskInstance
from eduplanbench.simulators import RuleBKTStudentSimulator


class EduPlanEnv:
    def __init__(self, task: TaskInstance, *, seed: int = 0) -> None:
        self.task = task
        self.seed = seed
        self.simulator = RuleBKTStudentSimulator(task, seed=seed)
        self.step_count = 0
        self.recent_feedback: list[str] = []
        self.current_plan = ""
        self.active_horizon = task.horizon
        self.unavailable_resources: set[str] = set()
        self.applied_perturbations: set[int] = set()
        self.dynamic_events: list[dict[str, Any]] = []

    def reset(self) -> Observation:
        self.step_count = 0
        self.recent_feedback = []
        self.current_plan = ""
        self.active_horizon = self.task.horizon
        self.unavailable_resources = set()
        self.applied_perturbations = set()
        self.dynamic_events = []
        self.simulator = RuleBKTStudentSimulator(self.task, seed=self.seed)
        return self._observation()

    def step(self, action: Action) -> StepResult:
        candidates = self._candidate_resources()
        valid, error = action.validate_for(candidates)
        resource = next((item for item in candidates if item.resource_id == action.resource_id), None)
        if not valid:
            info = {"valid_action": False, "validation_error": error, "hidden": self._hidden_info()}
            self.recent_feedback.append(f"Invalid action: {error}")
            self.recent_feedback = self.recent_feedback[-5:]
            self.step_count += 1
            return StepResult(self._observation(), reward=-0.1, done=self._done(), info=info)

        sim_info = self.simulator.apply(action, resource, self.step_count)
        if action.plan_update:
            self.current_plan = action.plan_update
        feedback = sim_info.get("student_feedback", {}).get("feedback_text", "")
        if feedback:
            self.recent_feedback.append(feedback)
            self.recent_feedback = self.recent_feedback[-5:]
        self.step_count += 1
        reward = self._reward(sim_info)
        done = self._done()
        info = {
            "valid_action": True,
            "mastery_gain": sim_info.get("mastery_gain", 0.0),
            "student_feedback": sim_info.get("student_feedback", {}),
            "dropout_risk": self.simulator.dropout_risk,
            "confusion": self.simulator.confusion,
            "hidden": self._hidden_info(),
        }
        return StepResult(self._observation(), reward=reward, done=done, info=info)

    def _observation(self) -> Observation:
        self._apply_due_perturbations()
        candidates = self._candidate_resources()
        return Observation(
            task_id=self.task.task_id,
            step=self.step_count,
            goal=self.task.goal,
            learner_summary=self.task.learner_profile.profile_text,
            estimated_mastery=dict(self.simulator.visible_mastery),
            recent_feedback=list(self.recent_feedback),
            available_actions=[
                "recommend_exercise",
                "recommend_explanation",
                "recommend_review",
                "diagnostic_quiz",
                "diagnose_misconception",
                "update_plan",
                "wait_or_reduce_load",
            ],
            candidate_resources=candidates,
            current_plan=self.current_plan,
            metadata={
                "track": self.task.track,
                **self.task.constraints,
                "active_horizon": self.active_horizon,
                "dynamic_events": list(self.dynamic_events[-5:]),
                "unavailable_resources": sorted(self.unavailable_resources),
            },
        )

    def _done(self) -> bool:
        if self.step_count >= self.active_horizon:
            return True
        return all(
            self.simulator.true_mastery.get(concept, 0.0) >= self.task.goal.target_mastery
            for concept in self.task.goal.target_concepts
        )

    def _reward(self, sim_info: dict) -> float:
        target_gain = sum(
            self.simulator.true_mastery.get(concept, 0.0) for concept in self.task.goal.target_concepts
        ) / max(1, len(self.task.goal.target_concepts))
        return (
            float(sim_info.get("mastery_gain", 0.0))
            + 0.05 * target_gain
            - 0.03 * self.simulator.confusion
            - 0.05 * self.simulator.dropout_risk
        )

    def _hidden_info(self) -> dict:
        return {
            "true_mastery": dict(self.simulator.true_mastery),
            "visible_mastery": dict(self.simulator.visible_mastery),
            "dropout_risk": self.simulator.dropout_risk,
            "confusion": self.simulator.confusion,
            "dynamic_events": list(self.dynamic_events),
        }

    def _candidate_resources(self):
        visible = [resource for resource in self.task.resource_pool if resource.resource_id not in self.unavailable_resources]
        limit = 30 if self.task.metadata.get("task_type") == "Long-context Memory" else 20
        return visible[:limit]

    def _apply_due_perturbations(self) -> None:
        for idx, event in enumerate(self.task.metadata.get("perturbations", []) or []):
            if idx in self.applied_perturbations:
                continue
            if int(event.get("step", -1)) > self.step_count:
                continue
            self.applied_perturbations.add(idx)
            kind = str(event.get("type", ""))
            if kind == "forced_error_streak":
                self._forced_error_streak(event)
            elif kind == "forget_prerequisite":
                self._forget_prerequisite(event)
            elif kind == "resource_unavailable":
                self._resource_unavailable(event)
            elif kind == "time_budget_reduction":
                self._time_budget_reduction(event)

    def _forced_error_streak(self, event: dict[str, Any]) -> None:
        severity = float(event.get("severity", 0.1))
        self.simulator.confusion = _clamp(self.simulator.confusion + severity)
        self.simulator.dropout_risk = _clamp(self.simulator.dropout_risk + severity * 0.5)
        message = str(event.get("message") or "learner shows an unexpected error streak")
        self.recent_feedback.append(message)
        self.recent_feedback = self.recent_feedback[-5:]
        self.dynamic_events.append({"step": self.step_count, "type": "forced_error_streak", "severity": severity})

    def _forget_prerequisite(self, event: dict[str, Any]) -> None:
        concept = event.get("target_concept") or (self.task.goal.target_concepts[0] if self.task.goal.target_concepts else None)
        if not concept:
            return
        delta = float(event.get("delta", 0.1))
        before = self.simulator.true_mastery.get(str(concept), 0.3)
        self.simulator.true_mastery[str(concept)] = _clamp(before - delta)
        self.simulator.visible_mastery[str(concept)] = _clamp(self.simulator.visible_mastery.get(str(concept), before) - delta * 0.45)
        self.recent_feedback.append(f"learner appears to have forgotten prerequisite {concept}")
        self.recent_feedback = self.recent_feedback[-5:]
        self.dynamic_events.append({"step": self.step_count, "type": "forget_prerequisite", "concept": str(concept), "delta": delta})

    def _resource_unavailable(self, event: dict[str, Any]) -> None:
        rid = event.get("resource_id")
        if not rid and self.task.resource_pool:
            rid = self.task.resource_pool[-1].resource_id
        if not rid:
            return
        self.unavailable_resources.add(str(rid))
        self.recent_feedback.append(f"resource became unavailable: {rid}")
        self.recent_feedback = self.recent_feedback[-5:]
        self.dynamic_events.append({"step": self.step_count, "type": "resource_unavailable", "resource_id": str(rid)})

    def _time_budget_reduction(self, event: dict[str, Any]) -> None:
        remaining = int(event.get("remaining_horizon", max(1, self.active_horizon - self.step_count)))
        self.active_horizon = min(self.active_horizon, self.step_count + max(1, remaining))
        self.recent_feedback.append(f"time budget reduced; active horizon is now {self.active_horizon}")
        self.recent_feedback = self.recent_feedback[-5:]
        self.dynamic_events.append({"step": self.step_count, "type": "time_budget_reduction", "active_horizon": self.active_horizon})


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
