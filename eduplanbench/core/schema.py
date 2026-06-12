from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


TRACK1 = "track1_text_math"
TRACK2 = "track2_mooc_planning"
TRACK3 = "track3_kt_simulator"
ALL_TRACKS = (TRACK1, TRACK2, TRACK3)


@dataclass(slots=True)
class LearnerProfile:
    profile_text: str
    estimated_mastery: dict[str, float] = field(default_factory=dict)
    weak_concepts: list[str] = field(default_factory=list)
    recent_errors: list[dict[str, Any]] = field(default_factory=list)
    learning_pattern: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GoalSpec:
    target_concepts: list[str]
    target_mastery: float = 0.75
    horizon: int = 30
    description: str = ""


@dataclass(slots=True)
class Resource:
    resource_id: str
    type: str
    text: str = ""
    concepts: list[str] = field(default_factory=list)
    difficulty: float = 0.5
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskInstance:
    task_id: str
    track: str
    domain: str
    horizon: int
    learner_profile: LearnerProfile
    goal: GoalSpec
    resource_pool: list[Resource] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Action:
    action_type: str
    resource_id: str | None = None
    target_concepts: list[str] = field(default_factory=list)
    rationale: str = ""
    plan_update: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    VALID_ACTIONS: ClassVar[set[str]] = {
        "recommend_exercise",
        "recommend_explanation",
        "recommend_review",
        "diagnostic_quiz",
        "recommend_diagnostic",
        "update_plan",
        "update_path",
        "recommend_lecture_text",
        "recommend_problem",
        "recommend_easier_problem",
        "recommend_similar_problem",
        "diagnose_misconception",
        "wait_or_reduce_load",
        "summarize_knowledge",
    }

    def validate_for(self, resources: list[Resource]) -> tuple[bool, str]:
        if self.action_type not in self.VALID_ACTIONS:
            return False, f"unknown action_type: {self.action_type}"
        resource_actions = {
            "recommend_exercise",
            "recommend_explanation",
            "recommend_review",
            "recommend_lecture_text",
            "recommend_problem",
            "recommend_easier_problem",
            "recommend_similar_problem",
        }
        if self.action_type in resource_actions:
            ids = {r.resource_id for r in resources}
            if not self.resource_id:
                return False, "resource_id is required"
            if self.resource_id not in ids:
                return False, f"resource_id not found: {self.resource_id}"
        return True, ""


@dataclass(slots=True)
class Observation:
    task_id: str
    step: int
    goal: GoalSpec
    learner_summary: str
    estimated_mastery: dict[str, float]
    recent_feedback: list[str]
    available_actions: list[str]
    candidate_resources: list[Resource]
    current_plan: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrace:
    task: TaskInstance
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_info: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        observation: Observation,
        action: Action,
        result: StepResult,
        valid_action: bool,
        validation_error: str = "",
    ) -> None:
        self.steps.append(
            {
                "observation": to_plain(observation),
                "action": to_plain(action),
                "reward": result.reward,
                "done": result.done,
                "info": result.info,
                "valid_action": valid_action,
                "validation_error": validation_error,
            }
        )


@dataclass(slots=True)
class MetricReport:
    run_id: str
    metrics: dict[str, float]
    by_track: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def to_plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    return value
