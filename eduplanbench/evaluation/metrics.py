from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from eduplanbench.core.schema import EpisodeTrace, MetricReport, TRACK1, TRACK2, TRACK3


def evaluate_traces(traces: list[EpisodeTrace], *, run_id: str = "run") -> MetricReport:
    rows = [_metrics_for_trace(trace) for trace in traces]
    metrics = _aggregate(rows)
    by_track: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for trace, row in zip(traces, rows, strict=False):
        grouped[trace.task.track].append(row)
    for track, values in grouped.items():
        by_track[track] = _aggregate(values)
    return MetricReport(run_id=run_id, metrics=metrics, by_track=by_track, metadata={"episodes": len(traces)})


def _metrics_for_trace(trace: EpisodeTrace) -> dict[str, float]:
    task = trace.task
    steps = len(trace.steps)
    valid = [1.0 if step.get("valid_action") else 0.0 for step in trace.steps]
    final_hidden = trace.final_info.get("hidden", {})
    true_mastery = final_hidden.get("true_mastery", {})
    initial = task.learner_profile.estimated_mastery
    target_scores = [float(true_mastery.get(c, initial.get(c, 0.0))) for c in task.goal.target_concepts]
    initial_scores = [float(initial.get(c, 0.0)) for c in task.goal.target_concepts]
    final_avg = sum(target_scores) / max(1, len(target_scores))
    initial_avg = sum(initial_scores) / max(1, len(initial_scores))
    progress = max(0.0, final_avg - initial_avg) / max(1e-6, task.goal.target_mastery - initial_avg)
    row = {
        "gsr": 1.0 if final_avg >= task.goal.target_mastery else 0.0,
        "pr": min(1.0, progress),
        "steps": float(steps),
        "normalized_steps": steps / max(1, task.horizon),
        "valid_action_rate": sum(valid) / max(1, len(valid)),
        "context_cost": _context_cost(trace),
        "time_cost": sum(float(step.get("info", {}).get("student_feedback", {}).get("elapsed_time", 0.0)) for step in trace.steps),
        "adaptation_quality": _adaptation_quality(trace),
        "cognitive_load_alignment": _cognitive_load_alignment(trace),
        "personalization_score": _personalization_score(trace),
        "redundancy_rate": _repetition_rate(trace),
        "plan_drift": 1.0 - _path_coherence(trace),
    }
    track_score = 0.0
    if task.track == TRACK1:
        row.update(_track1(trace))
        track_score = (
            0.40 * row["misconception_diagnosis_accuracy"]
            + 0.30 * row["remediation_success"]
            + 0.20 * row["feedback_grounding"]
            + 0.10 * row["hint_helpfulness"]
        )
    elif task.track == TRACK2:
        row.update(_track2(trace))
        track_score = (
            0.30 * row["knowledge_sequence_consistency"]
            + 0.25 * row["resource_concept_match"]
            + 0.20 * row["constraint_satisfaction"]
            - 0.15 * row["prerequisite_violation_rate"]
            + 0.10 * row["path_coherence"]
        )
    elif task.track == TRACK3:
        row.update(_track3(trace, final_avg, initial_avg))
        track_score = (
            0.35 * row["mastery_gain"]
            + 0.25 * row["retention_gain"]
            + 0.20 * row["learning_efficiency"]
            - 0.10 * row["dropout_risk"]
            - 0.10 * row["simulator_exploitation_rate"]
        )
    row["core_score"] = (
        0.35 * row["gsr"]
        + 0.25 * row["pr"]
        + 0.15 * row["adaptation_quality"]
        + 0.10 * row["valid_action_rate"]
        - 0.10 * row["normalized_steps"]
        - 0.05 * min(1.0, row["context_cost"] / 10_000)
    )
    row["track_score"] = track_score
    row["overall_score"] = 0.5 * row["core_score"] + 0.5 * row["track_score"]
    return row


def _track1(trace: EpisodeTrace) -> dict[str, float]:
    actions = [step["action"]["action_type"] for step in trace.steps]
    reference = str(trace.task.metadata.get("case", {}).get("misconception", "")).lower()
    diagnoses = [
        str(step["action"].get("payload", {}).get("diagnosis") or step["action"].get("rationale", "")).lower()
        for step in trace.steps
    ]
    diagnosis_hit = _best_overlap(reference, diagnoses)
    grounded = _feedback_grounding(trace)
    hint = sum(1 for step in trace.steps if step["action"].get("payload", {}).get("hint") or "hint" in step["action"].get("rationale", "").lower()) / max(1, len(trace.steps))
    return {
        "misconception_diagnosis_accuracy": diagnosis_hit if reference else (1.0 if "diagnose_misconception" in actions or "recommend_explanation" in actions else 0.0),
        "remediation_success": 1.0 if any(a in actions for a in ("recommend_review", "recommend_explanation", "recommend_easier_problem", "recommend_similar_problem")) else 0.0,
        "remediation_match": _remediation_match(trace),
        "feedback_grounding": grounded,
        "hint_helpfulness": max(hint, grounded * 0.5),
        "error_aware_replanning_quality": _adaptation_quality(trace),
        "direct_answer_rate": _direct_answer_rate(trace),
    }


def _track2(trace: EpisodeTrace) -> dict[str, float]:
    prereqs = set(trace.task.constraints.get("prerequisites", []))
    seen = set()
    violations = 0
    resource_matches = 0
    for step in trace.steps:
        action = step["action"]
        targets = set(action.get("target_concepts") or [])
        if trace.task.goal.target_concepts and targets.intersection(trace.task.goal.target_concepts) and prereqs and not prereqs.intersection(seen):
            violations += 1
        seen.update(targets)
        if targets:
            resource_matches += 1
    return {
        "prerequisite_violation_rate": violations / max(1, len(trace.steps)),
        "knowledge_sequence_consistency": _sequence_consistency(trace, violations),
        "resource_concept_match": resource_matches / max(1, len(trace.steps)),
        "constraint_satisfaction": _constraint_satisfaction(trace, violations),
        "path_coherence": _path_coherence(trace),
        "difficulty_alignment": _cognitive_load_alignment(trace),
    }


def _track3(trace: EpisodeTrace, final_avg: float, initial_avg: float) -> dict[str, float]:
    total_gain = sum(float(step.get("info", {}).get("mastery_gain", 0.0)) for step in trace.steps)
    dropout = trace.final_info.get("hidden", {}).get("dropout_risk", 0.0)
    return {
        "mastery_gain": max(0.0, final_avg - initial_avg),
        "retention_gain": max(0.0, total_gain) * 0.5,
        "learning_efficiency": max(0.0, total_gain) / max(1, len(trace.steps)),
        "dropout_risk": float(dropout),
        "overload_rate": _overload_rate(trace),
        "recovery_rate": _recovery_rate(trace),
        "simulator_exploitation_rate": _repetition_rate(trace),
    }


def action_distribution(trace: EpisodeTrace) -> dict[str, float]:
    steps = max(1, len(trace.steps))
    counts = defaultdict(int)
    difficulties = []
    target_hits = 0
    fallback = 0
    resource_ids = []
    for step in trace.steps:
        action = step["action"]
        action_type = action.get("action_type", "")
        counts[action_type] += 1
        if action.get("payload", {}).get("fallback_normalized") or not step.get("valid_action", True):
            fallback += 1
        rid = action.get("resource_id")
        if rid:
            resource_ids.append(rid)
        resources = step.get("observation", {}).get("candidate_resources", [])
        resource = next((item for item in resources if item.get("resource_id") == rid), None)
        if resource:
            difficulties.append(float(resource.get("difficulty", 0.0)))
        targets = set(step.get("observation", {}).get("goal", {}).get("target_concepts", []))
        concepts = set(action.get("target_concepts") or [])
        if targets and concepts.intersection(targets):
            target_hits += 1
    return {
        "exercise_pct": counts["recommend_exercise"] / steps,
        "review_pct": counts["recommend_review"] / steps,
        "explanation_pct": (counts["recommend_explanation"] + counts["recommend_lecture_text"]) / steps,
        "diagnostic_pct": (counts["diagnostic_quiz"] + counts["recommend_diagnostic"]) / steps,
        "avg_difficulty": sum(difficulties) / max(1, len(difficulties)),
        "target_concept_hit": target_hits / steps,
        "fallback_rate": fallback / steps,
        "unique_resources": len(set(resource_ids)) / max(1, len(resource_ids)) if resource_ids else 0.0,
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}


def _context_cost(trace: EpisodeTrace) -> float:
    total = 0
    for step in trace.steps:
        total += _estimate_tokens(step.get("observation", {}))
        total += _estimate_tokens(step.get("action", {}))
    return float(total)


def _estimate_tokens(value) -> int:
    """Cheap tokenizer-free proxy: ASCII chars / 4 plus non-ASCII chars.

    This is not provider billing usage. It is a stable context-size estimate
    for comparing agents on the same benchmark traces.
    """
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return int(ascii_chars / 4 + non_ascii_chars)


def _adaptation_quality(trace: EpisodeTrace) -> float:
    if not trace.steps:
        return 0.0
    updates = sum(1 for step in trace.steps if step["action"].get("plan_update") or step["action"]["action_type"] == "update_plan")
    feedback_steps = sum(1 for step in trace.steps if step.get("observation", {}).get("recent_feedback"))
    return min(1.0, (updates + 0.5 * feedback_steps) / max(1, len(trace.steps)))


def _path_coherence(trace: EpisodeTrace) -> float:
    targets = trace.task.goal.target_concepts
    aligned = 0
    for step in trace.steps:
        concepts = step["action"].get("target_concepts") or []
        if set(concepts).intersection(targets):
            aligned += 1
    return aligned / max(1, len(trace.steps))


def _sequence_consistency(trace: EpisodeTrace, violations: int) -> float:
    changes = 0
    prev: set[str] | None = None
    for step in trace.steps:
        concepts = set(step["action"].get("target_concepts") or [])
        if prev is not None and concepts and prev and not concepts.intersection(prev | set(trace.task.goal.target_concepts)):
            changes += 1
        if concepts:
            prev = concepts
    penalty = (violations + changes) / max(1, len(trace.steps))
    return max(0.0, 1.0 - penalty)


def _constraint_satisfaction(trace: EpisodeTrace, violations: int) -> float:
    difficulty_bad = _overload_rate(trace)
    invalid = 1.0 - (sum(1.0 if step.get("valid_action") else 0.0 for step in trace.steps) / max(1, len(trace.steps)))
    over_budget = 1.0 if len(trace.steps) > trace.task.horizon else 0.0
    penalty = 0.45 * violations / max(1, len(trace.steps)) + 0.35 * difficulty_bad + 0.15 * invalid + 0.05 * over_budget
    return max(0.0, 1.0 - penalty)


def _repetition_rate(trace: EpisodeTrace) -> float:
    ids = [step["action"].get("resource_id") for step in trace.steps if step["action"].get("resource_id")]
    return 1.0 - len(set(ids)) / max(1, len(ids)) if ids else 0.0


def _overload_rate(trace: EpisodeTrace) -> float:
    overload = 0
    total = 0
    for step in trace.steps:
        action = step["action"]
        rid = action.get("resource_id")
        resources = step.get("observation", {}).get("candidate_resources", [])
        resource = next((item for item in resources if item.get("resource_id") == rid), None)
        if not resource:
            continue
        concepts = resource.get("concepts") or []
        mastery = step.get("observation", {}).get("estimated_mastery") or {}
        avg = sum(float(mastery.get(c, 0.4)) for c in concepts) / max(1, len(concepts))
        total += 1
        if float(resource.get("difficulty", 0.5)) > avg + 0.30:
            overload += 1
    return overload / max(1, total)


def _recovery_rate(trace: EpisodeTrace) -> float:
    recoveries = 0
    opportunities = 0
    prev_wrong = False
    for step in trace.steps:
        feedback = step.get("info", {}).get("student_feedback", {})
        action_type = step.get("action", {}).get("action_type", "")
        if prev_wrong:
            opportunities += 1
            if action_type in {"recommend_review", "recommend_explanation", "diagnostic_quiz", "recommend_easier_problem", "update_plan"}:
                recoveries += 1
        prev_wrong = feedback.get("correct") is False
    return recoveries / max(1, opportunities)


def _direct_answer_rate(trace: EpisodeTrace) -> float:
    direct = 0
    total = 0
    answers = []
    case = trace.task.metadata.get("case", {})
    for key in ("correct_answer", "answer"):
        value = case.get(key)
        if value:
            answers.append(str(value).strip().lower())
    for step in trace.steps:
        action = step.get("action", {})
        text = " ".join(
            [
                str(action.get("rationale", "")),
                str(action.get("payload", {}).get("hint", "")),
                str(action.get("payload", {}).get("explanation", "")),
            ]
        ).lower()
        if not text:
            continue
        total += 1
        if any(answer and answer in text for answer in answers) or "final answer" in text or "the answer is" in text:
            direct += 1
    return direct / max(1, total)


def _remediation_match(trace: EpisodeTrace) -> float:
    case = trace.task.metadata.get("case", {})
    refs = [
        str(case.get("misconception", "")),
        str(trace.task.metadata.get("misstep_reference", {}).get("remediation_reference", "")),
        str(trace.task.metadata.get("mathdial_reference", {}).get("remediation_reference", "")),
    ]
    ref = " ".join(refs).lower()
    if not ref.strip():
        return 0.0
    actions = [
        " ".join(
            [
                str(step["action"].get("rationale", "")),
                str(step["action"].get("plan_update", "")),
                str(step["action"].get("payload", {})),
            ]
        ).lower()
        for step in trace.steps
    ]
    return _best_overlap(ref, actions)


def _cognitive_load_alignment(trace: EpisodeTrace) -> float:
    scores = []
    for step in trace.steps:
        action = step["action"]
        obs = step["observation"]
        resource_id = action.get("resource_id")
        resources = obs.get("candidate_resources", [])
        resource = next((item for item in resources if item.get("resource_id") == resource_id), None)
        if not resource:
            continue
        concepts = resource.get("concepts") or []
        mastery = obs.get("estimated_mastery") or {}
        avg = sum(float(mastery.get(c, 0.4)) for c in concepts) / max(1, len(concepts))
        diff = abs(float(resource.get("difficulty", 0.5)) - min(avg + 0.15, 0.8))
        scores.append(max(0.0, 1.0 - diff / 0.8))
    return sum(scores) / max(1, len(scores)) if scores else 0.0


def _personalization_score(trace: EpisodeTrace) -> float:
    weak = set(trace.task.learner_profile.weak_concepts)
    target = set(trace.task.goal.target_concepts)
    hits = 0
    for step in trace.steps:
        text = " ".join(
            [
                step["action"].get("rationale", ""),
                step["action"].get("plan_update", ""),
                str(step["action"].get("target_concepts", "")),
            ]
        )
        if any(item and item in text for item in weak | target):
            hits += 1
    return hits / max(1, len(trace.steps))


def _feedback_grounding(trace: EpisodeTrace) -> float:
    hits = 0
    for step in trace.steps:
        feedback = step.get("observation", {}).get("recent_feedback", [])
        rationale = step.get("action", {}).get("rationale", "")
        if feedback and any(str(item).split(";")[0].lower()[:20] in rationale.lower() for item in feedback):
            hits += 1
        elif feedback and rationale:
            hits += 0.5
    return hits / max(1, len(trace.steps))


def _best_overlap(reference: str, candidates: list[str]) -> float:
    ref_tokens = {tok for tok in reference.replace("_", " ").split() if len(tok) > 2}
    if not ref_tokens:
        return 0.0
    best = 0.0
    for cand in candidates:
        tokens = {tok for tok in cand.replace("_", " ").split() if len(tok) > 2}
        best = max(best, len(ref_tokens & tokens) / len(ref_tokens))
    return best
