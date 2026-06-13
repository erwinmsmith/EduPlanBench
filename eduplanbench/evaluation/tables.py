from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from eduplanbench.core.io import ensure_dir, read_json, read_jsonl, write_json
from eduplanbench.core.schema import EpisodeTrace
from eduplanbench.data.task_builders import task_from_dict
from eduplanbench.evaluation.metrics import _metrics_for_trace, action_distribution


AGENT_LABELS = {
    "one_shot": "One-shot Planner",
    "react": "ReAct Planner",
    "cot": "CoT Planner",
    "step_by_step": "Step-by-step Planner",
    "external:llm_pddl": "LLM+P",
    "llm_pddl": "LLM+P",
    "external:lats": "LATS",
    "lats": "LATS",
    "external:plan_and_act": "Plan-and-Act",
    "plan_and_act": "Plan-and-Act",
    "external:reactree": "ReAcTree",
    "reactree": "ReAcTree",
    "external:hiagent": "HiAgent",
    "hiagent": "HiAgent",
}

PREFERRED_AGENT_ORDER = [
    "one_shot",
    "cot",
    "react",
    "step_by_step",
    "external:llm_pddl",
    "external:lats",
    "external:plan_and_act",
    "external:reactree",
    "external:hiagent",
]

TRACK_LABELS = {
    "track1_text_math": "Track 1",
    "track2_mooc_planning": "Track 2",
    "track3_kt_simulator": "Track 3",
}


def build_tables_from_experiment(experiment_dir: Path, output_dir: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    output_dir = ensure_dir(output_dir or experiment_dir / "tables")
    runs = _load_matrix_runs(experiment_dir)
    per_combo: dict[tuple[str, str], list[tuple[EpisodeTrace, dict[str, float]]]] = {}
    for row in runs:
        traces = _load_episode_traces(Path(row["run_dir"]))
        combo_rows = [(trace, _metrics_for_trace(trace)) for trace in traces]
        per_combo[(row["track"], row["agent"])] = combo_rows

    tables = {
        "Track1_Main": _main_table(per_combo, "track1_text_math"),
        "Track1_Specific": _track1_specific(per_combo),
        "Track1_Difficulty": _track1_difficulty(per_combo),
        "Track2_Main": _main_table(per_combo, "track2_mooc_planning"),
        "Track2_PathQuality": _track2_specific(per_combo),
        "Track2_TaskTypes": _track2_task_types(per_combo),
        "Track3_Main": _main_table(per_combo, "track3_kt_simulator"),
        "Track3_ClosedLoop": _track3_specific(per_combo),
        "Track3_ActionDiag": _track3_action_diag(per_combo),
        "Robustness": _load_robustness_table(experiment_dir),
    }
    write_json(output_dir / "tables.json", tables)
    for name, rows in tables.items():
        _write_table_csv(output_dir / f"{name}.csv", rows)
    return tables


def add_robustness_table(tables: dict[str, list[dict[str, Any]]], robustness_rows: list[dict[str, Any]], output_dir: Path) -> None:
    tables["Robustness"] = robustness_rows
    write_json(output_dir / "tables.json", tables)
    _write_table_csv(output_dir / "Robustness.csv", robustness_rows)


def _load_matrix_runs(experiment_dir: Path) -> list[dict[str, str]]:
    path = experiment_dir / "matrix_results.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_episode_traces(run_dir: Path) -> list[EpisodeTrace]:
    path = run_dir / "episodes.jsonl.gz"
    if not path.exists():
        path = run_dir / "episodes.jsonl"
    traces: list[EpisodeTrace] = []
    for row in read_jsonl(path):
        trace = EpisodeTrace(task=task_from_dict(row["task"]))
        trace.steps = row.get("steps", [])
        trace.final_info = row.get("final_info", trace.steps[-1]["info"] if trace.steps else {})
        traces.append(trace)
    return traces


def _load_robustness_table(experiment_dir: Path) -> list[dict[str, Any]]:
    robustness_path = experiment_dir / "robustness" / "robustness_table.json"
    if robustness_path.exists():
        return read_json(robustness_path)
    existing = experiment_dir / "tables" / "tables.json"
    if existing.exists():
        payload = read_json(existing)
        rows = payload.get("Robustness", [])
        if rows:
            return rows
    return []


def _main_table(per_combo: dict, track: str) -> list[dict[str, Any]]:
    rows = []
    for agent in _agents_for_track(per_combo, track):
        metrics = _agg([row for _, row in per_combo.get((track, agent), [])])
        rows.append(
            {
                "Agent": _agent_label(agent),
                "Overall ↑": metrics.get("overall_score"),
                "Core ↑": metrics.get("core_score"),
                "Track ↑": metrics.get("track_score"),
                "GSR ↑": metrics.get("gsr"),
                "PR ↑": metrics.get("pr"),
                "Steps ↓": metrics.get("steps"),
                "Valid ↑": metrics.get("valid_action_rate"),
                "Context ↓": metrics.get("context_cost"),
                "Time ↓": metrics.get("time_cost"),
            }
        )
    return rows


def _track1_specific(per_combo: dict) -> list[dict[str, Any]]:
    rows = []
    for agent in _agents_for_track(per_combo, "track1_text_math"):
        metrics = _agg([row for _, row in per_combo.get(("track1_text_math", agent), [])])
        rows.append(
            {
                "Agent": _agent_label(agent),
                "Misconception Acc ↑": metrics.get("misconception_diagnosis_accuracy"),
                "Feedback Grounding ↑": metrics.get("feedback_grounding"),
                "Remediation Match ↑": metrics.get("remediation_match"),
                "Hint Helpfulness ↑": metrics.get("hint_helpfulness"),
                "Error-aware Replan ↑": metrics.get("error_aware_replanning_quality"),
                "Redundancy ↓": metrics.get("redundancy_rate"),
                "Direct-answer Rate ↓": metrics.get("direct_answer_rate"),
            }
        )
    return rows


def _track1_difficulty(per_combo: dict) -> list[dict[str, Any]]:
    rows = []
    for agent in _agents_for_track(per_combo, "track1_text_math"):
        groups = {"Easy": [], "Medium": [], "Hard": []}
        for trace, metrics in per_combo.get(("track1_text_math", agent), []):
            diff = _task_difficulty(trace)
            groups[diff].append(metrics)
        row = {"Agent": _agent_label(agent)}
        for label in ("Easy", "Medium", "Hard"):
            agg = _agg(groups[label])
            row[f"{label} GSR ↑"] = agg.get("gsr")
            row[f"{label} PR ↑"] = agg.get("pr")
        rows.append(row)
    return rows


def _track2_specific(per_combo: dict) -> list[dict[str, Any]]:
    rows = []
    for agent in _agents_for_track(per_combo, "track2_mooc_planning"):
        metrics = _agg([row for _, row in per_combo.get(("track2_mooc_planning", agent), [])])
        rows.append(
            {
                "Agent": _agent_label(agent),
                "Prereq Violation ↓": metrics.get("prerequisite_violation_rate"),
                "Sequence Consistency ↑": metrics.get("knowledge_sequence_consistency"),
                "Resource-Concept Match ↑": metrics.get("resource_concept_match"),
                "Path Coherence ↑": metrics.get("path_coherence"),
                "Constraint Satisfaction ↑": metrics.get("constraint_satisfaction"),
                "Difficulty Alignment ↑": metrics.get("difficulty_alignment"),
                "Plan Drift ↓": metrics.get("plan_drift"),
            }
        )
    return rows


def _track2_task_types(per_combo: dict) -> list[dict[str, Any]]:
    rows = []
    task_types = ["Goal-to-Path", "Adaptive Replan", "Constraint Planning", "Long-context Memory", "Retention Planning"]
    for agent in _agents_for_track(per_combo, "track2_mooc_planning"):
        buckets: dict[str, list[dict[str, float]]] = {t: [] for t in task_types}
        for trace, metrics in per_combo.get(("track2_mooc_planning", agent), []):
            for t in _task_types(trace):
                buckets[t].append(metrics)
        row = {"Agent": _agent_label(agent)}
        for t in task_types:
            row[f"{t} PR ↑"] = _agg(buckets[t]).get("pr")
        rows.append(row)
    return rows


def _track3_specific(per_combo: dict) -> list[dict[str, Any]]:
    rows = []
    for agent in _agents_for_track(per_combo, "track3_kt_simulator"):
        metrics = _agg([row for _, row in per_combo.get(("track3_kt_simulator", agent), [])])
        rows.append(
            {
                "Agent": _agent_label(agent),
                "Mastery Gain ↑": metrics.get("mastery_gain"),
                "Retention Gain ↑": metrics.get("retention_gain"),
                "Learning Efficiency ↑": metrics.get("learning_efficiency"),
                "Dropout Risk ↓": metrics.get("dropout_risk"),
                "Overload Rate ↓": metrics.get("overload_rate"),
                "Recovery Rate ↑": metrics.get("recovery_rate"),
                "Simulator Exploit ↓": metrics.get("simulator_exploitation_rate"),
            }
        )
    return rows


def _track3_action_diag(per_combo: dict) -> list[dict[str, Any]]:
    rows = []
    for agent in _agents_for_track(per_combo, "track3_kt_simulator"):
        dists = [action_distribution(trace) for trace, _ in per_combo.get(("track3_kt_simulator", agent), [])]
        metrics = _agg(dists)
        rows.append(
            {
                "Agent": _agent_label(agent),
                "Exercise %": metrics.get("exercise_pct"),
                "Review %": metrics.get("review_pct"),
                "Explanation %": metrics.get("explanation_pct"),
                "Diagnostic %": metrics.get("diagnostic_pct"),
                "Avg Difficulty": metrics.get("avg_difficulty"),
                "Target-concept Hit ↑": metrics.get("target_concept_hit"),
                "Fallback Rate ↓": metrics.get("fallback_rate"),
                "Unique Resources ↑": metrics.get("unique_resources"),
            }
        )
    return rows


def _agg(rows: list[dict[str, float]]) -> dict[str, float | None]:
    if not rows:
        return defaultdict(lambda: None)
    keys = sorted({k for row in rows for k in row})
    return {k: sum(float(row.get(k, 0.0) or 0.0) for row in rows) / len(rows) for k in keys}


def _agents_for_track(per_combo: dict, track: str) -> list[str]:
    present = [agent for combo_track, agent in per_combo if combo_track == track]
    if not present:
        return []
    ordered = [agent for agent in PREFERRED_AGENT_ORDER if agent in present]
    rest = sorted(agent for agent in present if agent not in set(ordered))
    return ordered + rest


def _agent_label(agent: str) -> str:
    if agent in AGENT_LABELS:
        return AGENT_LABELS[agent]
    if agent.startswith("external:"):
        raw = agent.split(":", 1)[1]
        return raw.replace("_", "-")
    return agent


def _write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _task_difficulty(trace: EpisodeTrace) -> str:
    label = trace.task.metadata.get("difficulty_group")
    if label in {"Easy", "Medium", "Hard"}:
        return label
    pool = trace.task.resource_pool
    avg = sum(r.difficulty for r in pool) / max(1, len(pool))
    if avg < 0.4:
        return "Easy"
    if avg < 0.65:
        return "Medium"
    return "Hard"


def _task_types(trace: EpisodeTrace) -> list[str]:
    explicit = trace.task.metadata.get("task_type") or trace.task.constraints.get("task_type")
    if explicit:
        return [str(explicit)]
    out = ["Goal-to-Path"]
    constraints = trace.task.constraints
    if constraints.get("must_follow_prerequisites"):
        out.append("Constraint Planning")
    if len(trace.task.resource_pool) > 10:
        out.append("Long-context Memory")
    if trace.task.horizon >= 30:
        out.append("Retention Planning")
    if trace.task.metadata.get("perturbations") or constraints.get("adaptive"):
        out.append("Adaptive Replan")
    return out
