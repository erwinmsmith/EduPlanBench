from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from eduplanbench.agents import create_agent
from eduplanbench.core.io import ensure_dir, read_jsonl, write_json, write_jsonl
from eduplanbench.core.schema import EpisodeTrace, TRACK1, TRACK2, TRACK3, to_plain
from eduplanbench.data.task_builders import load_tasks
from eduplanbench.data.task_builders import task_from_dict
from eduplanbench.envs import EduPlanEnv
from eduplanbench.evaluation.metrics import _metrics_for_trace


ROBUSTNESS_HORIZONS = [10, 30, 50, 100]
ROBUSTNESS_TRACKS = [TRACK1, TRACK2, TRACK3]
ROBUSTNESS_AGENTS = ["one_shot", "react"]


def run_robustness(
    *,
    tasks_dir: Path,
    output_dir: Path,
    limit: int = 1,
    seed: int = 0,
    llm: str = "deepseek",
    sample: str = "random",
    sample_seed: int = 42,
    agents: list[str] | None = None,
) -> list[dict[str, Any]]:
    out_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    task_ids: dict[str, list[str]] = {}
    agent_names = agents or list(ROBUSTNESS_AGENTS)
    for track in ROBUSTNESS_TRACKS:
        base_tasks = load_tasks(tasks_dir, track, limit=limit, sample=sample, seed=sample_seed)
        task_ids[track] = [task.task_id for task in base_tasks]
        for agent_name in agent_names:
            row: dict[str, Any] = {
                "Track": {"track1_text_math": "Track 1", "track2_mooc_planning": "Track 2", "track3_kt_simulator": "Track 3"}[track],
                "Agent": _agent_label(agent_name),
            }
            for horizon in ROBUSTNESS_HORIZONS:
                row[f"H={horizon} GSR ↑"] = None
                row[f"H={horizon} PR ↑"] = None
            for horizon in ROBUSTNESS_HORIZONS:
                trace_path = out_dir / f"{track}_{agent_name}_h{horizon}.jsonl.gz"
                traces = _load_existing_traces(trace_path, expected=len(base_tasks))
                if traces is None:
                    traces = []
                    for idx, task in enumerate(base_tasks):
                        task_h = replace(task, horizon=horizon, goal=replace(task.goal, horizon=horizon))
                        trace = _run_single(task_h, agent_name, seed=seed + idx + horizon, llm=llm)
                        traces.append(trace)
                    write_jsonl(trace_path, (to_plain(t) for t in traces))
                metrics = [_metrics_for_trace(trace) for trace in traces]
                row[f"H={horizon} GSR ↑"] = _avg(metrics, "gsr")
                row[f"H={horizon} PR ↑"] = _avg(metrics, "pr")
                _write_progress(out_dir, rows, row)
            if row not in rows:
                rows.append(row)
            write_json(out_dir / "robustness_table.json", rows)
    write_json(
        out_dir / "config.snapshot.json",
        {
            "limit": limit,
            "seed": seed,
            "llm": llm,
            "sample": sample,
            "sample_seed": sample_seed,
            "horizons": ROBUSTNESS_HORIZONS,
            "tracks": ROBUSTNESS_TRACKS,
            "agents": agent_names,
            "task_ids": task_ids,
        },
    )
    write_json(out_dir / "robustness_table.json", rows)
    return rows


def _write_progress(out_dir: Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    progress_rows = [*rows]
    if row not in progress_rows:
        progress_rows.append(row)
    write_json(out_dir / "robustness_table.json", progress_rows)


def _load_existing_traces(path: Path, *, expected: int) -> list[EpisodeTrace] | None:
    if not path.exists():
        return None
    rows = list(read_jsonl(path))
    if len(rows) != expected:
        return None
    traces: list[EpisodeTrace] = []
    for row in rows:
        trace = EpisodeTrace(task=task_from_dict(row["task"]))
        trace.steps = row.get("steps", [])
        trace.final_info = row.get("final_info", trace.steps[-1]["info"] if trace.steps else {})
        traces.append(trace)
    return traces


def _run_single(task, agent_name: str, *, seed: int, llm: str) -> EpisodeTrace:
    agent = create_agent(agent_name, llm=llm)
    agent.reset(task)
    env = EduPlanEnv(task, seed=seed)
    obs = env.reset()
    trace = EpisodeTrace(task=task)
    done = False
    while not done:
        action = agent.act(obs)
        valid, error = action.validate_for(obs.candidate_resources)
        result = env.step(action)
        trace.add_step(obs, action, result, valid, error)
        obs = result.observation
        done = result.done
    trace.final_info = trace.steps[-1]["info"] if trace.steps else {}
    return trace


def _avg(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def _agent_label(agent_name: str) -> str:
    return {
        "one_shot": "One-shot",
        "react": "ReAct",
        "cot": "CoT",
        "step_by_step": "Step-by-step",
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
    }.get(agent_name, agent_name)
