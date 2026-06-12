from __future__ import annotations

import time
from pathlib import Path

from eduplanbench.agents import create_agent
from eduplanbench.core.io import ensure_dir, write_json, write_jsonl
from eduplanbench.core.schema import EpisodeTrace, to_plain
from eduplanbench.data.task_builders import load_tasks
from eduplanbench.envs import EduPlanEnv
from eduplanbench.evaluation.metrics import evaluate_traces
from eduplanbench.evaluation.report import write_report


def run_benchmark(
    *,
    tasks_dir: Path,
    outputs_dir: Path,
    track: str,
    agent_name: str,
    limit: int,
    llm: str | None = None,
    seed: int = 0,
    sample: str = "random",
    sample_seed: int = 42,
) -> Path:
    tasks = load_tasks(tasks_dir, track, limit=limit, sample=sample, seed=sample_seed)
    if not tasks:
        raise FileNotFoundError(f"no tasks found for {track}; run build-tasks first")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ensure_dir(outputs_dir / run_id)
    latest = outputs_dir / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.name)
    traces: list[EpisodeTrace] = []
    for idx, task in enumerate(tasks):
        agent = create_agent(agent_name, llm=llm)
        agent.reset(task)
        env = EduPlanEnv(task, seed=seed + idx)
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
        traces.append(trace)
    write_jsonl(out_dir / "episodes.jsonl.gz", (to_plain(trace) for trace in traces))
    write_json(
        out_dir / "config.snapshot.json",
        {
            "track": track,
            "agent": agent_name,
            "limit": limit,
            "llm": llm,
            "seed": seed,
            "sample": sample,
            "sample_seed": sample_seed,
            "task_ids": [task.task_id for task in tasks],
        },
    )
    report = evaluate_traces(traces, run_id=run_id)
    write_report(report, out_dir)
    return out_dir
