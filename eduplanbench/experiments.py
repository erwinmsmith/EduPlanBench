from __future__ import annotations

import time
from pathlib import Path

from eduplanbench.core.io import ensure_dir, write_csv, write_json
from eduplanbench.core.schema import ALL_TRACKS
from eduplanbench.runner import run_benchmark


DEFAULT_AGENT_SYSTEMS = ["react", "one_shot", "step_by_step", "cot"]


def run_experiment_matrix(
    *,
    tasks_dir: Path,
    outputs_dir: Path,
    tracks: list[str] | None = None,
    agents: list[str] | None = None,
    limit: int = 5,
    llm: str = "deepseek",
    seed: int = 0,
    sample: str = "random",
    sample_seed: int = 42,
) -> Path:
    tracks = tracks or list(ALL_TRACKS)
    agents = agents or list(DEFAULT_AGENT_SYSTEMS)
    experiment_id = f"{time.strftime('experiment-%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
    experiment_dir = ensure_dir(outputs_dir / experiment_id)
    rows: list[dict] = []
    for track in tracks:
        for agent in agents:
            run_dir = run_benchmark(
                tasks_dir=tasks_dir,
                outputs_dir=experiment_dir,
                track=track,
                agent_name=agent,
                llm=llm,
                limit=limit,
                seed=seed,
                sample=sample,
                sample_seed=sample_seed,
            )
            metrics = _read_metrics(run_dir)
            row = {"track": track, "agent": agent, "run_dir": str(run_dir), "sample": sample, "sample_seed": sample_seed, **metrics}
            rows.append(row)
    write_csv(experiment_dir / "matrix_results.csv", rows)
    write_json(experiment_dir / "matrix_results.json", rows)
    _write_matrix_report(experiment_dir, rows)
    return experiment_dir


def _read_metrics(run_dir: Path) -> dict:
    import json

    with (run_dir / "metrics.json").open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("metrics", {})


def _write_matrix_report(experiment_dir: Path, rows: list[dict]) -> None:
    with (experiment_dir / "matrix_report.md").open("w", encoding="utf-8") as fh:
        fh.write("# EduPlanBench Experiment Matrix\n\n")
        fh.write("| Track | Agent | Overall | Core | Track Score | GSR | PR | Valid |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fh.write(
                f"| {row['track']} | {row['agent']} | "
                f"{row.get('overall_score', 0):.4f} | {row.get('core_score', 0):.4f} | "
                f"{row.get('track_score', 0):.4f} | {row.get('gsr', 0):.4f} | "
                f"{row.get('pr', 0):.4f} | {row.get('valid_action_rate', 0):.4f} |\n"
            )
