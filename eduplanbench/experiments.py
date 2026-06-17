from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
    experiment_dir: Path | None = None,
) -> Path:
    tracks = tracks or list(ALL_TRACKS)
    agents = agents or list(DEFAULT_AGENT_SYSTEMS)
    if experiment_dir is None:
        experiment_id = f"{time.strftime('experiment-%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
        experiment_dir = ensure_dir(outputs_dir / experiment_id)
    else:
        experiment_dir = ensure_dir(experiment_dir)
    rows_by_key = _load_completed_rows(experiment_dir)
    for track in tracks:
        for agent in agents:
            key = (track, agent)
            if key in rows_by_key:
                continue
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
            rows_by_key[key] = row
            _write_matrix_outputs(experiment_dir, _ordered_rows(rows_by_key, tracks, agents))
    _write_matrix_outputs(experiment_dir, _ordered_rows(rows_by_key, tracks, agents))
    return experiment_dir


def _read_metrics(run_dir: Path) -> dict:
    import json

    with (run_dir / "metrics.json").open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("metrics", {})


def _load_completed_rows(experiment_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for config_path in sorted(experiment_dir.glob("*/config.snapshot.json")):
        if config_path.parent.is_symlink():
            continue
        metrics_path = config_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            metrics = _read_metrics(config_path.parent)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        track = str(config.get("track", ""))
        agent = str(config.get("agent", ""))
        if not track or not agent:
            continue
        rows[(track, agent)] = {
            "track": track,
            "agent": agent,
            "run_dir": str(config_path.parent),
            "sample": config.get("sample", ""),
            "sample_seed": config.get("sample_seed", ""),
            **metrics,
        }
    return rows


def _ordered_rows(rows_by_key: dict[tuple[str, str], dict[str, Any]], tracks: list[str], agents: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for track in tracks:
        for agent in agents:
            key = (track, agent)
            if key in rows_by_key:
                rows.append(rows_by_key[key])
                seen.add(key)
    for key in sorted(set(rows_by_key) - seen):
        rows.append(rows_by_key[key])
    return rows


def _write_matrix_outputs(experiment_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(experiment_dir / "matrix_results.csv", rows)
    write_json(experiment_dir / "matrix_results.json", rows)
    _write_matrix_report(experiment_dir, rows)


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
