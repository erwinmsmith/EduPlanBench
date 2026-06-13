from __future__ import annotations

import argparse
from pathlib import Path

from eduplanbench.core.env import load_dotenv
from eduplanbench.core.io import read_jsonl
from eduplanbench.core.schema import ALL_TRACKS, TRACK1, TRACK2, TRACK3, EpisodeTrace
from eduplanbench.agents import external_agent_status
from eduplanbench.data.downloaders import fetch_huggingface_dataset, fetch_mooccubex_minimal
from eduplanbench.data.prepare import prepare_track1, prepare_track2, prepare_track3
from eduplanbench.data.task_builders import build_tasks, task_from_dict
from eduplanbench.evaluation.metrics import evaluate_traces
from eduplanbench.evaluation.report import write_report
from eduplanbench.evaluation.tables import build_tables_from_experiment
from eduplanbench.evaluation.robustness import run_robustness
from eduplanbench.evaluation.tables import add_robustness_table
from eduplanbench.runner import run_benchmark
from eduplanbench.experiments import DEFAULT_AGENT_SYSTEMS, run_experiment_matrix


DEFAULT_RAW = Path("rawdataset")
DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_TASKS = Path("data/tasks")
DEFAULT_OUTPUTS = Path("outputs/runs")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="eduplanbench")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch")
    fetch.add_argument("--track", choices=["track1", "track2", TRACK1, TRACK2], required=True)
    fetch.add_argument("--datasets", default="mathdial,misstepmath")
    fetch.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)

    prepare = data_sub.add_parser("prepare")
    prepare.add_argument("--track", default="all", choices=["all", *ALL_TRACKS, "track1", "track2", "track3"])
    prepare.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    prepare.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    prepare.add_argument("--limit", type=int, default=0, help="0 means full available data")

    build = sub.add_parser("build-tasks")
    build.add_argument("--track", default="all", choices=["all", *ALL_TRACKS, "track1", "track2", "track3"])
    build.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    build.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS)
    build.add_argument("--limit", type=int, default=100)

    run = sub.add_parser("run")
    run.add_argument("--track", default=TRACK3, choices=["all", *ALL_TRACKS, "track1", "track2", "track3"])
    run.add_argument("--agent", default="kt_recommender")
    run.add_argument("--llm", default=None)
    run.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS)
    run.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    run.add_argument("--limit", type=int, default=10)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--sample", choices=["random", "first"], default="random")
    run.add_argument("--sample-seed", type=int, default=42)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--runs", type=Path, required=True)

    report = sub.add_parser("report")
    report.add_argument("--runs", type=Path, required=True)

    tables = sub.add_parser("tables")
    tables.add_argument("--experiment-dir", type=Path, required=True)
    tables.add_argument("--output-dir", type=Path, default=None)

    robustness = sub.add_parser("robustness")
    robustness.add_argument("--experiment-dir", type=Path, required=True)
    robustness.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS)
    robustness.add_argument("--limit", type=int, default=1)
    robustness.add_argument("--llm", default="deepseek")
    robustness.add_argument("--seed", type=int, default=0)
    robustness.add_argument("--sample", choices=["random", "first"], default="random")
    robustness.add_argument("--sample-seed", type=int, default=42)

    experiment = sub.add_parser("experiment")
    experiment.add_argument("--tracks", default="all")
    experiment.add_argument("--agents", default=",".join(DEFAULT_AGENT_SYSTEMS))
    experiment.add_argument("--llm", default="deepseek")
    experiment.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS)
    experiment.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    experiment.add_argument("--limit", type=int, default=5)
    experiment.add_argument("--seed", type=int, default=0)
    experiment.add_argument("--sample", choices=["random", "first"], default="random")
    experiment.add_argument("--sample-seed", type=int, default=42)

    agents_cmd = sub.add_parser("agents")
    agents_sub = agents_cmd.add_subparsers(dest="agents_command", required=True)
    agents_sub.add_parser("list")

    args = parser.parse_args(argv)
    if args.command == "data" and args.data_command == "fetch":
        _cmd_fetch(args)
    elif args.command == "data" and args.data_command == "prepare":
        _cmd_prepare(args)
    elif args.command == "build-tasks":
        _cmd_build(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "tables":
        _cmd_tables(args)
    elif args.command == "robustness":
        _cmd_robustness(args)
    elif args.command == "experiment":
        _cmd_experiment(args)
    elif args.command == "agents" and args.agents_command == "list":
        _cmd_agents_list(args)


def _track_alias(track: str) -> str:
    return {"track1": TRACK1, "track2": TRACK2, "track3": TRACK3}.get(track, track)


def _cmd_fetch(args: argparse.Namespace) -> None:
    track = _track_alias(args.track)
    if track == TRACK1:
        for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
            result = fetch_huggingface_dataset(dataset, args.raw_dir)
            print(f"{result.dataset}: {result.message} -> {result.path}")
    elif track == TRACK2:
        for result in fetch_mooccubex_minimal(args.raw_dir):
            print(f"{result.dataset}: {result.message} -> {result.path}")


def _cmd_prepare(args: argparse.Namespace) -> None:
    tracks = ALL_TRACKS if args.track == "all" else (_track_alias(args.track),)
    limit = None if args.limit <= 0 else args.limit
    for track in tracks:
        if track == TRACK1:
            manifest = prepare_track1(args.raw_dir, args.processed_dir, limit=limit)
        elif track == TRACK2:
            manifest = prepare_track2(args.raw_dir, args.processed_dir, limit=limit)
        elif track == TRACK3:
            manifest = prepare_track3(args.raw_dir, args.processed_dir, limit=limit)
        else:
            raise ValueError(track)
        print(manifest)


def _cmd_build(args: argparse.Namespace) -> None:
    track = "all" if args.track == "all" else _track_alias(args.track)
    print(build_tasks(track, args.processed_dir, args.tasks_dir, limit=args.limit))


def _cmd_run(args: argparse.Namespace) -> None:
    track = "all" if args.track == "all" else _track_alias(args.track)
    out_dir = run_benchmark(
        tasks_dir=args.tasks_dir,
        outputs_dir=args.outputs_dir,
        track=track,
        agent_name=args.agent,
        llm=args.llm,
        limit=args.limit,
        seed=args.seed,
        sample=args.sample,
        sample_seed=args.sample_seed,
    )
    print(f"run written to {out_dir}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    traces = _load_episode_traces(args.runs)
    metric_report = evaluate_traces(traces, run_id=args.runs.name)
    write_report(metric_report, args.runs)
    print(f"evaluated {len(traces)} episodes in {args.runs}")


def _cmd_report(args: argparse.Namespace) -> None:
    metrics = args.runs / "metrics.json"
    report = args.runs / "report.md"
    if not metrics.exists():
        _cmd_evaluate(args)
    print(report)


def _cmd_experiment(args: argparse.Namespace) -> None:
    tracks = list(ALL_TRACKS) if args.tracks == "all" else [_track_alias(item.strip()) for item in args.tracks.split(",") if item.strip()]
    agents = [item.strip() for item in args.agents.split(",") if item.strip()]
    out_dir = run_experiment_matrix(
        tasks_dir=args.tasks_dir,
        outputs_dir=args.outputs_dir,
        tracks=tracks,
        agents=agents,
        llm=args.llm,
        limit=args.limit,
        seed=args.seed,
        sample=args.sample,
        sample_seed=args.sample_seed,
    )
    print(f"experiment written to {out_dir}")


def _cmd_tables(args: argparse.Namespace) -> None:
    out = args.output_dir or (args.experiment_dir / "tables")
    build_tables_from_experiment(args.experiment_dir, out)
    print(f"tables written to {out}")


def _cmd_robustness(args: argparse.Namespace) -> None:
    table_dir = args.experiment_dir / "tables"
    tables = build_tables_from_experiment(args.experiment_dir, table_dir)
    rows = run_robustness(
        tasks_dir=args.tasks_dir,
        output_dir=args.experiment_dir / "robustness",
        limit=args.limit,
        seed=args.seed,
        llm=args.llm,
        sample=args.sample,
        sample_seed=args.sample_seed,
    )
    add_robustness_table(tables, rows, table_dir)
    print(f"robustness written to {args.experiment_dir / 'robustness'}")


def _cmd_agents_list(args: argparse.Namespace) -> None:
    builtins = [
        "react",
        "one_shot",
        "step_by_step",
        "cot",
        "kt_recommender",
        "static_one_shot",
        "random",
        "prerequisite_rule",
        "difficulty_rule",
        "oracle",
    ]
    print("Built-in agents:")
    for name in builtins:
        print(f"  {name}")
    print("\nExternal agents:")
    for row in external_agent_status():
        enabled = "enabled" if row["enabled"] else "disabled"
        repo = "repo-ok" if row["repo_exists"] else "repo-missing"
        print(f"  {row['name']}: {row['display_name']} [{enabled}, {repo}, {row['protocol']}]")


def _load_episode_traces(run_dir: Path) -> list[EpisodeTrace]:
    path = run_dir / "episodes.jsonl"
    if not path.exists():
        path = run_dir / "episodes.jsonl.gz"
    traces: list[EpisodeTrace] = []
    for row in read_jsonl(path):
        trace = EpisodeTrace(task=task_from_dict(row["task"]))
        trace.steps = row.get("steps", [])
        trace.final_info = row.get("final_info", trace.steps[-1]["info"] if trace.steps else {})
        traces.append(trace)
    return traces


if __name__ == "__main__":
    main()
