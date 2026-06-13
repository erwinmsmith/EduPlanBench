from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eduplanbench.core.schema import ALL_TRACKS
from eduplanbench.evaluation.robustness import run_robustness
from eduplanbench.evaluation.tables import add_robustness_table, build_tables_from_experiment
from eduplanbench.experiments import DEFAULT_AGENT_SYSTEMS, run_experiment_matrix
from scripts.build_experiment_tables_compact_xlsx import build_compact_workbook


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EduPlanBench main matrix, robustness, tables, and compact Excel.")
    parser.add_argument("--tracks", default="all", help="Comma-separated tracks or all.")
    parser.add_argument("--agents", default=",".join(DEFAULT_AGENT_SYSTEMS), help="Comma-separated agent systems.")
    parser.add_argument("--llm", default="deepseek")
    parser.add_argument("--tasks-dir", type=Path, default=Path("data/tasks"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--main-limit", type=int, default=300)
    parser.add_argument("--robust-limit", type=int, default=50)
    parser.add_argument(
        "--robust-agents",
        default="one_shot,react",
        help="Comma-separated agents for robustness runs; use 'same' to reuse --agents.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", choices=["random", "first"], default="random")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--workbook", type=Path, default=None)
    args = parser.parse_args()

    tracks = list(ALL_TRACKS) if args.tracks == "all" else [item.strip() for item in args.tracks.split(",") if item.strip()]
    agents = [item.strip() for item in args.agents.split(",") if item.strip()]
    robust_agents = agents if args.robust_agents == "same" else [item.strip() for item in args.robust_agents.split(",") if item.strip()]

    experiment_dir = run_experiment_matrix(
        tasks_dir=args.tasks_dir,
        outputs_dir=args.outputs_dir,
        tracks=tracks,
        agents=agents,
        limit=args.main_limit,
        llm=args.llm,
        seed=args.seed,
        sample=args.sample,
        sample_seed=args.sample_seed,
    )
    print(f"main experiment written to {experiment_dir}")

    table_dir = experiment_dir / "tables"
    tables = build_tables_from_experiment(experiment_dir, table_dir)
    robustness_rows = run_robustness(
        tasks_dir=args.tasks_dir,
        output_dir=experiment_dir / "robustness",
        limit=args.robust_limit,
        seed=args.seed,
        llm=args.llm,
        sample=args.sample,
        sample_seed=args.sample_seed,
        agents=robust_agents,
    )
    add_robustness_table(tables, robustness_rows, table_dir)
    print(f"robustness written to {experiment_dir / 'robustness'}")

    # Rebuild once more so any future table logic that reads robustness artifacts is reflected.
    build_tables_from_experiment(experiment_dir, table_dir)
    workbook = args.workbook or Path("outputs/workbooks") / f"EduPlanBench_Experiment_Tables_{experiment_dir.name}_compact.xlsx"
    build_compact_workbook(experiment_dir, workbook)
    print(f"compact workbook written to {workbook}")


if __name__ == "__main__":
    main()
