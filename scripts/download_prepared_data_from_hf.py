#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download prepared EduPlanBench data from a Hugging Face dataset repo into data/.",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("EDUPLAN_HF_DATASET_REPO"),
        help="Hugging Face dataset repo id, for example your-name/EduPlanBench-data. "
        "Defaults to EDUPLAN_HF_DATASET_REPO.",
    )
    parser.add_argument("--data-dir", default="data", help="Local destination directory.")
    parser.add_argument("--revision", default="main", help="Hugging Face revision to download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("Set --repo-id or EDUPLAN_HF_DATASET_REPO.")

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(data_dir),
        allow_patterns=[
            "README.md",
            "processed/**",
            "tasks/**",
        ],
        ignore_patterns=[
            "**/.DS_Store",
            "**/__MACOSX/**",
            "**/__pycache__/**",
            "**/*.pyc",
        ],
    )
    print(f"Downloaded https://huggingface.co/datasets/{args.repo_id} into {data_dir}")
    print("You can now run, for example:")
    print(
        "  python3 -m eduplanbench experiment "
        "--tracks all --agents react,one_shot,step_by_step,cot "
        "--llm deepseek --limit 300 --sample random --sample-seed 42"
    )


if __name__ == "__main__":
    main()
