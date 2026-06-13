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
    parser.add_argument(
        "--version",
        default=os.environ.get("EDUPLAN_PREPARED_DATA_VERSION", "10k"),
        help="Prepared-data version name to download from versions/<version>/.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Override the Hugging Face source path. Defaults to versions/<version>.",
    )
    parser.add_argument("--revision", default="main", help="Hugging Face revision to download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("Set --repo-id or EDUPLAN_HF_DATASET_REPO.")

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    path_in_repo = args.path_in_repo or f"versions/{args.version.strip('/')}"
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(data_dir),
        allow_patterns=[
            f"{path_in_repo}/README.md",
            f"{path_in_repo}/processed/**",
            f"{path_in_repo}/tasks/**",
        ],
        ignore_patterns=[
            "**/.DS_Store",
            "**/__MACOSX/**",
            "**/__pycache__/**",
            "**/*.pyc",
        ],
    )
    nested = data_dir / path_in_repo
    if nested.exists():
        for child in nested.iterdir():
            target = data_dir / child.name
            if target.exists():
                if target.is_dir():
                    import shutil

                    shutil.rmtree(target)
                else:
                    target.unlink()
            child.replace(target)
        parts = Path(path_in_repo).parts
        cleanup = data_dir
        for part in parts:
            cleanup = cleanup / part
        for parent in [cleanup, *cleanup.parents]:
            if parent == data_dir or not parent.exists():
                break
            try:
                parent.rmdir()
            except OSError:
                break
    print(f"Downloaded https://huggingface.co/datasets/{args.repo_id}/tree/{args.revision}/{path_in_repo} into {data_dir}")
    print("You can now run, for example:")
    print(
        "  python3 -m eduplanbench experiment "
        "--tracks all --agents react,one_shot,step_by_step,cot "
        "--llm deepseek --limit 300 --sample random --sample-seed 42"
    )


if __name__ == "__main__":
    main()
