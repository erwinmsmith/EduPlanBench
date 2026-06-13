#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED_TRACKS = (
    "track1_text_math",
    "track2_mooc_planning",
    "track3_kt_simulator",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload prepared EduPlanBench data/processed and data/tasks to a Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("EDUPLAN_HF_DATASET_REPO"),
        help="Hugging Face dataset repo id, for example your-name/EduPlanBench-data. "
        "Defaults to EDUPLAN_HF_DATASET_REPO.",
    )
    parser.add_argument("--data-dir", default="data", help="Local prepared data directory.")
    parser.add_argument(
        "--version",
        default=os.environ.get("EDUPLAN_PREPARED_DATA_VERSION", "10k"),
        help="Prepared-data version name. Files are uploaded to versions/<version>/.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Override the Hugging Face destination path. Defaults to versions/<version>.",
    )
    parser.add_argument("--revision", default="main", help="Target branch/revision on Hugging Face.")
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private.")
    parser.add_argument(
        "--commit-message",
        default="Upload EduPlanBench prepared data",
        help="Hugging Face commit message.",
    )
    return parser.parse_args()


def require_prepared_data(data_dir: Path) -> None:
    missing: list[str] = []
    for track in REQUIRED_TRACKS:
        for rel_path in (
            Path("processed") / track / "manifest.json",
            Path("tasks") / track / "manifest.json",
            Path("tasks") / track / "tasks.jsonl",
        ):
            if not (data_dir / rel_path).exists():
                missing.append(str(data_dir / rel_path))
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Prepared data is incomplete. Build it before upload:\n"
            "  python3 -m eduplanbench data prepare --track all\n"
            "  python3 -m eduplanbench build-tasks --track all --limit 10000\n"
            f"Missing:\n{lines}"
        )


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("Set --repo-id or EDUPLAN_HF_DATASET_REPO.")

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"data directory does not exist: {data_dir}")
    require_prepared_data(data_dir)
    path_in_repo = args.path_in_repo or f"versions/{args.version.strip('/')}"

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    readme_path = data_dir / "README.md"
    if readme_path.exists():
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            revision=args.revision,
            commit_message=f"Update EduPlanBench prepared data card for {args.version}",
        )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(data_dir),
        path_in_repo=path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=[
            "README.md",
            ".hfignore",
            "processed/**",
            "tasks/**",
        ],
        ignore_patterns=[
            "**/.DS_Store",
            "**/__MACOSX/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "rawdataset/**",
            "outputs/**",
        ],
    )
    print(f"Uploaded {data_dir} to https://huggingface.co/datasets/{args.repo_id}/tree/{args.revision}/{path_in_repo}")


if __name__ == "__main__":
    main()
