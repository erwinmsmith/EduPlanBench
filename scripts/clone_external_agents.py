from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "external_agents.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone external agent repositories used by EduPlanBench adapters.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agent", default="all", help="Agent id to clone, or all.")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--update", action="store_true", help="Pull latest changes if a repository already exists.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    agents = config.get("agents", {})
    selected = agents if args.agent == "all" else {args.agent: agents[args.agent]}
    for name, spec in selected.items():
        clone_or_update(name, spec, depth=args.depth, update=args.update)


def clone_or_update(name: str, spec: dict, *, depth: int, update: bool) -> None:
    repo_url = spec["repo_url"]
    repo_path = ROOT / spec["repo_path"]
    if (repo_path / ".git").exists():
        print(f"{name}: exists at {repo_path}")
        if update:
            subprocess.run(["git", "-C", str(repo_path), "pull", "--ff-only"], check=True)
        return
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth), repo_url, str(repo_path)]
    print(f"{name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
