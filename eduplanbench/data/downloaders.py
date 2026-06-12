from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from eduplanbench.core.io import ensure_dir


MOOCCUBEX_BASE = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2"
MOOCCUBEX_MINIMAL_FILES: dict[str, str] = {
    "entities/course.json": f"{MOOCCUBEX_BASE}/entities/course.json",
    "entities/concept.json": f"{MOOCCUBEX_BASE}/entities/concept.json",
    "relations/concept-course.txt": f"{MOOCCUBEX_BASE}/relations/concept-course.txt",
    "relations/concept-video.txt": f"{MOOCCUBEX_BASE}/relations/concept-video.txt",
    "relations/concept-problem.txt": f"{MOOCCUBEX_BASE}/relations/concept-problem.txt",
    "prerequisites/math.json": f"{MOOCCUBEX_BASE}/prerequisites/math.json",
    "prerequisites/cs.json": f"{MOOCCUBEX_BASE}/prerequisites/cs.json",
    "prerequisites/psy.json": f"{MOOCCUBEX_BASE}/prerequisites/psy.json",
}


HF_DATASETS = {
    "mathdial": "eth-nlped/mathdial",
    "misstepmath": "LLMEducation/MisstepMath",
}


@dataclass(slots=True)
class DownloadResult:
    dataset: str
    path: Path
    downloaded: bool
    message: str


def fetch_huggingface_dataset(dataset: str, raw_dir: Path) -> DownloadResult:
    if dataset not in HF_DATASETS:
        raise ValueError(f"unknown Hugging Face dataset alias: {dataset}")
    target = raw_dir / dataset
    if target.exists() and any(target.iterdir()):
        return DownloadResult(dataset, target, False, "already present")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return DownloadResult(
            dataset,
            target,
            False,
            "huggingface_hub is not installed; install eduplanbench[download] or place files manually",
        )

    ensure_dir(target)
    snapshot_download(
        repo_id=HF_DATASETS[dataset],
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return DownloadResult(dataset, target, True, "downloaded")


def fetch_mooccubex_minimal(raw_dir: Path) -> list[DownloadResult]:
    base = raw_dir / "MOOCCubeX"
    results: list[DownloadResult] = []
    for rel_path, url in MOOCCUBEX_MINIMAL_FILES.items():
        target = base / rel_path
        if target.exists() and target.stat().st_size > 0:
            results.append(DownloadResult(rel_path, target, False, "already present"))
            continue
        ensure_dir(target.parent)
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        results.append(DownloadResult(rel_path, target, True, "downloaded"))
    return results
