from __future__ import annotations

import csv
import gzip
import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import to_plain


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(to_plain(payload), fh, ensure_ascii=False, indent=indent)
        fh.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    target = Path(path)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    target = Path(path)
    ensure_dir(target.parent)
    count = 0
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "wt", encoding="utf-8") as fh:
        for row in rows:
            if is_dataclass(row):
                row = to_plain(row)
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def iter_csv(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield {str(key): value for key, value in row.items()}


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    keys: list[str] = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_list_cell(value: str | None) -> list[str]:
    if not value:
        return []
    text = value.strip()
    if text in {"", "-1"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            loaded = json.loads(text)
            return [str(item) for item in loaded]
        except json.JSONDecodeError:
            pass
    sep = "_" if "_" in text and "," not in text else ","
    return [part.strip() for part in text.split(sep) if part.strip() and part.strip() != "-1"]
