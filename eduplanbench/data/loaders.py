from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from eduplanbench.core.io import iter_csv, parse_list_cell, read_json


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def iter_eedi_answers(eedi_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    train_dir = require_path(eedi_dir / "train_data", "Eedi train_data")
    required = {"QuestionId", "UserId", "AnswerId", "IsCorrect", "CorrectAnswer", "AnswerValue"}
    seen = 0
    for csv_path in sorted(train_dir.glob("*.csv")):
        for row in iter_csv(csv_path):
            missing = required - set(row)
            if missing:
                raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
            yield {
                "source": "eedi",
                "question_id": row["QuestionId"],
                "student_id": row["UserId"],
                "answer_id": row["AnswerId"],
                "is_correct": row["IsCorrect"] == "1",
                "correct_answer": row["CorrectAnswer"],
                "answer_value": row["AnswerValue"],
            }
            seen += 1
            if limit is not None and seen >= limit:
                return


def load_eedi_subjects(eedi_dir: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for csv_path in sorted((eedi_dir / "metadata").glob("question_metadata*.csv")):
        for row in iter_csv(csv_path):
            out[row["QuestionId"]] = parse_list_cell(row.get("SubjectId"))
    return out


def load_subject_names(eedi_dir: Path) -> dict[str, str]:
    path = eedi_dir / "metadata" / "subject_metadata.csv"
    if not path.exists():
        return {}
    return {row["SubjectId"]: row.get("Name", row["SubjectId"]) for row in iter_csv(path)}


def iter_mathdial(mathdial_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    yield from _iter_rows_from_dir(mathdial_dir, limit=limit, source="mathdial")


def iter_misstepmath(misstep_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    yield from _iter_rows_from_dir(misstep_dir, limit=limit, source="misstepmath")


def _iter_rows_from_dir(path: Path, *, limit: int | None, source: str) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    count = 0
    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() == ".csv":
            for row in iter_csv(file_path):
                row["source"] = source
                yield row
                count += 1
                if limit is not None and count >= limit:
                    return
        elif file_path.suffix.lower() in {".jsonl", ".json"}:
            for row in _iter_json_like(file_path):
                if isinstance(row, dict):
                    row["source"] = source
                    yield row
                    count += 1
                    if limit is not None and count >= limit:
                        return


def _iter_json_like(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    payload = read_json(path)
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        for key in ("train", "test", "data", "rows"):
            if isinstance(payload.get(key), list):
                yield from payload[key]
                return
        yield payload


def iter_xes_sequences(xes_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    csv_path = xes_dir / "kc_level" / "train_valid_sequences.csv"
    require_path(csv_path, "XES3G5M kc_level train_valid_sequences.csv")
    questions = load_xes_questions(xes_dir)
    kc_names = load_xes_kc_names(xes_dir)
    for idx, row in enumerate(iter_csv(csv_path, limit=limit)):
        question_ids = parse_list_cell(row.get("questions"))
        concept_ids = parse_list_cell(row.get("concepts"))
        yield {
            "source": "xes3g5m",
            "student_id": row["uid"],
            "questions": question_ids,
            "concepts": concept_ids,
            "responses": parse_list_cell(row.get("responses")),
            "timestamps": parse_list_cell(row.get("timestamps")),
            "is_repeat": parse_list_cell(row.get("is_repeat")),
            "question_texts": {qid: questions.get(qid, {}) for qid in set(question_ids) if qid in questions},
            "kc_names": {cid: kc_names.get(cid, cid) for cid in set(concept_ids)},
        }


def load_xes_questions(xes_dir: Path) -> dict[str, dict[str, Any]]:
    path = xes_dir / "metadata" / "questions.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return {str(key): value for key, value in payload.items()}


def load_xes_kc_names(xes_dir: Path) -> dict[str, str]:
    path = xes_dir / "metadata" / "kc_routes_map.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return {str(key): str(value) for key, value in payload.items()}


def iter_kt1_sequences(kt1_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    if not kt1_dir.exists():
        return
    count = 0
    for csv_path in sorted(kt1_dir.glob("*.csv")):
        rows = list(iter_csv(csv_path))
        if not rows:
            continue
        yield {
            "source": "kt1",
            "student_id": csv_path.stem.lstrip("u"),
            "events": rows,
        }
        count += 1
        if limit is not None and count >= limit:
            return


def iter_kt1_index(kt1_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    if not kt1_dir.exists():
        return
    count = 0
    for csv_path in sorted(kt1_dir.glob("*.csv")):
        yield {
            "source": "kt1",
            "student_id": csv_path.stem.lstrip("u"),
            "path": str(csv_path),
        }
        count += 1
        if limit is not None and count >= limit:
            return
