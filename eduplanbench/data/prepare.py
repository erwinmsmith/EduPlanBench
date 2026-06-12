from __future__ import annotations

from pathlib import Path
from typing import Any

from eduplanbench.core.io import ensure_dir, write_json, write_jsonl
from eduplanbench.core.schema import TRACK1, TRACK2, TRACK3
from eduplanbench.data.loaders import (
    iter_eedi_answers,
    iter_kt1_sequences,
    iter_kt1_index,
    iter_mathdial,
    iter_misstepmath,
    iter_xes_sequences,
    load_eedi_subjects,
    load_subject_names,
)
from eduplanbench.graphs.resource_graph import TextResourceGraph


def prepare_track1(raw_dir: Path, processed_dir: Path, *, limit: int | None = 10_000) -> dict[str, Any]:
    out_dir = ensure_dir(processed_dir / TRACK1)
    eedi_dir = raw_dir / "public_data"
    subjects = load_eedi_subjects(eedi_dir)
    subject_names = load_subject_names(eedi_dir)

    def eedi_rows():
        for row in iter_eedi_answers(eedi_dir, limit=limit):
            subject_ids = subjects.get(row["question_id"], [])
            concepts = [subject_names.get(item, item) for item in subject_ids]
            yield {
                "case_id": f"eedi_{row['answer_id']}",
                "source": "eedi",
                "student_id": row["student_id"],
                "problem_id": row["question_id"],
                "problem_text": f"Eedi diagnostic question {row['question_id']}",
                "student_answer": row["answer_value"],
                "correct_answer": row["correct_answer"],
                "is_correct": row["is_correct"],
                "concepts": concepts,
                "misconception": "" if row["is_correct"] else "wrong diagnostic option",
                "remediation_reference": "",
                "metadata": row,
            }

    eedi_count = write_jsonl(out_dir / "eedi_cases.jsonl.gz", eedi_rows())

    def mathdial_rows():
        for idx, row in enumerate(iter_mathdial(raw_dir / "mathdial", limit=limit)):
            yield {
                "case_id": f"mathdial_{row.get('qid', idx)}",
                "source": "mathdial",
                "student_id": "",
                "problem_id": str(row.get("qid", idx)),
                "problem_text": row.get("question", ""),
                "student_answer": row.get("student_incorrect_solution", ""),
                "correct_answer": row.get("ground_truth", ""),
                "is_correct": False,
                "concepts": [],
                "misconception": row.get("teacher_described_confusion", ""),
                "remediation_reference": row.get("conversation", ""),
                "metadata": row,
            }

    mathdial_count = write_jsonl(out_dir / "mathdial_cases.jsonl.gz", mathdial_rows())

    def misstep_rows():
        for idx, row in enumerate(iter_misstepmath(raw_dir / "misstepmath", limit=limit)):
            topic = row.get("Topic") or row.get("topic") or ""
            subtopic = row.get("Sub Topic") or row.get("SubTopic") or row.get("subtopic") or ""
            yield {
                "case_id": f"misstepmath_{idx}",
                "source": "misstepmath",
                "student_id": "",
                "problem_id": str(idx),
                "problem_text": row.get("Problem", ""),
                "student_answer": row.get("Student's mistake prompt", ""),
                "correct_answer": "",
                "is_correct": False,
                "concepts": [item for item in (topic, subtopic) if item],
                "misconception": row.get("Challenge faced", ""),
                "remediation_reference": row.get("Teacher's resolution- text based prompt", ""),
                "metadata": row,
            }

    misstep_count = write_jsonl(out_dir / "misstepmath_cases.jsonl.gz", misstep_rows())
    manifest = {
        "track": TRACK1,
        "eedi_cases": eedi_count,
        "mathdial_cases": mathdial_count,
        "misstepmath_cases": misstep_count,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def prepare_track2(raw_dir: Path, processed_dir: Path, *, limit: int | None = 50_000) -> dict[str, Any]:
    out_dir = ensure_dir(processed_dir / TRACK2)
    graph = TextResourceGraph.from_mooccubex(raw_dir / "MOOCCubeX", limit=limit)
    graph_path = out_dir / "resource_graph.json"
    graph.to_json(graph_path)
    manifest = {
        "track": TRACK2,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "graph": str(graph_path),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def prepare_track3(raw_dir: Path, processed_dir: Path, *, limit: int | None = 10_000) -> dict[str, Any]:
    out_dir = ensure_dir(processed_dir / TRACK3)
    xes_count = write_jsonl(out_dir / "xes_sequences.jsonl.gz", iter_xes_sequences(raw_dir / "XES3G5M", limit=limit))
    kt1_index_count = write_jsonl(out_dir / "kt1_index.jsonl.gz", iter_kt1_index(raw_dir / "KT1", limit=None))
    kt1_sample_count = write_jsonl(out_dir / "kt1_sequences.sample.jsonl.gz", iter_kt1_sequences(raw_dir / "KT1", limit=limit or 1000))
    manifest = {
        "track": TRACK3,
        "xes_sequences": xes_count,
        "kt1_indexed_students": kt1_index_count,
        "kt1_sample_sequences": kt1_sample_count,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
