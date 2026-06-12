from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

from eduplanbench.core.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from eduplanbench.core.schema import (
    ALL_TRACKS,
    TRACK1,
    TRACK2,
    TRACK3,
    GoalSpec,
    LearnerProfile,
    Resource,
    TaskInstance,
    to_plain,
)
from eduplanbench.graphs.resource_graph import TextResourceGraph


def build_tasks(track: str, processed_dir: Path, tasks_dir: Path, *, limit: int = 100) -> dict[str, Any]:
    if track == "all":
        manifests = {}
        for item in ALL_TRACKS:
            manifests[item] = build_tasks(item, processed_dir, tasks_dir, limit=limit)
        return manifests
    out_dir = ensure_dir(tasks_dir / track)
    if track == TRACK1:
        tasks = list(_build_track1(processed_dir / track, limit=limit))
    elif track == TRACK2:
        tasks = list(_build_track2(processed_dir / track, limit=limit))
    elif track == TRACK3:
        tasks = list(_build_track3(processed_dir / track, limit=limit))
    else:
        raise ValueError(f"unknown track: {track}")
    count = write_jsonl(out_dir / "tasks.jsonl", (to_plain(task) for task in tasks))
    manifest = {"track": track, "tasks": count, "path": str(out_dir / "tasks.jsonl")}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_tasks(
    tasks_dir: Path,
    track: str,
    *,
    limit: int | None = None,
    sample: str = "first",
    seed: int = 0,
) -> list[TaskInstance]:
    tracks = ALL_TRACKS if track == "all" else (track,)
    tasks: list[TaskInstance] = []
    rng = random.Random(seed)
    for item in tracks:
        path = tasks_dir / item / "tasks.jsonl"
        if not path.exists():
            continue
        rows = _sample_rows(path, limit=limit, sample=sample, rng=rng)
        tasks.extend(task_from_dict(row) for row in rows)
    return tasks


def _sample_rows(path: Path, *, limit: int | None, sample: str, rng: random.Random) -> list[dict[str, Any]]:
    if sample not in {"first", "random"}:
        raise ValueError(f"unknown sample mode: {sample}")
    if sample == "first":
        rows = []
        for row in read_jsonl(path):
            if limit is not None and len(rows) >= limit:
                break
            rows.append(row)
        return rows
    rows = list(read_jsonl(path))
    if limit is None or limit >= len(rows):
        rng.shuffle(rows)
        return rows
    return rng.sample(rows, limit)


def task_from_dict(row: dict[str, Any]) -> TaskInstance:
    return TaskInstance(
        task_id=row["task_id"],
        track=row["track"],
        domain=row["domain"],
        horizon=int(row["horizon"]),
        learner_profile=LearnerProfile(**row["learner_profile"]),
        goal=GoalSpec(**row["goal"]),
        resource_pool=[Resource(**item) for item in row.get("resource_pool", [])],
        constraints=row.get("constraints", {}),
        metadata=row.get("metadata", {}),
    )


def _build_track1(root: Path, *, limit: int) -> Iterator[TaskInstance]:
    eedi = _take_jsonl(_first_existing(root, "eedi_cases.jsonl.gz", "eedi_cases.jsonl"), limit)
    mathdial = _take_jsonl(_first_existing(root, "mathdial_cases.jsonl.gz", "mathdial_cases.jsonl"), max(limit, 1))
    misstep = _take_jsonl(_first_existing(root, "misstepmath_cases.jsonl.gz", "misstepmath_cases.jsonl"), max(limit, 1))
    if not eedi:
        eedi = mathdial or misstep
    for idx, case in enumerate(eedi[:limit]):
        difficulty_label, resource_difficulty, target_mastery, horizon = _track1_difficulty_profile(idx)
        concepts = case.get("concepts") or ["general_math"]
        md = mathdial[idx % len(mathdial)] if mathdial else {}
        ms = _best_misstep(concepts, misstep) if misstep else {}
        exercise = Resource(
            resource_id=f"track1_case_{idx}",
            type="exercise",
            title=f"Diagnostic case {idx}",
            text=case.get("problem_text", ""),
            concepts=concepts,
            difficulty=resource_difficulty,
            metadata=case,
        )
        resources = [exercise]
        if md:
            resources.append(
                Resource(
                    resource_id=f"mathdial_ref_{idx}",
                    type="explanation",
                    title="MathDial tutoring reference",
                    text=f"Question: {md.get('problem_text', '')}\nStudent mistake: {md.get('student_answer', '')}\nTeacher dialogue: {md.get('remediation_reference', '')}",
                    concepts=concepts,
                    difficulty=max(0.2, resource_difficulty - 0.2),
                    metadata=md,
                )
            )
        if ms:
            resources.append(
                Resource(
                    resource_id=f"misstep_ref_{idx}",
                    type="explanation",
                    title="MisstepMath remediation reference",
                    text=f"Misconception: {ms.get('misconception', '')}\nRemediation: {ms.get('remediation_reference', '')}",
                    concepts=ms.get("concepts") or concepts,
                    difficulty=max(0.2, resource_difficulty - 0.25),
                    metadata=ms,
                )
            )
        base_mastery = {"Easy": 0.45, "Medium": 0.34, "Hard": 0.24}[difficulty_label]
        profile = LearnerProfile(
            profile_text=f"Student has a recent incorrect or diagnostic interaction on {', '.join(concepts[:3])}.",
            estimated_mastery={concept: base_mastery for concept in concepts[:5]},
            weak_concepts=concepts[:5],
            recent_errors=[
                {
                    "concept": concepts[0],
                    "error_type": case.get("misconception", "unknown"),
                    "evidence": case.get("student_answer", ""),
                }
            ],
        )
        yield TaskInstance(
            task_id=f"track1_{idx:06d}",
            track=TRACK1,
            domain="math",
            horizon=horizon,
            learner_profile=profile,
            goal=GoalSpec(target_concepts=concepts[:1], target_mastery=target_mastery, horizon=horizon),
            resource_pool=resources,
            constraints={"text_only": True, "must_address_misconception": True},
            metadata={
                "case": case,
                "mathdial_reference": md,
                "misstep_reference": ms,
                "difficulty_group": difficulty_label,
                "perturbations": _track1_perturbations(difficulty_label, concepts[:1]),
            },
        )


def _best_misstep(concepts: list[str], cases: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {item.lower() for item in concepts}
    best = cases[0] if cases else {}
    best_score = -1
    for case in cases:
        tokens = " ".join(case.get("concepts", []) + [case.get("misconception", ""), case.get("problem_text", "")]).lower()
        score = sum(1 for item in wanted if item and item in tokens)
        if score > best_score:
            best = case
            best_score = score
    return best


def _build_track2(root: Path, *, limit: int) -> Iterator[TaskInstance]:
    graph_path = root / "resource_graph.json"
    if not graph_path.exists():
        return
    graph = TextResourceGraph.from_json(graph_path)
    concepts = [node for node in graph.nodes.values() if node.type == "concept"]
    for idx, concept in enumerate(concepts[:limit]):
        task_type = _track2_task_type(idx)
        prereqs = graph.prerequisites_of(concept.node_id)
        graph_concepts = prereqs[:4] + [concept.node_id]
        resources = graph.resources_for_concepts(graph_concepts, limit=30)
        pool = [
            Resource(
                resource_id=node.node_id,
                type=node.type if node.type != "lecture_text" else "lecture_text",
                title=node.title,
                text=node.text,
                concepts=node.concepts or [concept.node_id],
                difficulty=_track2_resource_difficulty(node.concepts or [concept.node_id], concept.node_id, prereqs),
                metadata=node.metadata,
            )
            for node in resources
        ]
        if not pool:
            pool = [Resource(resource_id=f"concept_{concept.node_id}", type="explanation", text=concept.text, concepts=[concept.node_id])]
        if task_type == "Long-context Memory" and len(pool) < 15:
            pool.extend(_derived_context_resources(idx, concept, prereqs, graph, needed=15 - len(pool)))
        profile_mastery = {item: 0.35 for item in prereqs[:5]}
        profile_mastery[concept.node_id] = 0.2
        horizon = 42 if task_type in {"Adaptive Replan", "Retention Planning", "Long-context Memory"} else 30
        constraints = {
            "must_follow_prerequisites": True,
            "prerequisites": prereqs,
            "text_only": True,
            "task_type": task_type,
            "resource_budget": 8 if task_type == "Constraint Planning" else 12,
            "requires_retention": task_type == "Retention Planning",
            "adaptive": task_type == "Adaptive Replan",
        }
        yield TaskInstance(
            task_id=f"track2_{idx:06d}",
            track=TRACK2,
            domain="mooc",
            horizon=horizon,
            learner_profile=LearnerProfile(
                profile_text=f"Learner is preparing for concept {concept.title}.",
                estimated_mastery=profile_mastery,
                weak_concepts=[concept.node_id],
            ),
            goal=GoalSpec(target_concepts=[concept.node_id], target_mastery=0.75, horizon=horizon, description=concept.title),
            resource_pool=pool,
            constraints=constraints,
            metadata={
                "concept": to_plain(concept),
                "task_type": task_type,
                "perturbations": _track2_perturbations(task_type, concept.node_id, prereqs, pool),
            },
        )


def _build_track3(root: Path, *, limit: int) -> Iterator[TaskInstance]:
    path = _first_existing(root, "xes_sequences.jsonl.gz", "xes_sequences.jsonl")
    if not path.exists():
        return
    count = 0
    for seq in read_jsonl(path):
        questions = seq.get("questions", [])
        concepts = seq.get("concepts", [])
        responses = seq.get("responses", [])
        if not questions or not concepts:
            continue
        recent = list(zip(questions[:20], concepts[:20], responses[:20], strict=False))
        concept = next((item for item, response in zip(concepts, responses, strict=False) if response == "0"), concepts[-1])
        kc_name = seq.get("kc_names", {}).get(concept, concept)
        resources = []
        seen = set()
        for qid, cid, response in recent:
            if qid in seen:
                continue
            seen.add(qid)
            qmeta = seq.get("question_texts", {}).get(qid, {})
            resources.append(
                Resource(
                    resource_id=f"q_{qid}",
                    type="exercise",
                    title=f"Question {qid}",
                    text=qmeta.get("content", f"Question {qid}"),
                    concepts=[cid],
                    difficulty=0.35 + (0.25 if response == "0" else 0.0),
                    metadata={"question_id": qid, "response": response, "answer": qmeta.get("answer", [])},
                )
            )
        profile = LearnerProfile(
            profile_text=f"Chinese math learner with recent sequence history. Weak KC: {kc_name}.",
            estimated_mastery={cid: _estimate_mastery(cid, concepts, responses) for cid in set(concepts[:50])},
            weak_concepts=[concept],
            recent_errors=[{"concept": concept, "error_type": "incorrect response", "evidence": "recent sequence"}],
        )
        yield TaskInstance(
            task_id=f"track3_{count:06d}",
            track=TRACK3,
            domain="chinese_math",
            horizon=20,
            learner_profile=profile,
            goal=GoalSpec(target_concepts=[concept], target_mastery=0.75, horizon=20, description=kc_name),
            resource_pool=resources[:30],
            constraints={"text_only": True, "max_difficulty_jump": 0.25},
            metadata={
                "student_id": seq.get("student_id"),
                "source": seq.get("source"),
                "perturbations": _track3_perturbations(concept),
            },
        )
        count += 1
        if count >= limit:
            return


def _first_existing(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return root / names[0]


def _take_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for idx, row in enumerate(read_jsonl(path)):
        if idx >= limit:
            break
        out.append(row)
    return out


def _track1_difficulty_profile(idx: int) -> tuple[str, float, float, int]:
    profiles = (
        ("Easy", 0.34, 0.70, 10),
        ("Medium", 0.56, 0.75, 12),
        ("Hard", 0.78, 0.80, 14),
    )
    return profiles[idx % len(profiles)]


def _track1_perturbations(difficulty: str, targets: list[str]) -> list[dict[str, Any]]:
    if difficulty == "Easy":
        return []
    if difficulty == "Medium":
        return [{"step": 4, "type": "forced_error_streak", "severity": 0.12, "message": "student made two similar mistakes after initial remediation"}]
    return [
        {"step": 3, "type": "forced_error_streak", "severity": 0.16, "message": "student made repeated errors on the misconception pattern"},
        {"step": 7, "type": "forget_prerequisite", "target_concept": targets[0] if targets else None, "delta": 0.10},
    ]


def _track2_task_type(idx: int) -> str:
    types = ("Goal-to-Path", "Adaptive Replan", "Constraint Planning", "Long-context Memory", "Retention Planning")
    return types[idx % len(types)]


def _track2_resource_difficulty(concepts: list[str], target: str, prereqs: list[str]) -> float:
    concept_set = set(concepts)
    if target in concept_set:
        return 0.62
    if concept_set.intersection(prereqs):
        return 0.38
    return 0.50


def _derived_context_resources(idx: int, concept: Any, prereqs: list[str], graph: TextResourceGraph, *, needed: int) -> list[Resource]:
    resources: list[Resource] = []
    source_concepts = prereqs[:needed] or [concept.node_id]
    for j, cid in enumerate(source_concepts):
        node = graph.nodes.get(cid)
        title = node.title if node else cid
        text = node.text if node else str(title)
        resources.append(
            Resource(
                resource_id=f"derived_context_{idx}_{j}_{cid}",
                type="lecture_text",
                title=f"Concept context: {title}",
                text=text,
                concepts=[cid],
                difficulty=0.34 if cid in prereqs else 0.58,
                metadata={"derived_from_real_concept": cid},
            )
        )
    while len(resources) < needed:
        j = len(resources)
        resources.append(
            Resource(
                resource_id=f"derived_context_{idx}_{j}_{concept.node_id}",
                type="lecture_text",
                title=f"Long-context planning note {j + 1}",
                text=concept.text,
                concepts=[concept.node_id],
                difficulty=0.55,
                metadata={"derived_from_real_concept": concept.node_id},
            )
        )
    return resources


def _track2_perturbations(task_type: str, target: str, prereqs: list[str], pool: list[Resource]) -> list[dict[str, Any]]:
    if task_type == "Adaptive Replan":
        return [
            {"step": 5, "type": "forced_error_streak", "severity": 0.14, "message": "learner failed two planned resources and needs replanning"},
            {"step": 8, "type": "resource_unavailable", "resource_id": pool[-1].resource_id if pool else None},
        ]
    if task_type == "Retention Planning":
        concept = prereqs[0] if prereqs else target
        return [{"step": 10, "type": "forget_prerequisite", "target_concept": concept, "delta": 0.12}]
    if task_type == "Constraint Planning":
        return [{"step": 6, "type": "time_budget_reduction", "remaining_horizon": 18}]
    if task_type == "Long-context Memory":
        return [{"step": 12, "type": "forced_error_streak", "severity": 0.08, "message": "learner confused a prior context item with the target concept"}]
    return []


def _track3_perturbations(target: str) -> list[dict[str, Any]]:
    return [
        {"step": 6, "type": "forget_prerequisite", "target_concept": target, "delta": 0.08},
        {"step": 10, "type": "forced_error_streak", "severity": 0.10, "message": "learner shows a short error streak on the target KC"},
    ]


def _estimate_mastery(concept: str, concepts: list[str], responses: list[str]) -> float:
    vals = [int(resp) for cid, resp in zip(concepts, responses, strict=False) if cid == concept and resp in {"0", "1"}]
    if not vals:
        return 0.3
    return max(0.05, min(0.95, sum(vals[-10:]) / len(vals[-10:])))
