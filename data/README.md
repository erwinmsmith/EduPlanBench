---
pretty_name: EduPlanBench Prepared Data
language:
  - en
task_categories:
  - question-answering
  - text-generation
  - reinforcement-learning
tags:
  - education
  - benchmark
  - planning
  - knowledge-tracing
  - tutoring
license: other
---

# EduPlanBench Prepared Data

This artifact contains the prepared EduPlanBench data used by the benchmark code. It is intended to be downloaded into the repository's `data/` directory so experiments can run without rebuilding from raw datasets.

## Contents

```text
processed/
  track1_text_math/
  track2_mooc_planning/
  track3_kt_simulator/
tasks/
  track1_text_math/
  track2_mooc_planning/
  track3_kt_simulator/
```

Current local build:

| Track | Task file | Tasks |
| --- | --- | ---: |
| `track1_text_math` | `tasks/track1_text_math/tasks.jsonl` | 10000 |
| `track2_mooc_planning` | `tasks/track2_mooc_planning/tasks.jsonl` | 10000 |
| `track3_kt_simulator` | `tasks/track3_kt_simulator/tasks.jsonl` | 10000 |

## Use In EduPlanBench

From the EduPlanBench repository root:

```bash
python3 scripts/download_prepared_data_from_hf.py \
  --repo-id erwinmsmith/EduPlanBench-data \
  --data-dir data
```

Then run experiments directly:

```bash
python3 -m eduplanbench experiment \
  --tracks all \
  --agents react,one_shot,step_by_step,cot \
  --llm deepseek \
  --limit 300 \
  --sample random \
  --sample-seed 42
```

## Source Data

The prepared files are derived from Eedi public data, MathDial, MisstepMath, MOOCCubeX, XES3G5M, and EdNet/KT1-style data. Raw datasets are not included here. Follow the upstream licenses and access rules for each original dataset. The repository file `rawdataset/README.md` documents the expected raw layout and source download locations for rebuilding from scratch.

## Rebuild From Raw Data

If all raw datasets are placed under `rawdataset/`:

```bash
python3 -m eduplanbench data prepare --track all
python3 -m eduplanbench build-tasks --track all --limit 10000
```
