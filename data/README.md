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

Prepared data is available in two access patterns:

- Use the Hugging Face branch/revision dropdown to switch between `10k` and `35k`.
- Stay on `main` and download from `versions/10k/` or `versions/35k/`.

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

Prepared versions:

| Version | Build command | Notes |
| --- | --- | --- |
| `10k` | `build-tasks --track all --limit 10000` | Track 1/2/3 each have 10000 task instances. |
| `35k` | `build-tasks --track all --limit 35000` | Track 1 has 35000, Track 2 has 35000, Track 3 has 33397 XES3G5M-derived task instances. |

Version counts:

| Version | Track 1 | Track 2 | Track 3 |
| --- | ---: | ---: | ---: |
| `10k` | 10000 | 10000 | 10000 |
| `35k` | 35000 | 35000 | 33397 |

## Use In EduPlanBench

From the EduPlanBench repository root:

```bash
python3 scripts/download_prepared_data_from_hf.py \
  --repo-id erwinmsmith/EduPlanBench-data \
  --version 10k \
  --data-dir data
```

Use `--version 35k` to download the larger prepared task bank.

To download from a Hugging Face branch/revision root instead:

```bash
python3 scripts/download_prepared_data_from_hf.py \
  --repo-id erwinmsmith/EduPlanBench-data \
  --revision 35k \
  --path-in-repo . \
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
