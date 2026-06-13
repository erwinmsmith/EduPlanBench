# Raw Dataset Layout

`rawdataset/` is intentionally ignored by git. It should contain original downloaded datasets only when you want to rebuild `data/processed/` and `data/tasks/` from scratch. If you use the prepared Hugging Face artifact, you do not need to upload or copy this directory to the server.

## Expected Directory Tree

```text
rawdataset/
  public_data/
    train_data/*.csv
    metadata/*.csv
  mathdial/
  misstepmath/
  MOOCCubeX/
    entities/course.json
    entities/concept.json
    relations/concept-course.txt
    relations/concept-video.txt
    relations/concept-problem.txt
    prerequisites/math.json
    prerequisites/cs.json
    prerequisites/psy.json
  XES3G5M/
    kc_level/train_valid_sequences.csv
    question_level/*.csv
    metadata/questions.json
    metadata/kc_routes_map.json
  KT1/
    *.csv
```

## Source Links

| Dataset | Used for | Source |
| --- | --- | --- |
| Eedi NeurIPS 2020 Education Challenge public data | Track 1 student answer history | https://www.eedischool.com/projects/neurips-education-challenge and https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip |
| MathDial | Track 1 tutoring dialogue references | https://huggingface.co/datasets/eth-nlped/mathdial |
| MisstepMath | Track 1 misconception/remediation cases | https://huggingface.co/datasets/LLMEducation/MisstepMath |
| MOOCCubeX | Track 2 concept/resource graph | https://github.com/THU-KEG/MOOCCubeX |
| XES3G5M | Track 3 KT interaction sequences | https://github.com/ai4ed/XES3G5M |
| EdNet-KT1 | Track 3 KT1-style supplemental sequences | https://github.com/riiid/ednet and http://ednet-leaderboard.s3-website-ap-northeast-1.amazonaws.com/ |

## Automated Downloads

EduPlanBench can download the two Hugging Face Track 1 auxiliary datasets and the minimal Track 2 MOOCCubeX files:

```bash
python3 -m eduplanbench data fetch --track track1 --datasets mathdial,misstepmath
python3 -m eduplanbench data fetch --track track2
```

The MOOCCubeX downloader intentionally fetches only the minimal text/graph files needed by Track 2, not the large behavioral logs:

```text
entities/course.json
entities/concept.json
relations/concept-course.txt
relations/concept-video.txt
relations/concept-problem.txt
prerequisites/math.json
prerequisites/cs.json
prerequisites/psy.json
```

## Manual Data

Place these manually because they are large or governed by upstream access/licensing:

- `rawdataset/public_data/`: unzip Eedi NeurIPS 2020 data so `train_data/` and `metadata/` are direct children.
- `rawdataset/XES3G5M/`: download the official XES3G5M release and keep the published `kc_level/`, `question_level/`, and `metadata/` folders.
- `rawdataset/KT1/`: download EdNet-KT1 and place the per-student CSV files directly under this directory.

## Rebuild Prepared Data

After the raw layout is complete:

```bash
python3 -m eduplanbench data prepare --track all
python3 -m eduplanbench build-tasks --track all --limit 10000
```

Prepared outputs are written to:

```text
data/processed/
data/tasks/
```
