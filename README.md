# EduPlanBench

EduPlanBench is a Python benchmark for long-horizon agentic learning planning. It builds task instances from real education datasets, runs different planning agent systems in a closed-loop learning environment, records episode traces, and exports track-specific evaluation tables.

The current implementation supports three tracks:

- Track 1, `track1_text_math`: text-math misconception diagnosis, remediation, and replanning using local Eedi data plus MathDial and MisstepMath.
- Track 2, `track2_mooc_planning`: MOOC knowledge-path and resource planning using the MOOCCubeX concept/resource graph.
- Track 3, `track3_kt_simulator`: KT-based closed-loop planning using local XES3G5M / KT1-style interaction sequences.

Default LLM calls use a DeepSeek OpenAI-compatible endpoint. The LLM layer is intentionally isolated so other OpenAI-compatible providers can be swapped through environment variables.

## Repository Layout

```text
eduplanbench/
  agents/          Agent interface, LLM planners, and rule baselines
  core/            Schemas, config helpers, IO utilities
  data/            Downloaders, loaders, normalizers, task builders
  envs/            Gym-style EduPlanBench environment
  evaluation/      Metrics, tables, reports, robustness runs
  graphs/          Text resource graph and prerequisite utilities
  llm/             OpenAI-compatible client with DeepSeek defaults
  simulators/      Rule/BKT student simulator
scripts/
  build_experiment_tables_compact_xlsx.py
tests/
configs/
docs/
  external_agents.md     External agent bridge setup and evaluation guide
```

Generated data and outputs are intentionally ignored by git:

```text
rawdataset/                 raw downloaded/local datasets
data/processed/             normalized processed data
data/tasks/                 generated benchmark task instances
outputs/                    run traces, metrics, reports, workbooks
.env                        local API keys and private config
```

## Documentation

- [docs/external_agents.md](docs/external_agents.md): how to clone, bridge, enable, and evaluate external agent systems such as LLM+P, LATS, Plan-and-Act, ReAcTree, and HiAgent.

## Setup

Use Python 3.10 or newer. Conda is the recommended setup for server runs:

```bash
conda env create -f environment.yml
conda activate eduplanbench
```

If you change dependencies later:

```bash
conda env update -f environment.yml --prune
```

You can also use a plain virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or create a conda environment manually and install with pip:

```bash
conda create -n eduplanbench python=3.11 -y
conda activate eduplanbench
pip install -r requirements.txt
```

Create local environment config:

```bash
cp .env.example .env
```

Fill in `.env`:

```bash
EDUPLAN_LLM_PROVIDER=deepseek
EDUPLAN_LLM_API_KEY=your_deepseek_api_key_here
EDUPLAN_LLM_BASE_URL=https://api.deepseek.com
EDUPLAN_LLM_MODEL=deepseek-chat
```

`EDUPLAN_LLM_API_KEY` is the preferred single key variable for built-in LLM agents and external agent bridges. For compatibility, EduPlanBench also accepts `DEEPSEEK_API_KEY` and `OPENAI_API_KEY`.

When EduPlanBench launches a `stdio_json` external agent, it forwards the same LLM config as environment variables:

```text
EDUPLAN_LLM_API_KEY
EDUPLAN_LLM_BASE_URL
EDUPLAN_LLM_MODEL
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
```

This lets third-party agents that use OpenAI-compatible clients share one `.env` file. HTTP external agents should load the same `.env` in their own server process; API keys are not sent in HTTP request bodies. `.env` is ignored and should not be committed.

The default external bridge uses agent-specific EduPlanBench prompts when an LLM key is available. Control this with:

```bash
EDUPLAN_EXTERNAL_BRIDGE_USE_LLM=auto   # default: use LLM when a key exists
EDUPLAN_EXTERNAL_BRIDGE_USE_LLM=0      # deterministic bridge fallback, useful for smoke tests
```

## Data Preparation

Expected raw data roots:

```text
rawdataset/public_data/     local Eedi public data
rawdataset/mathdial/        downloaded MathDial
rawdataset/misstepmath/     downloaded MisstepMath
rawdataset/MOOCCubeX/       downloaded MOOCCubeX minimal files
rawdataset/XES3G5M/         local XES3G5M data
rawdataset/KT1/             local KT1/EdNet-style data
```

Eedi is assumed to already exist locally at:

```text
rawdataset/public_data
```

Download Track 1 auxiliary datasets:

```bash
python3 -m eduplanbench data fetch --track track1 --datasets mathdial,misstepmath
```

Download Track 2 MOOCCubeX minimal graph files:

```bash
python3 -m eduplanbench data fetch --track track2
```

Prepare all tracks:

```bash
python3 -m eduplanbench data prepare --track all
```

For a small development pass:

```bash
python3 -m eduplanbench data prepare --track all --limit 1000
```

Prepared data is written to:

```text
data/processed/track1_text_math/
data/processed/track2_mooc_planning/
data/processed/track3_kt_simulator/
```

## Build Tasks

Build task instances for every track:

```bash
python3 -m eduplanbench build-tasks --track all --limit 100
```

For a quick but table-complete experiment, use at least `--limit 5`. The first tasks are stratified so Track 1 covers Easy/Medium/Hard and Track 2 covers all five task types.

```bash
python3 -m eduplanbench build-tasks --track all --limit 15
```

Generated tasks are written to:

```text
data/tasks/{track}/tasks.jsonl
data/tasks/{track}/manifest.json
```

## Run A Single Benchmark

Run a non-LLM baseline on Track 3:

```bash
python3 -m eduplanbench run \
  --track track3_kt_simulator \
  --agent kt_recommender \
  --limit 5 \
  --sample random \
  --sample-seed 42
```

Run an LLM planner:

```bash
python3 -m eduplanbench run \
  --track track1_text_math \
  --agent react \
  --llm deepseek \
  --limit 3 \
  --sample random \
  --sample-seed 42
```

Supported LLM planner systems:

```text
one_shot
react
step_by_step
cot
```

Supported non-LLM baselines:

```text
kt_recommender
static_one_shot
random
prerequisite_rule
difficulty_rule
oracle
oracle_prerequisite
oracle_simulator
```

External agent systems are exposed through JSON bridges instead of being copied into EduPlanBench. Registered external agent ids:

```text
llm_pddl
lats
plan_and_act
reactree
hiagent
```

Clone external repositories:

```bash
python3 scripts/clone_external_agents.py
```

Check availability:

```bash
python3 -m eduplanbench agents list
```

The five registered agents are enabled through `scripts/external_agent_bridge.py`, a persistent EduPlanBench JSONL bridge that maps benchmark observations into each agent system's planning style, uses agent-specific prompts when an LLM key is available, and returns strict EduPlanBench `Action` JSON. See [docs/external_agents.md](docs/external_agents.md).

To evaluate an external agent:

1. Clone the registered repos:

```bash
python3 scripts/clone_external_agents.py
```

2. Check that the bridge is enabled:

```bash
python3 -m eduplanbench agents list
```

3. Run a small smoke test on one track:

```bash
python3 -m eduplanbench run \
  --track track3_kt_simulator \
  --agent external:lats \
  --llm deepseek \
  --limit 3 \
  --sample random \
  --sample-seed 42
```

4. Run a pilot matrix with the external agent beside built-in baselines:

```bash
python3 -m eduplanbench experiment \
  --tracks all \
  --agents react,one_shot,external:lats \
  --llm deepseek \
  --limit 30 \
  --sample random \
  --sample-seed 42
```

5. Run robustness on the same agent set:

```bash
python3 -m eduplanbench robustness \
  --experiment-dir outputs/runs/experiment-YYYYMMDD-HHMMSS \
  --agents react,external:lats \
  --limit 10 \
  --llm deepseek \
  --sample random \
  --sample-seed 42
```

Each run writes:

```text
episodes.jsonl.gz          full step-by-step traces
metrics.json               aggregate metrics
metrics.csv                one-row metric table
report.md                  readable report
config.snapshot.json       reproducibility snapshot
```

## Run The Agent-System Matrix

Run the default four LLM systems across all three tracks:

```bash
python3 -m eduplanbench experiment \
  --tracks all \
  --agents react,one_shot,step_by_step,cot \
  --llm deepseek \
  --limit 5 \
  --sample random \
  --sample-seed 42
```

`--sample random` draws a fixed random subset from each track's task bank. Use `--sample-seed` to make the sampled task ids reproducible. Use `--sample first` only when you explicitly want the first N tasks in file order.

For paper experiments, the convenience bundle runs the main matrix, robustness, table generation, and compact Excel export in one command:

```bash
python3 scripts/run_experiment_bundle.py \
  --main-limit 300 \
  --robust-limit 50 \
  --robust-agents one_shot,react \
  --agents react,one_shot,step_by_step,cot \
  --llm deepseek \
  --sample random \
  --sample-seed 42
```

This keeps the task sampling reproducible while allowing the main and robustness sample sizes to differ. `--robust-limit` should usually be smaller than `--main-limit` because robustness runs four horizons and includes H=100.

To include an enabled external agent in the bundle:

```bash
python3 scripts/run_experiment_bundle.py \
  --main-limit 300 \
  --robust-limit 50 \
  --agents react,one_shot,step_by_step,cot,external:lats \
  --robust-agents react,external:lats \
  --llm deepseek \
  --sample random \
  --sample-seed 42
```

Use `--robust-agents same` only when you intentionally want robustness for every agent in `--agents`.

The experiment directory is printed at the end, for example:

```text
outputs/runs/experiment-YYYYMMDD-HHMMSS
```

The matrix writes:

```text
matrix_results.csv
matrix_results.json
matrix_report.md
```

## Robustness Runs

Robustness evaluates horizon sensitivity at H=10, H=30, H=50, and H=100. By default it runs one-shot and ReAct; pass `--agents` to include external systems:

```bash
python3 -m eduplanbench robustness \
  --experiment-dir outputs/runs/experiment-YYYYMMDD-HHMMSS \
  --agents one_shot,react \
  --limit 1 \
  --llm deepseek \
  --sample random \
  --sample-seed 42
```

This writes:

```text
outputs/runs/experiment-YYYYMMDD-HHMMSS/robustness/
```

## Generate Evaluation Tables

Build CSV/JSON tables for the experiment:

```bash
python3 -m eduplanbench tables \
  --experiment-dir outputs/runs/experiment-YYYYMMDD-HHMMSS
```

Tables are written to:

```text
outputs/runs/experiment-YYYYMMDD-HHMMSS/tables/
```

The table set contains:

```text
Track1_Main
Track1_Specific
Track1_Difficulty
Track2_Main
Track2_PathQuality
Track2_TaskTypes
Track3_Main
Track3_ClosedLoop
Track3_ActionDiag
Robustness
```

## Export Compact Excel

The compact Excel exporter is pure Python and uses `openpyxl`.

```bash
python3 scripts/build_experiment_tables_compact_xlsx.py \
  --experiment-dir outputs/runs/experiment-YYYYMMDD-HHMMSS \
  --output outputs/workbooks/EduPlanBench_Experiment_Tables_compact.xlsx
```

The workbook uses short column names and compact widths. It is intended for quick reading and manual inspection.

## Metrics

Core metrics:

- `GSR`: goal success rate.
- `PR`: progress rate toward target mastery.
- `Steps`: episode length.
- `Valid`: valid action rate.
- `CtxTok`: tokenizer-free context-size proxy, computed as ASCII characters / 4 plus non-ASCII characters.
- `Time`: simulated learner time cost.
- `Core`, `Track`, `Overall`: aggregate scores.

Track-specific metrics include:

- Track 1: misconception accuracy, feedback grounding, remediation match, hint helpfulness, error-aware replanning, redundancy, direct-answer rate.
- Track 2: prerequisite violation, sequence consistency, resource-concept match, path coherence, constraint satisfaction, difficulty alignment, plan drift.
- Track 3: mastery gain, retention gain, learning efficiency, dropout risk, overload, recovery, simulator exploitation.

## Long-Horizon Design

The environment exposes only observable learner state:

```text
learner summary
visible mastery estimate
recent feedback
candidate resources
goal
available actions
current plan
```

Hidden simulator state, true mastery, dropout risk internals, and future trajectory are available only to evaluation traces.

Tasks include dynamic events that require replanning:

```text
forced error streaks
forgotten prerequisite / target KC
resource unavailable
time budget reduction
```

This is why one-shot and stepwise systems can diverge even on the same task.

## Testing

Run tests with:

```bash
PYTHONPATH=. pytest -q
```

Run a syntax check:

```bash
python3 -m compileall eduplanbench
```

## Notes For Pushing

Do commit:

```text
eduplanbench/
scripts/
tests/
configs/
README.md
.env.example
.gitignore
pyproject.toml
```

Do not commit:

```text
.env
rawdataset/
data/processed/
data/tasks/
outputs/
__pycache__/
.pytest_cache/
*.pyc
```

Large datasets and run artifacts should be regenerated locally with the commands above.
