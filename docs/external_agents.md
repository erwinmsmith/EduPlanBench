# External Agent Integration

EduPlanBench can call third-party agent systems through a JSON bridge. It sends EduPlanBench task/observation state to an external process or HTTP endpoint and expects one EduPlanBench `Action` JSON object in return.

The five registered repositories target different native environments, such as PDDL domains, HotPotQA/WebShop, WebArena, VirtualHome/ALFRED, and AgentBoard. EduPlanBench therefore provides a default bridge at:

```text
scripts/external_agent_bridge.py
```

This bridge makes the agents usable inside EduPlanBench by mapping EduPlanBench observations into each system's planning style. It does not launch the original repositories' native benchmark environments. If you want to run a repo's native runtime directly, replace the command in `configs/external_agents.json` with your own bridge.

## Cloned Repositories

Use:

```bash
python3 scripts/clone_external_agents.py
```

This clones the registered repositories into:

```text
third_party/agents/
```

This directory is ignored by git. Do not commit third-party repositories into EduPlanBench.

Registered external agents:

```text
llm_pddl       https://github.com/Cranial-XIX/llm-pddl.git
lats           https://github.com/lapisrocks/LanguageAgentTreeSearch.git
plan_and_act   https://github.com/SqueezeAILab/plan-and-act.git
reactree       https://github.com/Choi-JaeWoo/ReAcTree.git
hiagent        https://github.com/HiAgent2024/HiAgent.git
```

Check status:

```bash
python3 -m eduplanbench agents list
```

## Unified LLM Environment

Use one `.env` file for built-in LLM agents and external bridges:

```bash
EDUPLAN_LLM_PROVIDER=deepseek
EDUPLAN_LLM_API_KEY=your_deepseek_api_key_here
EDUPLAN_LLM_BASE_URL=https://api.deepseek.com
EDUPLAN_LLM_MODEL=deepseek-chat
```

EduPlanBench also accepts `DEEPSEEK_API_KEY` and `OPENAI_API_KEY`, but `EDUPLAN_LLM_API_KEY` is preferred.

For `stdio_json` and `stdio_jsonl` bridges, EduPlanBench launches the bridge process with both EduPlanBench names and OpenAI-compatible aliases:

```text
EDUPLAN_LLM_API_KEY
EDUPLAN_LLM_BASE_URL
EDUPLAN_LLM_MODEL
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
```

The JSON request also includes non-secret LLM config:

```json
{
  "llm": {
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
  }
}
```

For `http_json` bridges, start the external server with the same `.env`. EduPlanBench does not send API keys in HTTP request bodies.

## Bridge Protocol

Config file:

```text
configs/external_agents.json
```

Each agent can use either:

```text
stdio_json
stdio_jsonl
http_json
```

### stdio_json

EduPlanBench runs the configured command and writes one JSON request to stdin. The bridge writes one JSON response to stdout.

### stdio_jsonl

EduPlanBench starts the configured command once per episode and exchanges one JSON request/response per line. This is the default for the five registered external agents because it avoids spawning a new process at every environment step.

Request shape for `act`:

```json
{
  "event": "act",
  "agent": "lats",
  "llm": {
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
  },
  "task": {},
  "observation": {},
  "action_schema": {}
}
```

Response shape:

```json
{
  "action_type": "recommend_exercise",
  "resource_id": "q_123",
  "target_concepts": ["fractions"],
  "rationale": "The learner is weak on fractions and this resource is aligned.",
  "plan_update": "",
  "payload": {}
}
```

For `reset` and `reflect`, bridges may return `{}`.

### http_json

Set:

```json
{
  "protocol": "http_json",
  "endpoint": "http://127.0.0.1:8765/act"
}
```

The endpoint receives the same JSON request and returns the same action JSON.

## Enabling An External Agent

1. Clone external repos:

```bash
python3 scripts/clone_external_agents.py
```

2. Use the default bridge or add a custom bridge script inside the external repo or elsewhere.

You can start from:

```text
scripts/external_agent_bridge_template.py
```

3. Edit `configs/external_agents.json`.

Example:

```json
{
  "enabled": true,
  "protocol": "stdio_jsonl",
  "command": ["{python}", "{repo_root}/scripts/external_agent_bridge.py", "--agent", "lats"],
  "cwd": "{repo_path}"
}
```

4. Run:

```bash
python3 -m eduplanbench run \
  --track track3_kt_simulator \
  --agent external:lats \
  --limit 5 \
  --sample random \
  --sample-seed 42
```

You can also use the short name if it is registered:

```bash
python3 -m eduplanbench run --track track3_kt_simulator --agent lats --limit 5
```

## Adding A New External Agent

Add one entry to `configs/external_agents.json`:

```json
{
  "new_agent": {
    "display_name": "New Agent",
    "repo_url": "https://github.com/org/new-agent.git",
    "repo_path": "third_party/agents/new-agent",
    "enabled": false,
    "protocol": "stdio_json",
    "command": ["python", "bridge.py"],
    "cwd": "{repo_path}",
    "timeout_seconds": 300,
    "notes": "Explain setup here."
  }
}
```

Then implement a bridge that returns EduPlanBench `Action` JSON.

## Evaluation Recipe

Use the same benchmark commands for external agents after the bridge is enabled.

Smoke test one track:

```bash
python3 -m eduplanbench run \
  --track track3_kt_simulator \
  --agent external:lats \
  --llm deepseek \
  --limit 3 \
  --sample random \
  --sample-seed 42
```

Pilot matrix across all tracks:

```bash
python3 -m eduplanbench experiment \
  --tracks all \
  --agents react,one_shot,external:lats \
  --llm deepseek \
  --limit 30 \
  --sample random \
  --sample-seed 42
```

Robustness table for horizon sensitivity:

```bash
python3 -m eduplanbench robustness \
  --experiment-dir outputs/runs/experiment-YYYYMMDD-HHMMSS \
  --agents react,external:lats \
  --limit 10 \
  --llm deepseek \
  --sample random \
  --sample-seed 42
```

Full bundle with compact Excel:

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

Use `--robust-agents same` only if you want robustness runs for every agent in `--agents`.

## Important Notes

Several registered agents are environment-specific:

- LLM+P expects PDDL domains and Fast Downward.
- LATS has HotPotQA/programming/WebShop entrypoints.
- Plan-and-Act targets WebArena.
- ReAcTree targets VirtualHome/ALFRED and has simulator dependencies.
- HiAgent depends on AgentBoard.

For these repos, the bridge is responsible for translating EduPlanBench observations into whatever prompt/state the external agent expects, and translating the external agent result back into an EduPlanBench action.
