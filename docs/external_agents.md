# External Agent Integration

EduPlanBench can call third-party agent systems through a JSON bridge. The benchmark does not reimplement external algorithms. It sends EduPlanBench task/observation state to an external process or HTTP endpoint and expects one EduPlanBench `Action` JSON object in return.

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

## Bridge Protocol

Config file:

```text
configs/external_agents.json
```

Each agent can use either:

```text
stdio_json
http_json
```

### stdio_json

EduPlanBench runs the configured command and writes one JSON request to stdin. The bridge writes one JSON response to stdout.

Request shape for `act`:

```json
{
  "event": "act",
  "agent": "lats",
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

2. Add a bridge script inside the external repo or elsewhere.

You can start from:

```text
scripts/external_agent_bridge_template.py
```

3. Edit `configs/external_agents.json`.

Example:

```json
{
  "enabled": true,
  "protocol": "stdio_json",
  "command": ["python", "bridge.py"],
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

## Important Notes

Several registered agents are environment-specific:

- LLM+P expects PDDL domains and Fast Downward.
- LATS has HotPotQA/programming/WebShop entrypoints.
- Plan-and-Act targets WebArena.
- ReAcTree targets VirtualHome/ALFRED and has simulator dependencies.
- HiAgent depends on AgentBoard.

For these repos, the bridge is responsible for translating EduPlanBench observations into whatever prompt/state the external agent expects, and translating the external agent result back into an EduPlanBench action.
