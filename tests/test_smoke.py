from __future__ import annotations

from eduplanbench.agents import create_agent
from eduplanbench.agents.external import external_agent_status
from eduplanbench.core.env import build_external_llm_env, get_llm_settings
from eduplanbench.core.io import write_jsonl
from eduplanbench.core.schema import GoalSpec, LearnerProfile, Resource, TaskInstance
from eduplanbench.data.task_builders import load_tasks
from eduplanbench.envs import EduPlanEnv
from eduplanbench.evaluation.metrics import evaluate_traces
from eduplanbench.core.schema import EpisodeTrace
from eduplanbench.runner import run_benchmark


def make_task() -> TaskInstance:
    return TaskInstance(
        task_id="tiny",
        track="track3_kt_simulator",
        domain="math",
        horizon=5,
        learner_profile=LearnerProfile(
            profile_text="Tiny learner",
            estimated_mastery={"fractions": 0.25},
            weak_concepts=["fractions"],
        ),
        goal=GoalSpec(target_concepts=["fractions"], target_mastery=0.45, horizon=5),
        resource_pool=[
            Resource(
                resource_id="r1",
                type="exercise",
                text="1/2 + 1/3",
                concepts=["fractions"],
                difficulty=0.3,
            )
        ],
    )


def test_env_agent_episode_smoke() -> None:
    task = make_task()
    env = EduPlanEnv(task, seed=1)
    agent = create_agent("kt_recommender")
    obs = env.reset()
    trace = EpisodeTrace(task=task)
    done = False
    while not done:
        action = agent.act(obs)
        valid, error = action.validate_for(task.resource_pool)
        result = env.step(action)
        trace.add_step(obs, action, result, valid, error)
        obs = result.observation
        done = result.done
    trace.final_info = trace.steps[-1]["info"]
    report = evaluate_traces([trace], run_id="test")
    assert report.metadata["episodes"] == 1
    assert report.metrics["valid_action_rate"] == 1.0


def test_invalid_action_is_detected() -> None:
    task = make_task()
    env = EduPlanEnv(task, seed=1)
    env.reset()
    result = env.step(create_agent("kt_recommender").act(env._observation()))
    assert "hidden" in result.info


def test_llm_flag_does_not_convert_rule_agent() -> None:
    agent = create_agent("random", llm="deepseek")
    assert agent.system_name == "random"


def test_random_task_sampling_is_seeded(tmp_path) -> None:
    track_dir = tmp_path / "track3_kt_simulator"
    rows = []
    for idx in range(10):
        task = make_task()
        task.task_id = f"task_{idx}"
        rows.append(task)
    write_jsonl(track_dir / "tasks.jsonl", rows)

    first = [task.task_id for task in load_tasks(tmp_path, "track3_kt_simulator", limit=4, sample="random", seed=7)]
    second = [task.task_id for task in load_tasks(tmp_path, "track3_kt_simulator", limit=4, sample="random", seed=7)]
    third = [task.task_id for task in load_tasks(tmp_path, "track3_kt_simulator", limit=4, sample="random", seed=8)]

    assert first == second
    assert first != third
    assert first != [f"task_{idx}" for idx in range(4)]


def test_run_benchmark_uses_unique_run_dirs(tmp_path) -> None:
    track_dir = tmp_path / "tasks" / "track3_kt_simulator"
    write_jsonl(track_dir / "tasks.jsonl", [make_task()])
    out = tmp_path / "runs"

    first = run_benchmark(
        tasks_dir=tmp_path / "tasks",
        outputs_dir=out,
        track="track3_kt_simulator",
        agent_name="random",
        limit=1,
    )
    second = run_benchmark(
        tasks_dir=tmp_path / "tasks",
        outputs_dir=out,
        track="track3_kt_simulator",
        agent_name="random",
        limit=1,
    )

    assert first != second
    assert (first / "episodes.jsonl.gz").exists()
    assert (second / "episodes.jsonl.gz").exists()


def test_external_agent_registry_is_available() -> None:
    rows = external_agent_status()
    names = {row["name"] for row in rows}
    assert {"llm_pddl", "lats", "plan_and_act", "reactree", "hiagent"}.issubset(names)


def test_enabled_external_agent_can_act() -> None:
    task = make_task()
    env = EduPlanEnv(task, seed=1)
    agent = create_agent("external:lats")
    agent.reset(task)
    action = agent.act(env.reset())
    valid, error = action.validate_for(task.resource_pool)
    assert valid, error
    assert action.payload["external_agent"] == "lats"


def test_unified_llm_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("EDUPLAN_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("EDUPLAN_LLM_API_KEY", "test-key")
    monkeypatch.setenv("EDUPLAN_LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("EDUPLAN_LLM_MODEL", "test-model")

    settings = get_llm_settings()
    env = build_external_llm_env({})

    assert settings["api_key"] == "test-key"
    assert env["EDUPLAN_LLM_API_KEY"] == "test-key"
    assert env["OPENAI_API_KEY"] == "test-key"
    assert env["OPENAI_BASE_URL"] == "https://example.test/v1"
    assert env["OPENAI_MODEL"] == "test-model"
