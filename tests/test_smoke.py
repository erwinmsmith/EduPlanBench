from __future__ import annotations

from eduplanbench.agents import create_agent
from eduplanbench.agents.external import external_agent_status
from eduplanbench.core.env import build_external_llm_env, get_llm_settings
from eduplanbench.core.io import write_jsonl
from eduplanbench.core.schema import GoalSpec, LearnerProfile, Resource, TaskInstance
from eduplanbench.data.task_builders import load_tasks
from eduplanbench.envs import EduPlanEnv
from eduplanbench.evaluation.metrics import evaluate_traces
from eduplanbench.evaluation.metrics import _metrics_for_trace
from eduplanbench.evaluation.tables import build_tables_from_experiment, _track1_difficulty, _track2_task_types
from eduplanbench.experiments import run_experiment_matrix
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


def test_enabled_external_agent_can_act(monkeypatch) -> None:
    monkeypatch.setenv("EDUPLAN_EXTERNAL_BRIDGE_USE_LLM", "0")
    task = make_task()
    env = EduPlanEnv(task, seed=1)
    agent = create_agent("external:lats")
    agent.reset(task)
    action = agent.act(env.reset())
    valid, error = action.validate_for(task.resource_pool)
    assert valid, error
    assert action.payload["external_agent"] == "lats"


def test_tables_include_external_agents(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EDUPLAN_EXTERNAL_BRIDGE_USE_LLM", "0")
    track_dir = tmp_path / "tasks" / "track3_kt_simulator"
    write_jsonl(track_dir / "tasks.jsonl", [make_task()])

    experiment_dir = run_experiment_matrix(
        tasks_dir=tmp_path / "tasks",
        outputs_dir=tmp_path / "runs",
        tracks=["track3_kt_simulator"],
        agents=["external:lats"],
        limit=1,
    )
    tables = build_tables_from_experiment(experiment_dir)
    rows = tables["Track3_Main"]

    assert [row["Agent"] for row in rows] == ["LATS"]
    assert rows[0]["Overall ↑"] is not None


def _trace_for_table(task: TaskInstance, *, final_mastery: float = 0.5) -> EpisodeTrace:
    trace = EpisodeTrace(task=task)
    observation = {
        "goal": {"target_concepts": task.goal.target_concepts},
        "estimated_mastery": task.learner_profile.estimated_mastery,
        "candidate_resources": [
            {
                "resource_id": resource.resource_id,
                "concepts": resource.concepts,
                "difficulty": resource.difficulty,
            }
            for resource in task.resource_pool
        ],
        "recent_feedback": ["incorrect; needs review"],
    }
    action = {
        "action_type": "recommend_exercise",
        "resource_id": task.resource_pool[0].resource_id,
        "target_concepts": task.goal.target_concepts,
        "rationale": "practice target concept",
        "payload": {},
    }
    trace.steps = [
        {
            "observation": observation,
            "action": action,
            "reward": 0.0,
            "done": True,
            "valid_action": True,
            "validation_error": "",
            "info": {"student_feedback": {"elapsed_time": 1.0}, "mastery_gain": 0.0},
        }
    ]
    trace.final_info = {"hidden": {"true_mastery": {task.goal.target_concepts[0]: final_mastery}, "dropout_risk": 0.0}}
    return trace


def test_subgroup_tables_report_counts_and_metrics() -> None:
    track1_rows = []
    for label in ("Easy", "Medium", "Hard"):
        task = make_task()
        task.track = "track1_text_math"
        task.metadata = {"difficulty_group": label}
        trace = _trace_for_table(task, final_mastery=0.5)
        track1_rows.append((trace, _metrics_for_trace(trace)))
    t1 = _track1_difficulty({("track1_text_math", "one_shot"): track1_rows})
    assert t1[0]["Easy N"] == 1
    assert t1[0]["Medium Overall ↑"] is not None
    assert t1[0]["Hard Track ↑"] is not None

    track2_rows = []
    for task_type in ("Goal-to-Path", "Adaptive Replan", "Constraint Planning", "Long-context Memory", "Retention Planning"):
        task = make_task()
        task.track = "track2_mooc_planning"
        task.metadata = {"task_type": task_type}
        task.constraints = {"task_type": task_type}
        trace = _trace_for_table(task, final_mastery=0.5)
        track2_rows.append((trace, _metrics_for_trace(trace)))
    t2 = _track2_task_types({("track2_mooc_planning", "one_shot"): track2_rows})
    assert t2[0]["Goal-to-Path N"] == 1
    assert t2[0]["Adaptive Replan GSR ↑"] is not None
    assert t2[0]["Retention Planning PR ↑"] is not None


def test_track3_track_score_is_clamped_non_negative() -> None:
    task = make_task()
    trace = _trace_for_table(task, final_mastery=0.25)
    trace.steps.append(dict(trace.steps[0]))
    trace.final_info = {"hidden": {"true_mastery": {"fractions": 0.25}, "dropout_risk": 1.0}}

    metrics = _metrics_for_trace(trace)

    assert metrics["track_score"] == 0.0
    assert metrics["overall_score"] >= 0.0


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
