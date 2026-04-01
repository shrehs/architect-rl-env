from unittest.mock import patch

from env.environment import ArchitectEnv
from env.models import Action, Observation


def test_reset_returns_observation_only() -> None:
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    assert isinstance(obs, Observation)


def test_step_returns_exact_tuple_types() -> None:
    env = ArchitectEnv(task_id="easy")
    env.reset()
    result = env.step(Action(type="ASK_USE_CASE"))

    assert isinstance(result, tuple)
    assert len(result) == 4
    obs, reward, done, info = result

    assert isinstance(obs, Observation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_state_returns_full_internal_dict() -> None:
    env = ArchitectEnv(task_id="easy")
    env.reset()
    state = env.state()

    for key in [
        "messages",
        "observed_constraints",
        "mode",
        "step_count",
        "done",
        "task_id",
        "last_assistant_message",
    ]:
        assert key in state


def test_reset_clears_state() -> None:
    env = ArchitectEnv(task_id="easy")
    env.reset()
    env.step(Action(type="ASK_USE_CASE"))

    state_before = env.state()
    assert state_before["step_count"] > 0

    env.reset()
    state_after = env.state()

    assert state_after["step_count"] == 0
    assert state_after["observed_constraints"] == {}
    assert state_after["done"] is False


def test_deterministic_same_sequence_same_outputs() -> None:
    actions = [
        "ASK_USE_CASE",
        "ASK_LATENCY",
        "ASK_ACCURACY",
        "ASK_DATA_SIZE",
        "ASK_UPDATE_FREQUENCY",
    ]

    with patch("env.environment.random.choice", return_value="clean"):
        env_a = ArchitectEnv(task_id="easy")
        env_b = ArchitectEnv(task_id="easy")

        env_a.reset()
        env_b.reset()

        outputs_a = []
        outputs_b = []

        for action_type in actions:
            out_a = env_a.step(Action(type=action_type))
            out_b = env_b.step(Action(type=action_type))
            outputs_a.append((out_a[0].model_dump(), out_a[1], out_a[2], out_a[3]))
            outputs_b.append((out_b[0].model_dump(), out_b[1], out_b[2], out_b[3]))

        assert outputs_a == outputs_b


def test_post_done_step_has_no_hidden_mutation() -> None:
    env = ArchitectEnv(task_id="easy", max_steps=1)
    env.reset()
    env.step(Action(type="ASK_USE_CASE"))

    state_done = env.state()
    try:
        env.step(Action(type="ASK_LATENCY"))
        assert False, "Expected RuntimeError when stepping after episode completion"
    except RuntimeError as exc:
        assert str(exc) == "Episode already finished. Call reset()."

    state_after = env.state()
    assert state_done == state_after


def test_observation_does_not_leak_hidden_state() -> None:
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    payload = obs.model_dump()

    assert set(payload.keys()) == {
        "last_assistant_message",
        "constraints_collected",
        "missing_constraints",
        "mode",
        "step_count",
    }
    assert "done" not in payload
    assert "task_id" not in payload


def test_determinism_evaluator_style() -> None:
    with patch("env.environment.random.choice", return_value="clean"):
        env1 = ArchitectEnv(task_id="easy")
        env2 = ArchitectEnv(task_id="easy")

        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1 == obs2

        actions = ["ASK_USE_CASE", "ASK_LATENCY", "ASK_ACCURACY"]
        for action in actions:
            o1, r1, d1, _ = env1.step(Action(type=action))
            o2, r2, d2, _ = env2.step(Action(type=action))

            assert o1 == o2
            assert r1 == r2
            assert d1 == d2


def test_info_contains_progress_and_efficiency_metrics() -> None:
    with patch("env.environment.random.choice", return_value="clean"):
        env = ArchitectEnv(task_id="easy")
        env.reset()

        _, reward, done, info = env.step(Action(type="ASK_BUDGET"))

        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info["constraints_collected_count"], int)
        assert isinstance(info["progress"], float)
        assert isinstance(info["step_efficiency"], float)
        assert info["mode"] in {"clean", "noisy", "adversarial"}
        assert 0.0 <= info["progress"] <= 1.0
        assert 0.0 < info["step_efficiency"] <= 1.0


def test_finalize_ends_episode_before_max_steps() -> None:
    with patch("env.environment.random.choice", return_value="clean"):
        env = ArchitectEnv(task_id="easy", max_steps=8)
        env.reset()

        _, _, done, info = env.step(Action(type="FINALIZE"))

        assert done is True
        assert env.state()["done"] is True
        assert 0.0 <= info["oracle_score"] <= 1.0
        assert info["missing_penalty"] >= 0.0


def test_finalize_with_compromise_ends_episode() -> None:
    with patch("env.environment.random.choice", return_value="clean"):
        env = ArchitectEnv(task_id="hard", max_steps=8)
        env.reset()

        _, _, done, info = env.step(Action(type="FINALIZE_WITH_COMPROMISE"))

        assert done is True
        assert env.state()["done"] is True
        assert info.get("compromise_detected", False) in {True, False}


def test_oracle_returns_structured_output() -> None:
    from env.oracle import oracle_recommend

    oracle = oracle_recommend(
        {
            "use_case": "recommendation ranking",
            "latency": "real-time",
            "accuracy": "high",
            "data_size": "moderate",
            "update_frequency": "hourly",
            "budget": "low",
        }
    )

    assert set(oracle.keys()) == {"model", "deployment", "architecture", "reasoning"}
    assert isinstance(oracle["reasoning"], list)
    assert isinstance(oracle["model"], str)
    assert isinstance(oracle["deployment"], str)
    assert isinstance(oracle["architecture"], str)


def test_oracle_hard_task_returns_compromise() -> None:
    from env.oracle import oracle_recommend

    oracle = oracle_recommend(
        {
            "use_case": "multimodal assistant",
            "latency": "real-time",
            "accuracy": "near-perfect",
            "data_size": "very large",
            "update_frequency": "continuous",
            "budget": "low",
        }
    )

    assert oracle["model"] == "small_transformer"
    assert oracle["deployment"] == "edge + batch hybrid"
    assert oracle["architecture"] == "cost-optimized streaming compromise"


def test_random_spam_does_not_score_high() -> None:
    with patch("env.environment.random.choice", return_value="clean"):
        env = ArchitectEnv(task_id="easy")
        env.reset()

        total_reward = 0.0
        for _ in range(8):
            _, reward, done, _ = env.step(Action(type="ASK_BUDGET"))
            total_reward += reward
            if done:
                break

        assert total_reward < 1.5
