from env.environment import ArchitectEnv
from env.models import Action, Observation


def test_reset_returns_observation_only() -> None:
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    assert isinstance(obs, Observation)


def test_step_returns_exact_tuple_types() -> None:
    env = ArchitectEnv(task_id="easy")
    env.reset()
    result = env.step(Action(user_reply="Use case is recommendations with low latency 20ms"))

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
        "constraints",
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
    env.step(Action(user_reply="use case is search"))

    state_before = env.state()
    assert state_before["step_count"] > 0

    env.reset()
    state_after = env.state()

    assert state_after["step_count"] == 0
    assert state_after["constraints"] == {}
    assert state_after["done"] is False


def test_deterministic_same_sequence_same_outputs() -> None:
    actions = [
        "use case is risk scoring with real-time needs",
        "latency must be under 20ms",
        "accuracy target is 99.9%",
        "dataset is 2TB",
        "updates are hourly",
    ]

    env_a = ArchitectEnv(task_id="easy")
    env_b = ArchitectEnv(task_id="easy")

    env_a.reset()
    env_b.reset()

    outputs_a = []
    outputs_b = []

    for text in actions:
        out_a = env_a.step(Action(user_reply=text))
        out_b = env_b.step(Action(user_reply=text))
        outputs_a.append((out_a[0].model_dump(), out_a[1], out_a[2], out_a[3]))
        outputs_b.append((out_b[0].model_dump(), out_b[1], out_b[2], out_b[3]))

    assert outputs_a == outputs_b


def test_post_done_step_has_no_hidden_mutation() -> None:
    env = ArchitectEnv(task_id="easy", max_steps=1)
    env.reset()
    env.step(Action(user_reply="use case is analytics"))

    state_done = env.state()
    try:
        env.step(Action(user_reply="another input"))
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
    env1 = ArchitectEnv(task_id="easy")
    env2 = ArchitectEnv(task_id="easy")

    obs1 = env1.reset()
    obs2 = env2.reset()
    assert obs1 == obs2

    actions = ["a", "b", "c"]
    for action in actions:
        o1, r1, d1, _ = env1.step(Action(user_reply=action))
        o2, r2, d2, _ = env2.step(Action(user_reply=action))

        assert o1 == o2
        assert r1 == r2
        assert d1 == d2


def test_info_contains_progress_and_efficiency_metrics() -> None:
    env = ArchitectEnv(task_id="easy")
    env.reset()

    _, reward, done, info = env.step(Action(user_reply="random text"))

    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info["constraints_collected_count"], int)
    assert isinstance(info["progress"], float)
    assert isinstance(info["step_efficiency"], float)
    assert info["mode"] == "consultant"
    assert 0.0 <= info["progress"] <= 1.0
    assert 0.0 < info["step_efficiency"] <= 1.0


def test_random_spam_does_not_score_high() -> None:
    env = ArchitectEnv(task_id="easy")
    env.reset()

    total_reward = 0.0
    for _ in range(8):
        _, reward, done, _ = env.step(Action(user_reply="random text"))
        total_reward += reward
        if done:
            break

    assert total_reward < 1.5
