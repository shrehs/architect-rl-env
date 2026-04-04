from unittest.mock import patch

from env.agents import ACTIONS, choose_action, heuristic_agent_step, improved_agent_step, random_agent_step
from env.models import Observation


def _obs(missing):
	return Observation(
		last_assistant_message="What would you like to ask next?",
		constraints_collected={},
		missing_constraints=missing,
		mode="clean",
		step_count=0,
	)


def test_random_agent_returns_valid_action() -> None:
	action = random_agent_step()
	assert action in ACTIONS


def test_heuristic_agent_finalizes_when_nothing_missing() -> None:
	action = heuristic_agent_step(_obs([]))
	assert action == "FINALIZE"


def test_heuristic_agent_prioritizes_use_case_then_latency() -> None:
	assert heuristic_agent_step(_obs(["latency", "use_case", "accuracy"])) == "ASK_USE_CASE"
	assert heuristic_agent_step(_obs(["latency", "accuracy"])) == "ASK_LATENCY"


def test_heuristic_agent_falls_back_to_first_missing_key() -> None:
	action = heuristic_agent_step(_obs(["data_size", "accuracy"]))
	assert action == "ASK_DATA_SIZE"


def test_improved_agent_finalizes_with_compromise_on_hard_conflict() -> None:
	observation = Observation(
		last_assistant_message="What would you like to ask next?",
		constraints_collected={
			"latency": "real-time",
			"accuracy": "near-perfect",
			"data_size": "very large",
			"update_frequency": "continuous",
		},
		missing_constraints=["use_case"],
		mode="clean",
		step_count=4,
	)

	with patch("env.agents.secrets.randbelow", return_value=0):
		assert improved_agent_step(observation) == "FINALIZE_WITH_COMPROMISE"


def test_choose_action_uses_requested_policy() -> None:
	observation = _obs(["use_case"])
	assert choose_action("heuristic", observation) == "ASK_USE_CASE"
	assert choose_action("random", observation) in ACTIONS


def test_choose_action_supports_improved_policy() -> None:
	observation = Observation(
		last_assistant_message="What would you like to ask next?",
		constraints_collected={
			"latency": "real-time",
			"accuracy": "near-perfect",
			"data_size": "very large",
			"update_frequency": "continuous",
		},
		missing_constraints=["use_case"],
		mode="clean",
		step_count=4,
	)

	with patch("env.agents.secrets.randbelow", return_value=0):
		assert choose_action("improved", observation) == "FINALIZE_WITH_COMPROMISE"
