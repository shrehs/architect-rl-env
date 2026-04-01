import random
import secrets

from .models import Observation


ACTIONS = [
    "ASK_USE_CASE",
    "ASK_LATENCY",
    "ASK_ACCURACY",
    "ASK_DATA_SIZE",
    "ASK_UPDATE_FREQUENCY",
    "ASK_BUDGET",
    "FINALIZE",
]

IMPROVED_ACTION = "FINALIZE_WITH_COMPROMISE"


def random_agent_step() -> str:
    return random.choice(ACTIONS)


def heuristic_agent_step(observation: Observation) -> str:
    missing = [key.upper() for key in observation.missing_constraints]

    if not missing:
        return "FINALIZE"

    for key in ["USE_CASE", "LATENCY", "BUDGET"]:
        if key in missing:
            return f"ASK_{key}"

    return f"ASK_{missing[0]}"


def hard_conflict_detected(observation: Observation) -> bool:
    collected = observation.constraints_collected
    latency = collected.get("latency", "").lower()
    accuracy = collected.get("accuracy", "").lower()
    data_size = collected.get("data_size", "").lower()
    update_frequency = collected.get("update_frequency", "").lower()

    latency_hard = any(token in latency for token in ["real-time", "realtime", "ms"])
    accuracy_hard = any(token in accuracy for token in ["high", "near-perfect", "perfect", "99"])
    scale_hard = any(token in data_size for token in ["large", "very large", "tb"])
    freshness_hard = any(token in update_frequency for token in ["continuous", "stream", "hourly"])

    return latency_hard and accuracy_hard and scale_hard and freshness_hard


def improved_agent_step(observation: Observation) -> str:
    should_compromise = hard_conflict_detected(observation) or (
        observation.mode == "adversarial" and observation.step_count >= 6
    )
    if should_compromise:
        return IMPROVED_ACTION if secrets.randbelow(100) < 45 else "FINALIZE"
    return heuristic_agent_step(observation)


def choose_action(agent_name: str, observation: Observation) -> str:
    normalized = agent_name.lower().strip()
    if normalized == "random":
        return random_agent_step()
    if normalized == "heuristic":
        return heuristic_agent_step(observation)
    if normalized == "improved":
        return improved_agent_step(observation)
    raise ValueError(f"Unknown agent policy: {agent_name}")
