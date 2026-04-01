from typing import Any, Dict, List


def _normalized_value(hidden_constraints: Dict[str, str], key: str) -> str:
    value = hidden_constraints.get(key, "")
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())


def _has_any(text: str, tokens: List[str]) -> bool:
    return any(token in text for token in tokens)


def _is_compromise_scenario(hidden_constraints: Dict[str, str]) -> bool:
    latency = _normalized_value(hidden_constraints, "latency")
    budget = _normalized_value(hidden_constraints, "budget")
    accuracy = _normalized_value(hidden_constraints, "accuracy")
    data_size = _normalized_value(hidden_constraints, "data_size")
    update_frequency = _normalized_value(hidden_constraints, "update_frequency")

    latency_hard = _has_any(latency, ["real-time", "realtime", "real_time", "under 20ms", "ms"])
    budget_hard = _has_any(budget, ["low", "limited"])
    accuracy_hard = _has_any(accuracy, ["high", "near-perfect", "perfect", "99%", "99.9", "99.99"])
    scale_hard = _has_any(data_size, ["large", "very large", "very_large", "tb"])
    freshness_hard = _has_any(update_frequency, ["continuous", "stream", "streaming"])

    # This captures infeasible targets where latency/cost/accuracy/scale conflict.
    return latency_hard and budget_hard and accuracy_hard and scale_hard and freshness_hard


def select_model(hidden_constraints: Dict[str, str]) -> str:
    latency = _normalized_value(hidden_constraints, "latency")
    budget = _normalized_value(hidden_constraints, "budget")
    accuracy = _normalized_value(hidden_constraints, "accuracy")
    data_size = _normalized_value(hidden_constraints, "data_size")

    if _has_any(latency, ["real-time", "realtime", "low latency", "under 20ms", "ms"]) and _has_any(
        budget, ["low", "limited"]
    ):
        return "small_cnn"

    if _has_any(accuracy, ["high", "near-perfect", "perfect", "99%", "99.9", "99.99"]):
        return "transformer"

    if _has_any(data_size, ["large", "very large", "tb"]):
        return "hybrid"

    return "hybrid"


def select_deployment(hidden_constraints: Dict[str, str]) -> str:
    latency = _normalized_value(hidden_constraints, "latency")
    budget = _normalized_value(hidden_constraints, "budget")
    update_frequency = _normalized_value(hidden_constraints, "update_frequency")
    data_size = _normalized_value(hidden_constraints, "data_size")

    if _has_any(latency, ["real-time", "realtime", "low latency", "under 20ms", "ms"]) and _has_any(
        budget, ["low", "limited"]
    ):
        return "edge_optimized"

    if _has_any(update_frequency, ["stream", "continuous", "hourly"]) or _has_any(latency, ["real-time", "realtime"]):
        return "streaming_service"

    if _has_any(data_size, ["large", "very large", "tb"]):
        return "batch_pipeline"

    return "standard_cloud"


def select_architecture(hidden_constraints: Dict[str, str]) -> str:
    latency = _normalized_value(hidden_constraints, "latency")
    accuracy = _normalized_value(hidden_constraints, "accuracy")
    data_size = _normalized_value(hidden_constraints, "data_size")
    budget = _normalized_value(hidden_constraints, "budget")

    if _has_any(latency, ["real-time", "realtime", "low latency"]) and _has_any(budget, ["low", "limited"]):
        return "event_driven_microservices"

    if _has_any(accuracy, ["high", "near-perfect", "perfect", "99%", "99.9", "99.99"]) and _has_any(
        data_size, ["large", "very large", "tb"]
    ):
        return "hybrid_lakehouse"

    return "service_oriented"


def derive_tradeoffs(hidden_constraints: Dict[str, str]) -> List[str]:
    tradeoffs: List[str] = []
    latency = _normalized_value(hidden_constraints, "latency")
    accuracy = _normalized_value(hidden_constraints, "accuracy")
    budget = _normalized_value(hidden_constraints, "budget")
    data_size = _normalized_value(hidden_constraints, "data_size")

    if _has_any(latency, ["real-time", "realtime", "low latency", "under 20ms", "ms"]):
        tradeoffs.append("prioritizes latency over model complexity")

    if _has_any(budget, ["low", "limited"]):
        tradeoffs.append("limits model size and infrastructure spend")

    if _has_any(accuracy, ["high", "near-perfect", "perfect", "99%", "99.9", "99.99"]):
        tradeoffs.append("keeps a higher-capacity model to protect quality")

    if _has_any(data_size, ["large", "very large", "tb"]):
        tradeoffs.append("favors scalable storage and offline processing")

    if _has_any(latency, ["real-time", "realtime"]) and _has_any(budget, ["low", "limited"]):
        tradeoffs.append("compromise solution: edge-optimized deployment balances latency and budget")

    return tradeoffs


def oracle_recommend(hidden_constraints: Dict[str, str]) -> Dict[str, Any]:
    if _is_compromise_scenario(hidden_constraints):
        return {
            "model": "small_transformer",
            "deployment": "edge + batch hybrid",
            "architecture": "cost-optimized streaming compromise",
            "reasoning": [
                "No single design satisfies latency, cost, and quality simultaneously.",
                "Use a small edge path for low-latency responses and a batch path for heavy processing.",
                "Prioritize a compromise that preserves responsiveness under strict budget.",
            ],
        }

    return {
        "model": select_model(hidden_constraints),
        "deployment": select_deployment(hidden_constraints),
        "architecture": select_architecture(hidden_constraints),
        "reasoning": derive_tradeoffs(hidden_constraints),
    }