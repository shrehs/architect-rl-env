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


def _generate_alternative_architectures(hidden_constraints: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Feature 9: Trajectory diversity - multiple valid architectural paths.
    Returns alternative valid solutions with different tradeoffs.
    Agents can choose ANY of these paths and receive equal credit.
    
    Examples:
    - Streaming (event-driven): Best for real-time updates, requires infra
    - Batch (lakehouse): Best for large-scale accuracy, tolerates latency
    - Hybrid (cloud): Balanced cost/latency/accuracy
    - Edge (optimized): Best for ultra-low latency, limited resources
    """
    latency = _normalized_value(hidden_constraints, "latency")
    accuracy = _normalized_value(hidden_constraints, "accuracy")
    data_size = _normalized_value(hidden_constraints, "data_size")
    budget = _normalized_value(hidden_constraints, "budget")
    update_frequency = _normalized_value(hidden_constraints, "update_frequency")
    
    alternatives = []
    
    # Path 1: Streaming (event-driven microservices)
    # Good for: continuous updates, real-time requirements
    if _has_any(update_frequency, ["stream", "continuous", "hourly"]) or _has_any(latency, ["real-time", "realtime"]):
        alternatives.append({
            "model": "small_cnn",
            "deployment": "streaming_service",
            "architecture": "event_driven_microservices",
            "tradeoffs": ["prioritizes real-time responsiveness", "requires Kafka/Kinesis infrastructure", "lower model complexity for throughput"],
            "rationale": "Continuous streaming architecture for always-on updates"
        })
    
    # Path 2: Batch (lakehouse for analytics)
    # Good for: high accuracy, large data, batch processing acceptable
    if _has_any(data_size, ["large", "very large", "tb"]):
        alternatives.append({
            "model": "transformer",
            "deployment": "batch_pipeline",
            "architecture": "hybrid_lakehouse",
            "tradeoffs": ["maximizes model accuracy via larger transformers", "uses batch processing (acceptable latency)", "leverages data lake for cost-effective storage"],
            "rationale": "Batch processing pipeline optimized for accuracy and scalability"
        })
    
    # Path 3: Hybrid Cloud (balanced)
    # Good for: when no extreme constraints force a specific path
    # FIXED: Only valid when agent didn't explicitly push toward specific paths
    # Make it more inclusive but still not the default for random agents
    if _has_any(data_size, ["small", "medium", "moderate"]) or \
       (not _has_any(latency, ["real-time", "realtime", "high"]) and \
        not _has_any(budget, ["low", "limited", "tight"]) and \
        not _has_any(data_size, ["large", "very large", "huge", "tb"])):
        alternatives.append({
            "model": "hybrid",
            "deployment": "standard_cloud",
            "architecture": "service_oriented",
            "tradeoffs": ["balanced cost/latency/accuracy", "standard cloud services", "moderate infrastructure requirements"],
            "rationale": "Cloud-based service architecture balancing all dimensions"
        })
    
    # Path 4: Edge-optimized (ultra-low latency, constrained)
    # Good for: extreme latency requirements, low budget
    if _has_any(latency, ["real-time", "realtime", "low latency", "under 20ms"]) and _has_any(budget, ["low", "limited"]):
        alternatives.append({
            "model": "small_cnn",
            "deployment": "edge_optimized",
            "architecture": "edge_optimized_inference",
            "tradeoffs": ["prioritizes latency (sub-20ms)", "severely constrains model size", "requires edge deployment infrastructure"],
            "rationale": "Ultra-low latency edge deployment for cost-sensitive scenarios"
        })
    
    # If no alternatives were generated, return empty (oracle must provide primary)
    return alternatives


def oracle_recommend(hidden_constraints: Dict[str, str]) -> Dict[str, Any]:
    """
    Feature 9: Multiple valid solutions with trajectory diversity.
    Returns primary recommendation + alternative valid paths.
    Agents receive full credit for ANY valid path, enabling exploration diversity.
    """
    if _is_compromise_scenario(hidden_constraints):
        primary = {
            "model": "small_transformer",
            "deployment": "edge + batch hybrid",
            "architecture": "cost-optimized streaming compromise",
            "reasoning": [
                "No single design satisfies latency, cost, and quality simultaneously.",
                "Use a small edge path for low-latency responses and a batch path for heavy processing.",
                "Prioritize a compromise that preserves responsiveness under strict budget.",
            ],
        }
        alternatives = _generate_alternative_architectures(hidden_constraints)
        return {
            "primary": primary,
            "alternatives": alternatives,
            "valid_paths": [primary] + alternatives,
            "path_count": len([primary] + alternatives),
        }

    primary = {
        "model": select_model(hidden_constraints),
        "deployment": select_deployment(hidden_constraints),
        "architecture": select_architecture(hidden_constraints),
        "reasoning": derive_tradeoffs(hidden_constraints),
    }
    
    # Generate alternative paths
    alternatives = _generate_alternative_architectures(hidden_constraints)
    
    return {
        "primary": primary,
        "alternatives": alternatives,
        "valid_paths": [primary] + alternatives,
        "path_count": len([primary] + alternatives),
    }