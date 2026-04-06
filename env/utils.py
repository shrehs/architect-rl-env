from typing import Dict, List


# Core required constraints for all system design interviews
REQUIRED_CONSTRAINTS: List[str] = [
    "use_case",
    "latency",
    "accuracy",
    "data_size",
    "update_frequency",
]

# Optional system design constraints that provide deeper understanding
SYSTEM_DESIGN_CONSTRAINTS: List[str] = [
    "consistency_requirement",  # CAP theorem
    "traffic_pattern",  # Load balancing
    "geography",  # CDN/replication
    "fault_tolerance",  # Availability
    "queueing_needs",  # Message queues
    "rate_limiting",  # API protection
    "budget",  # System design tradeoffs
]

# Full set of all constraints
ALL_CONSTRAINTS = REQUIRED_CONSTRAINTS + SYSTEM_DESIGN_CONSTRAINTS


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def extract_constraints(text: str, existing: Dict[str, str]) -> Dict[str, str]:
    normalized = _normalize_text(text)
    discovered: Dict[str, str] = {}

    # Core constraints
    if "use case" in normalized or "goal" in normalized or "application" in normalized:
        discovered["use_case"] = text

    if "ms" in normalized or "latency" in normalized or "real-time" in normalized or "response time" in normalized:
        discovered["latency"] = text

    if "accuracy" in normalized or "%" in normalized or "precise" in normalized or "correctness" in normalized:
        discovered["accuracy"] = text

    if "gb" in normalized or "tb" in normalized or "dataset" in normalized or "scale" in normalized or "volume" in normalized:
        discovered["data_size"] = text

    if "daily" in normalized or "hourly" in normalized or "stream" in normalized or "continuous" in normalized or "update" in normalized or "frequency" in normalized:
        discovered["update_frequency"] = text

    # System design constraints
    if "consistent" in normalized or "consistency" in normalized or "strong" in normalized or "eventual" in normalized or "cap" in normalized:
        discovered["consistency_requirement"] = text

    if "traffic" in normalized or "load" in normalized or "bursty" in normalized or "steady" in normalized or "spiky" in normalized:
        discovered["traffic_pattern"] = text

    if "region" in normalized or "global" in normalized or "cdn" in normalized or "geograph" in normalized or "latency" in normalized:
        discovered["geography"] = text

    if "fault" in normalized or "tolerance" in normalized or "availability" in normalized or "uptime" in normalized or "sla" in normalized:
        discovered["fault_tolerance"] = text

    if "queue" in normalized or "async" in normalized or "kafka" in normalized or "rabbit" in normalized or "message" in normalized:
        discovered["queueing_needs"] = text

    if "rate" in normalized or "limit" in normalized or "quota" in normalized or "throttle" in normalized or "protect" in normalized:
        discovered["rate_limiting"] = text

    if "budget" in normalized or "cost" in normalized or "expensive" in normalized or "cheap" in normalized:
        discovered["budget"] = text

    merged = dict(existing)
    merged.update(discovered)
    return merged


def missing_constraints(constraints: Dict[str, str]) -> List[str]:
    return [key for key in REQUIRED_CONSTRAINTS if key not in constraints]


def get_system_design_implications(constraints: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Maps constraints to real system design patterns and concepts.
    Helps agents understand CAP theorem, load balancing, availability, etc.
    """
    from .tasks import CONSTRAINT_CONCEPTS
    
    implications = {}
    
    for constraint_key, constraint_value in constraints.items():
        if constraint_key in CONSTRAINT_CONCEPTS:
            concept_info = CONSTRAINT_CONCEPTS[constraint_key]
            implications[constraint_key] = {
                "concept": concept_info.get("concept", ""),
                "description": concept_info.get("description", ""),
                "implications": concept_info.get("implications", []),
            }
    
    return implications


def generate_recommendation(constraints: Dict[str, str]) -> str:
    latency_text = constraints.get("latency", "")
    data_size_text = constraints.get("data_size", "")
    update_text = constraints.get("update_frequency", "")
    budget_text = constraints.get("budget", "")
    accuracy_text = constraints.get("accuracy", "")

    low_latency = any(token in latency_text.lower() for token in ["real-time", "ms", "low latency"])
    high_data = any(token in data_size_text.lower() for token in ["tb", "large", "million"])
    streaming = any(token in update_text.lower() for token in ["stream", "realtime", "continuous", "hourly"])
    low_budget = any(token in budget_text.lower() for token in ["low", "limited"])
    high_accuracy = any(token in accuracy_text.lower() for token in ["high", "near-perfect", "perfect", "99."])

    if low_latency and high_data and streaming and low_budget:
        return (
            "Recommend a balanced hybrid compromise with a small transformer on the edge, "
            "batch fallback processing, and a streaming control plane to preserve latency under budget."
        )

    if low_latency and high_accuracy and streaming:
        return (
            "Recommend a tradeoff-oriented hybrid architecture that keeps the latency-sensitive path lean "
            "while preserving accuracy with asynchronous model refinement."
        )

    if low_latency and streaming:
        return "Recommend event-driven microservices with Redis caching and Kafka stream processing."
    if high_data:
        return "Recommend batch-first lakehouse architecture with Spark jobs and feature store."
    return "Recommend modular service-oriented architecture with API gateway and relational datastore."


def has_conflicting_constraints(constraints: Dict[str, str]) -> bool:
    latency_text = constraints.get("latency", "").lower()
    accuracy_text = constraints.get("accuracy", "").lower()

    strict_latency = any(token in latency_text for token in ["<10ms", "ultra low", "real-time"])
    extreme_accuracy = any(token in accuracy_text for token in ["99.99", "near perfect", "perfect accuracy"])
    return strict_latency and extreme_accuracy
