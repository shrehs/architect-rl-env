from typing import Dict, List


REQUIRED_CONSTRAINTS: List[str] = [
    "use_case",
    "latency",
    "accuracy",
    "data_size",
    "update_frequency",
]


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def extract_constraints(text: str, existing: Dict[str, str]) -> Dict[str, str]:
    normalized = _normalize_text(text)
    discovered: Dict[str, str] = {}

    if "use case" in normalized or "goal" in normalized:
        discovered["use_case"] = text

    if "ms" in normalized or "latency" in normalized or "real-time" in normalized:
        discovered["latency"] = text

    if "accuracy" in normalized or "%" in normalized or "precise" in normalized:
        discovered["accuracy"] = text

    if "gb" in normalized or "tb" in normalized or "dataset" in normalized:
        discovered["data_size"] = text

    if "daily" in normalized or "hourly" in normalized or "stream" in normalized:
        discovered["update_frequency"] = text

    merged = dict(existing)
    merged.update(discovered)
    return merged


def missing_constraints(constraints: Dict[str, str]) -> List[str]:
    return [key for key in REQUIRED_CONSTRAINTS if key not in constraints]


def generate_recommendation(constraints: Dict[str, str]) -> str:
    latency_text = constraints.get("latency", "")
    data_size_text = constraints.get("data_size", "")
    update_text = constraints.get("update_frequency", "")

    low_latency = any(token in latency_text.lower() for token in ["real-time", "ms", "low latency"])
    high_data = any(token in data_size_text.lower() for token in ["tb", "large", "million"])
    streaming = any(token in update_text.lower() for token in ["stream", "realtime", "continuous", "hourly"])

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
