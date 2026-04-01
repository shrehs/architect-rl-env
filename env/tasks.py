from typing import Any, Dict

from .utils import REQUIRED_CONSTRAINTS, has_conflicting_constraints

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "Collect clear constraints and recommend architecture",
        "constraints": {
            "use_case": "recommendation ranking",
            "latency": "real-time",
            "accuracy": "high",
            "data_size": "moderate",
            "update_frequency": "hourly",
            "budget": "low",
        },
    },
    "medium": {
        "description": "Handle vague or partial user responses",
        "constraints": {
            "use_case": "fraud detection",
            "latency": "near-real-time",
            "accuracy": "high",
            "data_size": "large",
            "update_frequency": "streaming",
            "budget": "medium",
        },
    },
    "hard": {
        "description": "Handle conflicting constraints and optimize trade-offs",
        "constraints": {
            "use_case": "multimodal assistant",
            "latency": "real-time",
            "accuracy": "near-perfect",
            "data_size": "very large",
            "update_frequency": "continuous",
            "budget": "low",
        },
    },
}

TASK_CLASS_NAMES: Dict[str, str] = {
    "easy": "EasyTask",
    "medium": "MediumTask",
    "hard": "HardTask",
}


def grade_constraints(constraints: Dict[str, str], task_id: str) -> float:
    completeness = float(len([k for k in REQUIRED_CONSTRAINTS if k in constraints])) / float(len(REQUIRED_CONSTRAINTS))
    score = completeness

    if task_id == "medium" and completeness >= 0.6:
        score += 0.1

    if task_id == "hard":
        score += 0.15 if has_conflicting_constraints(constraints) else 0.0

    return float(max(0.0, min(1.0, score)))
