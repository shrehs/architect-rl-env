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


def _bind_value(constraints: Dict[str, str], key: str, limit: int = 90) -> str:
    """Return a compact raw constraint value for explicit recommendation grounding."""
    raw = str(constraints.get(key, "unknown")).strip()
    compact = " ".join(raw.split())
    if not compact:
        return "unknown"
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit - 3]}..."


def choose_architecture(constraints: Dict[str, str]) -> str:
    """Deterministic architecture mapping for finalize-time recommendation quality."""
    required_keys = ["latency", "data_size", "budget"]
    if any(not str(constraints.get(key, "")).strip() for key in required_keys):
        return "api + relational db + caching"

    latency_text = _normalize_text(str(constraints.get("latency", "")))
    data_size_text = _normalize_text(str(constraints.get("data_size", "")))
    budget_text = _normalize_text(str(constraints.get("budget", "")))
    use_case_text = _normalize_text(str(constraints.get("use_case", "")))
    update_text = _normalize_text(str(constraints.get("update_frequency", "")))

    realtime_like = any(token in latency_text for token in ["real-time", "realtime", "near-real-time", "ms"])
    strict_realtime = any(token in latency_text for token in ["real-time", "realtime", "50ms", "100ms", "p99", "under"]) 
    budget_low_or_medium = any(token in budget_text for token in ["low", "limited", "tight", "medium", "growth"])

    data_small_or_medium = any(token in data_size_text for token in ["small", "medium", "moderate", "gb"])
    data_medium_or_large = any(token in data_size_text for token in ["medium", "large", "tb", "10tb", "tb+"])
    data_very_large = any(token in data_size_text for token in ["very large", "100tb", "100 tb", "pb"])
    medium_backup_signal = any(token in use_case_text for token in ["fraud", "risk", "detection"]) or any(
        token in update_text for token in ["stream", "streaming", "near real-time", "continuous"]
    )

    # Case 3: hard-scale real-time systems.
    if data_very_large and strict_realtime:
        return "hybrid batch + real-time serving"

    # Case 2: medium practical architecture.
    if realtime_like and budget_low_or_medium and (data_medium_or_large or medium_backup_signal) and not data_very_large:
        return "API + caching + periodic batch updates"

    # Noisy-medium fallback: near-real-time with constrained budget should remain practical.
    if realtime_like and (not strict_realtime) and budget_low_or_medium and not data_very_large:
        return "API + caching + periodic batch updates"

    # Case 1: easy/simple baseline (default for non-hard scenarios).
    if (data_small_or_medium and not strict_realtime) or not data_very_large:
        return "simple service + relational DB + caching"

    return "api + relational db + caching"


def extract_constraints(text: str, existing: Dict[str, str]) -> Dict[str, str]:
    normalized = _normalize_text(text)
    discovered: Dict[str, str] = {}

    use_case_markers = [
        "use case", "goal", "application", "recommend", "ranking", "feed", "search", "fraud", "assistant",
    ]
    latency_markers = [
        "ms", "latency", "real-time", "real time", "response time", "p95", "p99",
    ]
    accuracy_markers = [
        "accuracy", "%", "precise", "correctness", "f1", "auc", "recall", "precision",
    ]
    data_size_markers = [
        "gb", "tb", "pb", "dataset", "scale", "volume", "records", "rows", "events",
    ]
    update_markers = [
        "daily", "hourly", "stream", "streaming", "continuous", "update", "frequency", "near real-time",
    ]

    # Core constraints
    if any(x in normalized for x in use_case_markers):
        discovered["use_case"] = text

    if any(x in normalized for x in latency_markers):
        discovered["latency"] = text

    if any(x in normalized for x in accuracy_markers):
        discovered["accuracy"] = text

    if any(x in normalized for x in data_size_markers):
        discovered["data_size"] = text

    if any(x in normalized for x in update_markers):
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

    data_size_bound = _bind_value(constraints, "data_size")
    latency_bound = _bind_value(constraints, "latency")
    budget_bound = _bind_value(constraints, "budget")
    update_bound = _bind_value(constraints, "update_frequency")
    accuracy_bound = _bind_value(constraints, "accuracy")

    selected_architecture = choose_architecture(constraints)
    simple_case = selected_architecture == "simple service + relational DB + caching"

    if selected_architecture == "hybrid batch + real-time serving":
        return (
            "Because:\n"
            f"- latency = {latency_bound}\n"
            f"- data_size = {data_size_bound}\n"
            f"- budget = {budget_bound}\n"
            "the system needs both low-latency serving and large-scale offline processing.\n"
            "Decision:\n"
            "- Keep real-time inference paths lightweight\n"
            "- Shift expensive feature/model refresh to batch windows\n"
            "Architecture:\n"
            "- hybrid batch + real-time serving"
        )

    if selected_architecture == "API + caching + periodic batch updates":
        return (
            "Because:\n"
            f"- latency = {latency_bound}\n"
            f"- update_frequency = {update_bound}\n"
            f"- budget = {budget_bound}\n"
            "the design should stay practical and scalable without over-engineering.\n"
            "Decision:\n"
            "- Serve through API endpoints with aggressive caching\n"
            "- Use periodic batch updates for heavy recomputation\n"
            "Architecture:\n"
            "- API + caching + periodic batch updates"
        )

    if simple_case:
        # Keep easy/simple recommendations free of over-complex keywords.
        return (
            "Because:\n"
            f"- latency = {latency_bound}\n"
            f"- data_size = {data_size_bound}\n"
            f"- accuracy = {accuracy_bound}\n"
            "a simple production baseline is sufficient and operationally safer.\n"
            "Decision:\n"
            "- Favor straightforward components with low maintenance overhead\n"
            "- Use cache-first reads and relational storage\n"
            "Architecture:\n"
            "- simple service + relational DB + caching"
        )

    if selected_architecture == "api + relational db + caching":
        return (
            f"Because latency ~ {latency_bound}, data_size ~ {data_size_bound}, accuracy ~ {accuracy_bound}, "
            f"and budget ~ {budget_bound}, default to a robust baseline while requirements are still incomplete. "
            "Architecture: api + relational db + caching."
        )

    if low_latency and high_data and streaming and low_budget:
        return (
            f"Because latency ~ {latency_bound}, data_size ~ {data_size_bound}, and budget ~ {budget_bound}, "
            "prioritize cached online reads and controlled batch refresh. "
            "Architecture: API + caching + periodic batch updates."
        )

    if low_latency and high_accuracy and streaming:
        return (
            f"Because latency ~ {latency_bound} and accuracy ~ {accuracy_bound} are both demanding, "
            f"with update_frequency ~ {update_bound}, use a practical staged serving pattern. "
            "Architecture: API + caching + periodic batch updates."
        )

    if low_latency and streaming:
        return (
            f"Because latency ~ {latency_bound} and update_frequency ~ {update_bound}, "
            "prioritize fast cached reads with periodic batch refresh. "
            "Architecture: API + caching + periodic batch updates."
        )
    if high_data:
        return (
            f"Because data_size ~ {data_size_bound} and latency ~ {latency_bound}, "
            "use a split design with offline heavy lifting and lean online serving. "
            "Architecture: hybrid batch + real-time serving."
        )
    return (
        f"Because latency ~ {latency_bound}, data_size ~ {data_size_bound}, accuracy ~ {accuracy_bound}, "
        f"and budget ~ {budget_bound}, keep the solution practical and maintainable. "
        "Architecture: simple service + relational DB + caching."
    )


def has_conflicting_constraints(constraints: Dict[str, str]) -> bool:
    latency_text = constraints.get("latency", "").lower()
    accuracy_text = constraints.get("accuracy", "").lower()

    strict_latency = any(token in latency_text for token in ["<10ms", "ultra low", "real-time"])
    extreme_accuracy = any(token in accuracy_text for token in ["99.99", "near perfect", "perfect accuracy"])
    return strict_latency and extreme_accuracy
