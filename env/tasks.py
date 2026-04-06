from typing import Any, Dict

from .utils import REQUIRED_CONSTRAINTS, has_conflicting_constraints

# =============================================================================
# SYSTEM DESIGN CONCEPT MAPPING
# =============================================================================
# Maps constraints to real system design interview concepts

CONSTRAINT_CONCEPTS = {
    # Scale Constraints
    "data_size": {
        "concept": "SCALABILITY",
        "description": "How does the system handle growing data volumes?",
        "implications": ["Database sharding", "Data partitioning", "Storage optimization"],
        "cap_relevance": "Affects consistency guarantees across partitions"
    },
    
    # Performance Constraints  
    "latency": {
        "concept": "LATENCY_THROUGHPUT_TRADEOFF",
        "description": "Response time vs request handling capacity",
        "implications": ["Caching strategies", "Async processing", "CDN"],
        "cap_relevance": "Lower latency may require eventual consistency"
    },
    
    # Correctness Constraints
    "accuracy": {
        "concept": "CONSISTENCY",
        "description": "CAP Theorem: Consistency vs Availability tradeoff",
        "implications": ["Strong vs eventual consistency", "Leader-follower vs multi-master"],
        "cap_relevance": "High accuracy = strong consistency requirement"
    },
    
    # Freshness Constraints
    "update_frequency": {
        "concept": "EVENTUAL_CONSISTENCY",
        "description": "How quickly data must propagate through system",
        "implications": ["Replication lag", "Event streaming", "Cache invalidation"],
        "cap_relevance": "Determines acceptable replication delay"
    },
    
    # Cost Constraints
    "budget": {
        "concept": "SYSTEM_DESIGN_TRADEOFFS",
        "description": "Financial constraints driving architecture decisions",
        "implications": ["Cloud vs on-premise", "Managed vs self-hosted", "Caching vs compute"],
        "cap_relevance": "Budget constraints force CAP tradeoff decisions"
    },
    
    # New System Design Constraints
    "consistency_requirement": {
        "concept": "CAP_THEOREM",
        "description": "Strong consistency vs eventual consistency requirement",
        "values": ["strong", "eventual", "bounded"],
        "implications": ["Synchronous replication", "Read replicas", "Version vectors"],
    },
    
    "traffic_pattern": {
        "concept": "LOAD_BALANCING",
        "description": "How traffic is distributed (affects load balancing strategy)",
        "values": ["steady", "bursty", "temporal-spikes", "geographical"],
        "implications": ["Load balancing algorithm", "Auto-scaling strategy", "Queue management"],
    },
    
    "geography": {
        "concept": "CDN_REPLICATION",
        "description": "Geographic distribution requirements",
        "values": ["single-region", "multi-region", "global"],
        "implications": ["Data replication", "CDN for static content", "Regional failover"],
    },
    
    "fault_tolerance": {
        "concept": "AVAILABILITY",
        "description": "Required uptime and failure recovery capability",
        "values": ["low", "medium", "high", "critical"],  
        "implications": ["Replication factor", "Backup strategy", "Circuit breakers"],
    },
    
    "queueing_needs": {
        "concept": "MESSAGE_QUEUES",
        "description": "Async processing and decoupling requirements",
        "values": ["none", "light", "heavy"],
        "implications": ["Kafka/RabbitMQ", "Dead letter queues", "Event sourcing"],
    },
    
    "rate_limiting": {
        "concept": "API_PROTECTION",
        "description": "Rate limiting and quota enforcement needs",
        "values": ["none", "per-user", "per-ip", "adaptive"],
        "implications": ["Token bucket algorithm", "Throttling strategies", "DDoS protection"],
    },
}

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "Collect clear constraints and recommend architecture",
        "use_case": "recommendation ranking",
        "constraints": {
            # Traditional constraints
            "use_case": "recommendation ranking",
            "latency": "real-time (100ms)",
            "accuracy": "high (95%+)",
            "data_size": "moderate (100GB-1TB)",
            "update_frequency": "hourly",
            "budget": "low (startup budget)",
            # System design constraints
            "consistency_requirement": "eventual (stale rankings acceptable)",
            "traffic_pattern": "steady",
            "geography": "single-region",
            "fault_tolerance": "low",
        },
        "system_design_focus": [
            "Caching (Redis for ranking cache)",
            "Database indexing (optimize query performance)",
            "CDN for static content",
        ],
    },
    "medium": {
        "description": "Handle vague or partial user responses, system design tradeoffs",
        "use_case": "fraud detection",
        "constraints": {
            # Traditional constraints
            "use_case": "fraud detection",
            "latency": "near-real-time (500ms)",
            "accuracy": "near-perfect (99.5%+)",
            "data_size": "large (10TB+)",
            "update_frequency": "streaming",
            "budget": "medium (growth stage)",
            # System design constraints
            "consistency_requirement": "strong (fraud signals must be propagated immediately)",
            "traffic_pattern": "bursty (traffic spikes at key times)",
            "geography": "multi-region",
            "fault_tolerance": "high (fraud detection can't go down)",
            "queueing_needs": "heavy (event stream processing)",
        },
        "system_design_focus": [
            "CAP Theorem: Consistency vs Availability",
            "Event Stream Processing (Kafka)",
            "Real-time feature computation",
            "Multi-region replication",
            "Circuit breakers and fallbacks",
        ],
    },
    "hard": {
        "description": "Handle conflicting constraints, major tradeoffs, distributed systems",
        "use_case": "multimodal assistant",
        "constraints": {
            # Traditional constraints
            "use_case": "multimodal assistant (text, image, audio)",
            "latency": "real-time (50ms p99)",
            "accuracy": "near-perfect (99.9%+)",
            "data_size": "very large (100TB+ models)",
            "update_frequency": "continuous (model updates)",
            "budget": "low (but needs cutting-edge performance)",
            # System design constraints
            "consistency_requirement": "bounded (eventual with < 1 second lag)",
            "traffic_pattern": "temporal-spikes (peak hours)",
            "geography": "global (CDN distribution)",
            "fault_tolerance": "critical (SLA 99.99%)",
            "queueing_needs": "heavy (complex job scheduling)",
            "rate_limiting": "adaptive (token bucket with priority)",
        },
        "system_design_focus": [
            "Global distribution with CDN",
            "Consistency models: CAP theorem tradeoffs",
            "Model serving infrastructure (distributed inference)",
            "Event-sourced audit logs",
            "Advanced load balancing (latency-aware)",
            "Chaos engineering & resilience",
        ],
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
