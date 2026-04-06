#!/usr/bin/env python3
"""
System Design Concepts Reference
Shows how the simulator maps to real system design interview questions
"""

from env.tasks import CONSTRAINT_CONCEPTS, TASKS
from env.utils import REQUIRED_CONSTRAINTS, SYSTEM_DESIGN_CONSTRAINTS, ALL_CONSTRAINTS

print("\n" + "="*120)
print("SYSTEM DESIGN SIMULATOR - CONCEPT MAPPING REFERENCE")
print("="*120)

print("\n[1] CORE REQUIRED CONSTRAINTS (Must ask about):")
print("-" * 120)
for i, constraint in enumerate(REQUIRED_CONSTRAINTS, 1):
    concept = CONSTRAINT_CONCEPTS.get(constraint, {})
    print(f"{i}. {constraint.upper():20s} -> {concept.get('concept', '?'):25s}")
    print(f"   {concept.get('description', '')}")

print("\n[2] OPTIONAL SYSTEM DESIGN CONSTRAINTS (Advanced):")
print("-" * 120)
for i, constraint in enumerate(SYSTEM_DESIGN_CONSTRAINTS, 1):
    concept = CONSTRAINT_CONCEPTS.get(constraint, {})
    print(f"{i}. {constraint.upper():20s} → {concept.get('concept', '?'):25s}")
    print(f"   {concept.get('description', '')}")
    if 'values' in concept:
        print(f"   Options: {', '.join(concept['values'])}")
    if 'implications' in concept and concept['implications']:
        print(f"   Patterns: {', '.join(concept['implications'][:2])}")

print("\n" + "="*120)
print("TASK DIFFICULTY LEVELS (What Agents Must Solve)")
print("="*120)

for task_id, task_info in TASKS.items():
    print(f"\n{task_id.upper()}:")
    print(f"  Use Case: {task_info.get('use_case', 'N/A')}")
    print(f"  Description: {task_info.get('description', '')}")
    print(f"  System Design Focus:")
    for focus in task_info.get('system_design_focus', []):
        print(f"    - {focus}")
    
    constraints = task_info.get('constraints', {})
    print(f"  Key Constraints:")
    print(f"    - Latency: {constraints.get('latency', 'N/A')}")
    print(f"    - Accuracy: {constraints.get('accuracy', 'N/A')}")
    print(f"    - Data Size: {constraints.get('data_size', 'N/A')}")
    print(f"    - Consistency: {constraints.get('consistency_requirement', 'N/A')}")
    if task_id == 'medium':
        print(f"    - Traffic Pattern: {constraints.get('traffic_pattern', 'N/A')}")
    if task_id == 'hard':
        print(f"    - Geography: {constraints.get('geography', 'N/A')}")
        print(f"    - Fault Tolerance: {constraints.get('fault_tolerance', 'N/A')}")

print("\n" + "="*120)
print("SYSTEM DESIGN PATTERNS COVERED")
print("="*120)

patterns = {
    "CAP THEOREM": [
        "Strong consistency (ACID)",
        "Eventual consistency (BASE)", 
        "Bounded consistency",
    ],
    "SCALABILITY": [
        "Database sharding",
        "Replication across shards",
        "Read replicas",
        "Consistent hashing",
    ],
    "LATENCY OPTIMIZATION": [
        "In-memory caching (Redis/Memcached)",
        "CDN for content delivery",
        "Connection pooling",
        "Async processing",
    ],
    "AVAILABILITY": [
        "Replication strategies",
        "Health checks",
        "Automatic failover",
        "Circuit breakers",
        "SLA management",
    ],
    "LOAD BALANCING": [
        "Round-robin",
        "Least-connections",
        "Auto-scaling",
        "Geo-routing",
    ],
    "ASYNC PROCESSING": [
        "Message queues (Kafka, RabbitMQ)",
        "Event sourcing",
        "CQRS (Command Query Responsibility Segregation)",
        "Dead letter queues",
    ],
    "API PROTECTION": [
        "Rate limiting (token bucket)",
        "Quota management",
        "Throttling",
        "DDoS protection",
    ],
    "DISTRIBUTED SYSTEMS": [
        "Conflict resolution",
        "Vector clocks",
        "CRDTs",
        "Consensus algorithms",
    ],
}

for pattern_category, patterns_list in patterns.items():
    print(f"\n{pattern_category}:")
    for pattern in patterns_list:
        print(f"  • {pattern}")

print("\n" + "="*120)
print("AGENT EVALUATION METRICS")
print("="*120)

print("""
Your agent will be evaluated on:

1. CONSTRAINT DISCOVERY (exploration_completeness)
   - Did agent ask about all relevant system design constraints?
   - Covers: use_case, latency, accuracy, data_size, update_frequency
   - Bonus: consistency, traffic_pattern, geography, fault_tolerance, etc.

2. CONCEPT UNDERSTANDING (utilization_score)
   - Did agent's final decision USE all discovered constraints?
   - Does recommendation address specific system design patterns?
   - Does agent explain CAP theorem tradeoff implications?

3. TRAJECTORY QUALITY (trajectory_score)
   - CONSISTENCY: Stable reasoning across constraints
   - EFFICIENCY: Optimal number of questions
   - RECOVERY: Can agent handle contradictory information?

4. REASONING QUALITY (justification_score)
   - Did agent explicitly mention system design concepts?
   - Does final recommendation discuss tradeoffs?
   - Does agent explain architectural choices?

5. FINAL ARCHITECTURE QUALITY (oracle_score)
   - Is recommendation architecturally sound?
   - Does it address all constraints?
   - Are the tradeoffs correctly balanced?

COMPOSITE SCORE:
  final_score = 0.7 * oracle_score + 0.3 * trajectory_score
  
This rewards BOTH accuracy AND process quality.
""")

print("\n" + "="*120)
print("EXAMPLE GOOD RESPONSES")
print("="*120)

print("""
EASY TASK - Recommendation Ranking:
  Agent asks about: latency (100ms), accuracy (95%+), data size (100GB)
  Recognizes: Single-region, eventual consistency OK
  Recommends: "SQL database with Redis cache for hot recommendations"
  Mentions: "Acceptable to cache for 1-2 minutes since hourly updates"
  
MEDIUM TASK - Fraud Detection:
  Agent asks about: consistency (critical!), traffic spikes, streaming, data volume
  Recognizes: CAP theorem -> must choose consistency over availability
  Recommends: "Leader-based replication with Kafka event stream"
  Mentions: "Fraud signals must propagate <1 second, using sync replication"
  
HARD TASK - Multimodal Assistant:
  Agent asks: latency (< 50ms p99), global requirements, SLA
  Recognizes: Multiple constraints conflict -> tradeoff analysis needed
  Recommends: "Distributed inference with CDN edge nodes, bounded eventual consistency"
  Mentions: "99.99% SLA requires multi-region failover + circuit breakers"
""")

print("\n" + "="*120)
print("System Design Simulator Ready for Training!")
print("="*120 + "\n")
