#!/usr/bin/env python3
"""Test the canonical output strategy and penalty neutralization"""
import sys
sys.path.insert(0, './env')

from environment import ArchitectEnv

# Test _compute_similarity with canonical outputs
env = ArchitectEnv(task_id="medium")

# Test 1: Canonical medium path should NOT get generic penalty
canonical_medium = {
    "model": "small_cnn",
    "deployment": "streaming_service",
    "architecture": "event_driven_microservices",
}

oracle_path = {
    "model": "small_cnn",
    "deployment": "streaming_service",
    "architecture": "event_driven_microservices",
}

score = env._compute_similarity(canonical_medium, oracle_path)
print(f"✅ Test 1: Canonical medium path (perfect match)")
print(f"   Score: {score}")
print(f"   Expected: 0.99+ (3 exact matches, no generic penalty)")
print(f"   Status: {'PASS' if score >= 0.99 else 'FAIL'}")
print()

# Test 2: Non-canonical should get penalty
non_canonical = {
    "model": "random_model",
    "deployment": "random_deploy",
    "architecture": "random api architecture",  # Has "api" word
}

score2 = env._compute_similarity(non_canonical, oracle_path)
print(f"✅ Test 2: Non-canonical with generic words (all mismatches + penalty)")
print(f"   Score: {score2}")
print(f"   Expected: < 0.2 (3 mismatches = -0.15, generic penalty = -0.10)")
print(f"   Status: {'PASS' if score2 < 0.3 else 'FAIL'}")
print()

# Test 3: Canonical easy path
canonical_easy = {
    "model": "hybrid",
    "deployment": "standard_cloud",
    "architecture": "service_oriented",
}

oracle_easy = {
    "model": "hybrid",
    "deployment": "standard_cloud",
    "architecture": "service_oriented",
}

score3 = env._compute_similarity(canonical_easy, oracle_easy)
print(f"✅ Test 3: Canonical easy path (perfect match, easy task)")
print(f"   Score: {score3}")
print(f"   Expected: 0.99+ (3 exact matches, no generic penalty even though 'service' is generic)")
print(f"   Status: {'PASS' if score3 >= 0.99 else 'FAIL'}")
print()

# Test 4: Recommendation text parsing
from utils import generate_recommendation

constraints_easy = {
    "use_case": "recommendation ranking",
    "latency": "real-time (100ms)",
    "accuracy": "high (95%+)",
    "data_size": "moderate (100GB-1TB)",
    "update_frequency": "hourly",
    "budget": "low (startup budget)",
}

rec_easy = generate_recommendation(constraints_easy)
print(f"✅ Test 4: Canonical recommendation text (easy task)")
print(f"   Output:\n{rec_easy}")
print(f"   Contains 'simple service + relational db + caching'? {('simple service + relational db' in rec_easy.lower())}")
print()

# Test 5: Parser should extract correctly
parsed = env._infer_agent_recommendation(rec_easy)
print(f"✅ Test 5: Parse canonical recommendation (easy task)")
print(f"   Parsed: {parsed}")
print(f"   Expected: {canonical_easy}")
print(f"   Status: {'PASS' if parsed == canonical_easy else 'FAIL'}")
print()

print("=" * 70)
print("✅ All Tests Complete!")
print("=" * 70)
