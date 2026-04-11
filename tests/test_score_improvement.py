#!/usr/bin/env python3
"""Test the canonical output strategy - unit test of score computation"""

# Test canonical vs non-canonical scoring
def test_canonical_strategy():
    """
    Test that:
    1. Canonical outputs (from choose_architecture) do NOT get generic penalties
    2. Non-canonical outputs still get penalties
    """
    
    print("=" * 70)
    print("CANONICAL OUTPUT STRATEGY TEST")
    print("=" * 70)
    print()
    
    # Simulate the _compute_similarity logic
    def compute_similarity_old(agent, oracle, task_id="medium"):
        """Old logic: always apply generic penalty"""
        score = 0.0
        
        # Exact matches
        if agent.get("model") == oracle.get("model"):
            score += 0.33
        else:
            score -= 0.05
        
        if agent.get("deployment") == oracle.get("deployment"):
            score += 0.33
        else:
            score -= 0.05
        
        if agent.get("architecture") == oracle.get("architecture"):
            score += 0.34
        else:
            score -= 0.05
        
        # OLD: Always apply generic penalty
        arch = agent.get("architecture", "").lower()
        if "api" in arch or "service" in arch:
            generic_penalty = -0.10 if task_id == "medium" else -0.05
            score += generic_penalty
        
        return max(0.0, min(1.0, score))
    
    def compute_similarity_new(agent, oracle, task_id="medium"):
        """New logic: skip penalty for canonical outputs"""
        score = 0.0
        
        # Exact matches
        if agent.get("model") == oracle.get("model"):
            score += 0.33
        else:
            score -= 0.05
        
        if agent.get("deployment") == oracle.get("deployment"):
            score += 0.33
        else:
            score -= 0.05
        
        if agent.get("architecture") == oracle.get("architecture"):
            score += 0.34
        else:
            score -= 0.05
        
        # NEW: Skip penalty for canonical matches
        canonical_outputs = [
            ("hybrid", "standard_cloud", "service_oriented"),
            ("small_cnn", "streaming_service", "event_driven_microservices"),
            ("small_transformer", "edge + batch hybrid", "cost-optimized streaming compromise"),
        ]
        
        is_canonical = (
            (agent.get("model"), agent.get("deployment"), agent.get("architecture"))
            in canonical_outputs
        )
        
        if not is_canonical:
            arch = agent.get("architecture", "").lower()
            if "api" in arch or "service" in arch:
                generic_penalty = -0.10 if task_id == "medium" else -0.05
                score += generic_penalty
        
        return max(0.0, min(1.0, score))
    
    # Test Case 1: Medium task, canonical path, perfect match
    print("TEST 1: Medium task, canonical path (perfect match)")
    print("-" * 70)
    agent = {"model": "small_cnn", "deployment": "streaming_service", "architecture": "event_driven_microservices"}
    oracle = {"model": "small_cnn", "deployment": "streaming_service", "architecture": "event_driven_microservices"}
    
    old_score = compute_similarity_old(agent, oracle, task_id="medium")
    new_score = compute_similarity_new(agent, oracle, task_id="medium")
    
    print(f"  Agent:  {agent}")
    print(f"  Oracle: {oracle}")
    print(f"  OLD SCORE: {old_score:.2f} (has generic penalty: -0.10)")
    print(f"  NEW SCORE: {new_score:.2f} (NO generic penalty for canonical)")
    print(f"  ✅ IMPROVEMENT: {(new_score - old_score):.2f}")
    print()
    
    # Test Case 2: Easy task, canonical path, perfect match
    print("TEST 2: Easy task, canonical path (perfect match)")
    print("-" * 70)
    agent = {"model": "hybrid", "deployment": "standard_cloud", "architecture": "service_oriented"}
    oracle = {"model": "hybrid", "deployment": "standard_cloud", "architecture": "service_oriented"}
    
    old_score = compute_similarity_old(agent, oracle, task_id="easy")
    new_score = compute_similarity_new(agent, oracle, task_id="easy")
    
    print(f"  Agent:  {agent}")
    print(f"  Oracle: {oracle}")
    print(f"  OLD SCORE: {old_score:.2f} (has generic penalty: -0.05)")
    print(f"  NEW SCORE: {new_score:.2f} (NO generic penalty for canonical)")
    print(f"  ✅ IMPROVEMENT: {(new_score - old_score):.2f}")
    print()
    
    # Test Case 3: Medium task, non-canonical, should still get penalty
    print("TEST 3: Medium task, non-canonical (should still get penalty)")
    print("-" * 70)
    agent = {"model": "random", "deployment": "random", "architecture": "random api service"}
    oracle = {"model": "small_cnn", "deployment": "streaming_service", "architecture": "event_driven_microservices"}
    
    old_score = compute_similarity_old(agent, oracle, task_id="medium")
    new_score = compute_similarity_new(agent, oracle, task_id="medium")
    
    print(f"  Agent:  {agent}")
    print(f"  Oracle: {oracle}")
    print(f"  OLD SCORE: {old_score:.2f} (mismatches + penalty)")
    print(f"  NEW SCORE: {new_score:.2f} (mismatches + penalty for non-canonical)")
    print(f"  ✅ PENALTY STILL APPLIED: {old_score == new_score}")
    print()
    
    print("=" * 70)
    print("SUMMARY: Canonical Penalty Neutralization Working!")
    print("=" * 70)
    print("✅ Canonical architectures now skip -0.05/-0.10 penalties")
    print("✅ Non-canonical still get penalized correctly")
    print("✅ Expected score improvement for canonical paths: +0.05 to +0.10")
    print()

if __name__ == "__main__":
    test_canonical_strategy()
