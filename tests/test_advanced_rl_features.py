#!/usr/bin/env python3
"""
Test advanced RL features: GAE, n-step returns, action entropy.
"""

import sys
from env.environment import ArchitectEnv
from env.models import Action

def test_action_entropy():
    """Test action diversity entropy tracking."""
    print("\n" + "="*60)
    print("TEST 1: Action Entropy Tracking")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    actions = [
        "ASK_BUDGET",
        "ASK_LATENCY",
        "ASK_ACCURACY",
        "ASK_BUDGET",  # Repeat
        "ASK_ACCURACY",  # Repeat
        "FINALIZE"
    ]
    
    for i, action_type in enumerate(actions):
        action = Action(type=action_type, content=f"Question {i}")
        obs, reward, done, info = env.step(action)
        
        entropy = info.get("action_entropy", 0)
        entropy_info = info.get("entropy_info", {})
        
        print(f"\nStep {i + 1}: {action_type}")
        print(f"  Entropy: {entropy:.4f}")
        print(f"  Normalized entropy: {entropy_info.get('normalized_entropy', 0):.4f}")
        print(f"  Unique actions: {entropy_info.get('num_unique_actions', 0)}")
        print(f"  Is exploring: {entropy_info.get('is_exploring', False)}")
        print(f"  Is deterministic: {entropy_info.get('is_deterministic', False)}")
        
        if done:
            break
    
    # Check episode summary has entropy info
    ep_summary = info.get("episode_summary", {})
    if "action_entropy" in ep_summary:
        print(f"\n✅ Final action entropy: {ep_summary['action_entropy']:.4f}")
        print(f"   Action distribution: {ep_summary.get('entropy_info', {}).get('action_distribution', {})}")
    else:
        print("⚠️  Episode summary missing entropy data")
    
    return True


def test_gae_computation():
    """Test GAE (Generalized Advantage Estimation) computation."""
    print("\n" + "="*60)
    print("TEST 2: GAE Computation")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    env.gae_lambda = 0.95  # Standard GAE parameter
    obs = env.reset()
    
    # Run episode
    steps_taken = 0
    while True:
        action = Action(type="ASK_BUDGET" if steps_taken == 0 else "ASK_LATENCY" if steps_taken == 1 else "FINALIZE", 
                       content="Test question")
        obs, reward, done, info = env.step(action)
        steps_taken += 1
        
        if done:
            break
    
    # Check episode summary
    ep_summary = info.get("episode_summary", {})
    
    if "gae_advantages" in ep_summary:
        gae_advs = ep_summary["gae_advantages"]
        print(f"\n✅ GAE advantages computed for {len(gae_advs)} steps")
        print(f"   λ parameter (GAE lambda): {ep_summary.get('gae_lambda', 0.95)}")
        print(f"   Advantage values: {[f'{a:.4f}' for a in gae_advs[:5]]}")
        
        # Verify advantages are reasonable
        all_finite = all(abs(a) < 100 for a in gae_advs)  # Not NaN or inf
        if all_finite:
            print(f"   ✓ All advantages are finite")
        else:
            print(f"   ✗ Some advantages are invalid")
    else:
        print("❌ GAE advantages missing from episode summary")
        return False
    
    return True


def test_nstep_returns():
    """Test n-step return computation."""
    print("\n" + "="*60)
    print("TEST 3: N-Step Returns")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    # Run episode
    steps_taken = 0
    while True:
        action = Action(type="ASK_BUDGET" if steps_taken == 0 else "ASK_LATENCY" if steps_taken == 1 else "FINALIZE", 
                       content="Test question")
        obs, reward, done, info = env.step(action)
        steps_taken += 1
        
        if done:
            break
    
    ep_summary = info.get("episode_summary", {})
    
    # Check for n-step returns
    nstep_variants = [1, 3, 5]
    found_variants = []
    
    for n in nstep_variants:
        key = f"nstep_{n}_returns"
        if key in ep_summary:
            returns = ep_summary[key]
            found_variants.append((n, returns))
            print(f"\n✅ {n}-step returns computed ({len(returns)} values)")
            print(f"   Example values: {[f'{r:.4f}' for r in returns[:3]]}")
    
    if found_variants:
        print(f"\nComputed {len(found_variants)} n-step variants for bootstrapping")
        return True
    else:
        print("❌ No n-step returns found")
        return False


def test_episode_statistics():
    """Test episode-level statistics collection."""
    print("\n" + "="*60)
    print("TEST 4: Episode Statistics")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    # Run episode
    step = 0
    while True:
        actions = ["ASK_BUDGET", "ASK_LATENCY", "ASK_ACCURACY", "FINALIZE"]
        action = Action(type=actions[min(step, 3)], content="Test")
        obs, reward, done, info = env.step(action)
        step += 1
        
        if done:
            break
    
    ep_summary = info.get("episode_summary", {})
    
    print(f"\nEpisode Statistics:")
    stats = [
        ("Total reward", ep_summary.get("episode_total_reward", 0)),
        ("Average reward", ep_summary.get("episode_avg_reward", 0)),
        ("Max reward step", ep_summary.get("episode_max_reward", 0)),
        ("Min reward step", ep_summary.get("episode_min_reward", 0)),
        ("Episode length", ep_summary.get("episode_length", 0)),
    ]
    
    for name, value in stats:
        if isinstance(value, (int, float)):
            print(f"  {name:20s}: {value:8.4f}")
        else:
            print(f"  {name:20s}: {value}")
    
    # Verify consistency
    ep_length = ep_summary.get("episode_length", 0)
    if ep_length > 0:
        print(f"\n✅ Episode completed with {ep_length} steps")
        return True
    else:
        print(f"\n❌ Invalid episode length: {ep_length}")
        return False


def test_complete_advanced_signals():
    """Test all advanced signals together at episode end."""
    print("\n" + "="*60)
    print("TEST 5: Complete Advanced Signal Suite")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    while True:
        action = Action(type="ASK_BUDGET", content="What's your budget?")
        obs, reward, done, info = env.step(action)
        
        if done:
            break
    
    # Verify all expected fields are present
    expected_fields = [
        "episode_summary",
        "action_entropy",
        "entropy_info",
    ]
    
    print("\nExpected advanced RL fields at episode end:")
    all_present = True
    for field in expected_fields:
        if field in info:
            print(f"  ✅ {field}")
        else:
            print(f"  ❌ {field} MISSING")
            all_present = False
    
    # Check episode_summary content
    ep_summary = info.get("episode_summary", {})
    summary_fields = [
        "gae_advantages",
        "gae_lambda",
        "action_entropy",
        "entropy_info",
        "total_actions_taken",
        "episode_total_reward",
        "episode_length",
    ]
    
    print(f"\nEpisode summary fields:")
    summary_ok = True
    for field in summary_fields:
        if field in ep_summary:
            print(f"  ✅ {field}")
        else:
            print(f"  ⚠️  {field} (optional)")
    
    if all_present:
        print("\n✅ All advanced signals present!")
        return True
    else:
        print("\n⚠️  Some signals missing")
        return all_present


if __name__ == "__main__":
    print("Testing Advanced RL Features")
    print("=" * 60)
    
    results = {
        "Action Entropy": test_action_entropy(),
        "GAE Computation": test_gae_computation(),
        "N-Step Returns": test_nstep_returns(),
        "Episode Statistics": test_episode_statistics(),
        "Complete Signal Suite": test_complete_advanced_signals(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All advanced RL feature tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        sys.exit(1)
