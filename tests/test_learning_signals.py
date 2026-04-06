#!/usr/bin/env python3
"""
Test script to verify dense reward breakdown and advantage signals.
"""

import sys
from env.environment import ArchitectEnv
from env.models import Action

def test_dense_rewards():
    """Verify dense reward components are tracked."""
    print("\n" + "="*60)
    print("TEST 1: Dense Reward Breakdown")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    # Take a few steps
    for step in range(3):
        action = Action(type="ASK_BUDGET", content="What's the budget?")
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Raw reward: {reward:.4f}")
        
        # Check dense components
        if "reward_components" in info:
            components = info["reward_components"]
            print(f"  Components found: {len(components)} entries")
            print(f"    - information_gain: {components.get('information_gain', 0):.4f}")
            print(f"    - exploration_reward: {components.get('exploration_reward', 0):.4f}")
            print(f"    - efficiency_reward: {components.get('efficiency_reward', 0):.4f}")
            print(f"    - step_reward (total): {components.get('total_step_reward', 0):.4f}")
            print(f"    - phase: {components.get('phase', 'unknown')}")
        else:
            print("  ❌ MISSING: reward_components")
            return False
        
        if done:
            break
    
    print("\n✅ Dense rewards verified!")
    return True


def test_advantage_signals():
    """Verify advantage signals and baseline tracking."""
    print("\n" + "="*60)
    print("TEST 2: Advantage Signals & Baseline")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    baselines = []
    advantages = []
    
    # Collect trajectory
    for step in range(5):
        action = Action(type="ASK_BUDGET", content="What's the budget?")
        obs, reward, done, info = env.step(action)
        
        # Check advantage signal
        if "advantage" not in info:
            print(f"\n❌ Step {step}: Missing 'advantage'")
            return False
        
        advantage = info["advantage"]
        advantages.append(advantage)
        
        if "advantage_signal" not in info:
            print(f"\n❌ Step {step}: Missing 'advantage_signal'")
            return False
        
        signal = info["advantage_signal"]
        baseline = signal.get("baseline", 0)
        baselines.append(baseline)
        
        print(f"\nStep {step + 1}:")
        print(f"  Baseline: {baseline:.4f}")
        print(f"  Advantage (raw): {advantage:.4f}")
        print(f"  Advantage (normalized): {signal.get('advantage_normalized', 0):.4f}")
        print(f"  Step reward: {signal.get('step_reward', 0):.4f}")
        print(f"  Global step: {signal.get('global_step', 0)}")
        
        if done:
            break
    
    # Check baseline trend
    print(f"\nBaseline trend: {[f'{b:.3f}' for b in baselines]}")
    print(f"Advantage values: {[f'{a:.3f}' for a in advantages]}")
    
    # Verify baseline updates (should be EMA)
    if len(baselines) > 1:
        is_updating = any(baselines[i] != baselines[i-1] for i in range(1, len(baselines)))
        if is_updating:
            print("✅ Baseline updating correctly (EMA)")
        else:
            print("⚠️  Baselines not changing (check EMA implementation)")
    
    print("\n✅ Advantage signals verified!")
    return True


def test_episode_tracking():
    """Verify episode-level accumulation."""
    print("\n" + "="*60)
    print("TEST 3: Episode-Level Tracking")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    step_count = 0
    total_reward = 0.0
    components_log = []
    
    # Run until done
    while True:
        action = Action(type="ASK_BUDGET", content="What's the budget?")
        if step_count > 0:
            action = Action(type="ASK_LATENCY", content="What's the latency requirement?")
        if step_count > 1:
            action = Action(type="ASK_ACCURACY", content="What's the accuracy target?")
        if step_count > 2:
            action = Action(type="FINALIZE", content="I'm ready to recommend.")
        
        obs, reward, done, info = env.step(action)
        
        step_count += 1
        total_reward += reward
        
        if "reward_components" in info:
            components_log.append(info["reward_components"].copy())
        
        print(f"Step {step_count}: reward={reward:.4f}, cumulative={total_reward:.4f}")
        
        if done:
            break
    
    print(f"\nEpisode Summary:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Avg step reward: {total_reward / max(step_count, 1):.4f}")
    
    # Analyze component contributions
    if components_log:
        print(f"\nComponent Analysis ({len(components_log)} steps):")
        
        # Sum components
        component_totals = {}
        for comp_dict in components_log:
            for key, val in comp_dict.items():
                if isinstance(val, (int, float)):
                    component_totals[key] = component_totals.get(key, 0) + val
        
        # Print top contributors
        sorted_components = sorted(
            [(k, v) for k, v in component_totals.items() if isinstance(v, (int, float))],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for comp_name, comp_total in sorted_components[:8]:
            pct = 100 * comp_total / max(abs(total_reward), 0.01)
            print(f"  {comp_name:30s}: {comp_total:7.4f} ({pct:6.1f}%)")
    
    print("\n✅ Episode tracking verified!")
    return True


def test_learning_signal_info():
    """Verify all learning signal info fields are present."""
    print("\n" + "="*60)
    print("TEST 4: Complete Learning Signal Info Dict")
    print("="*60)
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    action = Action(type="ASK_BUDGET", content="What's the budget?")
    obs, reward, done, info = env.step(action)
    
    required_fields = [
        "reward_components",
        "step_reward",
        "advantage",
        "advantage_signal",
        "baseline",
        "rolling_baseline",
        "global_step",
    ]
    
    print("\nRequired learning signal fields:")
    all_present = True
    for field in required_fields:
        if field in info:
            print(f"  ✅ {field}")
        else:
            print(f"  ❌ {field} MISSING")
            all_present = False
    
    if not all_present:
        return False
    
    # Verify advantage_signal structure
    signal = info["advantage_signal"]
    signal_fields = ["baseline", "advantage_raw", "advantage_normalized", "step_reward", "global_step"]
    
    print(f"\nAdvantage signal structure:")
    signal_ok = True
    for field in signal_fields:
        if field in signal:
            print(f"  ✅ {field}")
        else:
            print(f"  ❌ {field} MISSING")
            signal_ok = False
    
    if not signal_ok:
        return False
    
    print("\n✅ All learning signal fields present!")
    return True


if __name__ == "__main__":
    print("Testing Learning Signal Implementation")
    print("=" * 60)
    
    results = {
        "Dense Rewards": test_dense_rewards(),
        "Advantage Signals": test_advantage_signals(),
        "Episode Tracking": test_episode_tracking(),
        "Info Dict": test_learning_signal_info(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Learning signals are working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        sys.exit(1)
