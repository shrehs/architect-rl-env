#!/usr/bin/env python3
"""
Test Laplace smoothing and temperature control for exploration bonuses.

Features:
1. Laplace smoothing: frequency = (count + 1) / (total + num_paths)
   - Ensures no path ever gets zero probability
   - Keeps exploration alive even after 100+ episodes

2. Temperature control: bonus = 0.05 * (1 - frequency) ** alpha
   - alpha=1.0: standard behavior
   - alpha>1.0: stronger push to rare paths
   - alpha<1.0: softer effect
"""

from env.environment import ArchitectEnv
import json


def compute_bonus_with_alpha(path_frequency: float, alpha: float = 1.0) -> float:
    """Compute contextual bonus with temperature control."""
    frequency_penalty = (1.0 - path_frequency) ** alpha
    return 0.05 * frequency_penalty


def demonstrate_laplace_smoothing():
    """Show how Laplace smoothing prevents zero probabilities."""
    print("\n" + "="*70)
    print("LAPLACE SMOOTHING: No path ever gets zero probability")
    print("="*70)
    
    num_paths = 5
    total_episodes = 100
    
    scenarios = [
        {"streaming": 80, "batch": 15, "cloud": 4, "edge": 1, "other": 0},  # Very skewed
        {"streaming": 90, "batch": 5, "cloud": 3, "edge": 2, "other": 0},   # Extremely skewed
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: Distribution across 100 episodes")
        print(f"  Episode counts: {scenario}")
        total = sum(scenario.values())
        
        print("\n  Old Formula: count / total")
        print("  Path         | Count | Old Freq | Problem")
        print("  " + "-"*50)
        for path, count in scenario.items():
            old_freq = count / total if total > 0 else 0
            problem = "⚠️ ZERO!" if count == 0 else "✓"
            print(f"  {path:12} | {count:5} | {old_freq:8.4f} | {problem}")
        
        print("\n  New Formula: (count + 1) / (total + num_paths)")
        print("  Path         | Count | New Freq | Improvement")
        print("  " + "-"*50)
        for path, count in scenario.items():
            new_freq = (count + 1) / (total + num_paths)
            old_freq = count / total if total > 0 else 0
            improvement = "✅ Non-zero!" if count == 0 else f"Refined: {old_freq:.4f}→{new_freq:.4f}"
            print(f"  {path:12} | {count:5} | {new_freq:8.4f} | {improvement}")


def demonstrate_temperature_control():
    """Show how alpha parameter tunes exploration strength."""
    print("\n" + "="*70)
    print("TEMPERATURE CONTROL: Tune exploration strength with alpha")
    print("="*70)
    
    num_paths = 5
    total_episodes = 50
    
    # Simulate a realistic distribution from 50-episode run
    path_counts = {
        "primary": 15,         # 30%
        "streaming": 25,       # 50% (favorite)
        "batch": 7,            # 14%
        "edge": 2,             # 4%
        "cloud": 1             # 2%
    }
    
    print(f"\nBased on {total_episodes} episodes:")
    for path, count in path_counts.items():
        freq = count / total_episodes
        print(f"  {path:12}: {count:2} episodes ({freq*100:5.1f}%)")
    
    # Test different alpha values
    alpha_values = [0.5, 1.0, 1.5, 2.0]
    
    print("\n" + "-"*70)
    print("Bonus rewards for each path at different temperature levels:")
    print("-"*70)
    print(f"{'Path':<12} | ", end="")
    for alpha in alpha_values:
        print(f"α={alpha}    | ", end="")
    print()
    print("-"*70)
    
    for path, count in path_counts.items():
        # Laplace smoothing
        smoothed_freq = (count + 1) / (total_episodes + num_paths)
        print(f"{path:<12} | ", end="")
        
        for alpha in alpha_values:
            bonus = compute_bonus_with_alpha(smoothed_freq, alpha)
            print(f"${bonus:6.4f}  | ", end="")
        print()
    
    print("\nInterpretation:")
    print("  α=0.5 (soft):      Gentle exploration bonus")
    print("  α=1.0 (standard):  Balanced exploration (current default)")
    print("  α=1.5 (moderate):  Stronger incentive for rare paths")
    print("  α=2.0 (aggressive):More aggressive push to undiscovered architectures")


def test_with_different_alphas():
    """Create environments with different alpha values and show bonuses."""
    print("\n" + "="*70)
    print("LIVE TEST: Environments with different alpha values")
    print("="*70)
    
    alphas = [0.8, 1.0, 1.5, 2.0]
    
    for alpha in alphas:
        print(f"\n--- Environment with exploration_alpha={alpha} ---")
        env = ArchitectEnv(task_id="easy", exploration_alpha=alpha)
        
        # Simulate some path selections to build frequency
        env._path_frequency = {
            "primary": 10,
            "alternative_1": 5,
            "alternative_2": 3,
            "alternative_3": 1,
            "alternative_4": 0
        }
        env._total_episodes = 19
        
        print(f"Path frequencies after 19 episodes:")
        for path, count in env._path_frequency.items():
            raw_freq = count / 19
            smoothed_freq = (count + 1) / (19 + 5)
            print(f"  {path:15}: {count:2}/19 → raw={raw_freq:.3f}, smoothed={smoothed_freq:.3f}")
        
        print(f"\nContextual bonuses with α={alpha}:")
        for path, count in env._path_frequency.items():
            smoothed_freq = (count + 1) / (19 + 5)
            frequency_penalty = (1.0 - smoothed_freq) ** alpha
            bonus = 0.05 * frequency_penalty
            print(f"  {path:15}: {bonus:7.5f} (penalty={frequency_penalty:.4f})")


def demonstrate_exploration_policy():
    """Show how the combined system drives exploration."""
    print("\n" + "="*70)
    print("EXPLORATION POLICY: Laplace + Temperature Work Together")
    print("="*70)
    
    print("\nKey Benefits:")
    print("\n1. Laplace Smoothing (count + 1) / (total + num_paths):")
    print("   ✅ Path with 0 episodes: never gets 0% bonus")
    print("   ✅ Even after 1000 episodes, rare paths stay incentivized")
    print("   ✅ Prevents premature convergence to single architecture")
    
    print("\n2. Temperature Control (bonus ** alpha):")
    print("   ✅ Fine-grained tuning of exploration vs exploitation")
    print("   ✅ alpha=1.0: current balanced behavior")
    print("   ✅ alpha=1.5: stronger push early when learning about paths")
    print("   ✅ alpha<1.0: softer exploration for stable learning")
    
    print("\n3. Combined Effect:")
    print("   ✅ Self-balancing without hardcoded curriculum")
    print("   ✅ Rare paths maintain ~99% bonus at α=1.0")
    print("   ✅ Common paths (>80% frequency) get minimal bonus")
    print("   ✅ Temperature knob allows experiment-specific tuning")
    
    # Demonstrate concrete scenario
    print("\n" + "-"*70)
    print("Example: Streaming path frequency over 100 episodes")
    print("-"*70)
    
    num_paths = 5
    
    for total in [10, 30, 60, 100]:
        count = int(total * 0.65)  # Streaming converges to ~65%
        smooth_freq = (count + 1) / (total + num_paths)
        bonus_1_0 = 0.05 * (1.0 - smooth_freq) ** 1.0
        bonus_1_5 = 0.05 * (1.0 - smooth_freq) ** 1.5
        
        print(f"After {total:3d} episodes (count={count:3d}):")
        print(f"  Frequency (smoothed): {smooth_freq:.4f} ({smooth_freq*100:.1f}%)")
        print(f"  Bonus at α=1.0: ${bonus_1_0:.5f}")
        print(f"  Bonus at α=1.5: ${bonus_1_5:.5f}")
        print()


if __name__ == "__main__":
    demonstrate_laplace_smoothing()
    demonstrate_temperature_control()
    test_with_different_alphas()
    demonstrate_exploration_policy()
    
    print("\n" + "="*70)
    print("✅ All demonstrations complete!")
    print("="*70)
    print("\nNext: Run evaluation with different alpha values:")
    print("  python experiments/run_evaluation.py --exploration-alpha 1.0")
    print("  python experiments/run_evaluation.py --exploration-alpha 1.5")
    print("  python experiments/run_evaluation.py --exploration-alpha 0.8")
    print()
