import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt

from env.agents import choose_action
from env.environment import ArchitectEnv
from env.models import Action
from env.utils import REQUIRED_CONSTRAINTS

MODES = ["clean", "noisy", "adversarial"]
AGENTS = ["random", "heuristic", "improved"]


def infer_failure_reason(final_info: Dict[str, object], coverage: float, phase: str) -> str:
    """Classify failure reason based on coverage and phase-specific indicators."""
    if coverage < 0.3:
        return "insufficient_exploration"
    elif phase == "refinement" and final_info.get("duplicate_submission", False):
        return "inefficient_refinement"
    elif phase == "decision" and final_info.get("hard_conflict", False) and not final_info.get("compromise_detected", False):
        return "no_tradeoff_reasoning"
    elif coverage < 0.7:
        return "incomplete_coverage"
    return "other"


def run_one_episode(task_id: str, mode: str, agent: str, global_path_frequency: Dict[str, int] = None, exploration_alpha: float = 1.0) -> Dict[str, float | int | str]:
    """Run a single episode tracking contextual diversity bonuses with temperature control.
    
    Args:
        task_id: Task difficulty
        mode: Evaluation mode (clean, noisy, adversarial)
        agent: Agent type (random, heuristic, improved)
        global_path_frequency: Shared frequency counter across episodes for contextual bonuses
        exploration_alpha: Temperature control for exploration bonus strength (1.0=default, >1.0=stronger)
    """
    if global_path_frequency is None:
        global_path_frequency = {"primary": 0, "alternative_1": 0, "alternative_2": 0}
    
    with patch("env.environment.random.choice", return_value=mode):
        env = ArchitectEnv(task_id=task_id, exploration_alpha=exploration_alpha)
        # Store global frequency for contextual bonus computation
        env._path_frequency = global_path_frequency.copy()
        env._total_episodes = sum(global_path_frequency.values())
        
        observation = env.reset()

        total_reward = 0.0
        final_info: Dict[str, object] = {}
        done = False
        final_phase = "exploration"
        total_noisy_observations = 0

        for _ in range(env.max_steps):
            action_type = choose_action(agent, observation)
            observation, reward, done, info = env.step(Action(type=action_type))
            total_reward += float(reward)
            final_info = info
            final_phase = str(info.get("phase", "exploration"))
            total_noisy_observations += int(info.get("num_noisy_observations", 0))
            if done:
                break

    steps = int(observation.step_count)
    oracle_score = float(final_info.get("oracle_score", 0.0))
    oracle_score = min(max(oracle_score, 0.0), 1.0)  # Defensive: clamp to [0.0, 1.0]
    compromise_detected = int(bool(final_info.get("compromise_detected", False)))
    coverage = float(
        final_info.get(
            "coverage",
            float(len(observation.constraints_collected)) / float(len(REQUIRED_CONSTRAINTS)),
        )
    )
    
    # Feature metrics (Feature 1-3, 5-6)
    counterfactual_gain_diff = float(final_info.get("counterfactual_gain_diff", 0.0))
    counterfactual_reward = float(final_info.get("counterfactual_reward", 0.0))
    efficiency_reward = float(final_info.get("efficiency_reward", 0.0))
    consistency_reward = float(final_info.get("consistency_reward", 0.0))
    consistency_score = float(final_info.get("consistency_score", 1.0))
    global_efficiency_score = float(final_info.get("global_efficiency_score", 0.0))
    global_efficiency_reward = float(final_info.get("global_efficiency_reward", 0.0))
    recovery_score = float(final_info.get("recovery_score", 1.0))
    trajectory_score = float(final_info.get("trajectory_score", 0.0))
    
    # Feature: Exploration Completeness
    exploration_completeness = float(final_info.get("exploration_completeness", 0.0))
    exploration_bonus = float(final_info.get("exploration_bonus", 0.0))
    discovered_constraints = int(final_info.get("discovered_constraints", 0))
    
    # Feature: Trajectory-Level Evaluation (Step-wise Reasoning Quality)
    information_gain_score = float(final_info.get("information_gain_score", 1.0))
    early_discoveries = int(final_info.get("early_discoveries", 0))
    late_discoveries = int(final_info.get("late_discoveries", 0))
    utilization_score = float(final_info.get("utilization_score", 0.0))
    constraints_used = int(final_info.get("constraints_used", 0))
    constraints_observed = int(final_info.get("constraints_observed", 0))
    redundancy_score = float(final_info.get("redundancy_score", 1.0))
    total_questions = int(final_info.get("total_questions", 0))
    repeated_questions = int(final_info.get("repeated_questions", 0))
    trajectory_quality_bonus = float(final_info.get("trajectory_quality_bonus", 0.0))
    trajectory_quality_reward = float(final_info.get("trajectory_quality_reward", 0.0))
    
    justification_score = float(final_info.get("justification_score", 0.0))
    justification_reward = float(final_info.get("justification_reward", 0.0))
    contradiction_handling_reward = float(final_info.get("contradiction_handling_reward", 0.0))
    regret = float(final_info.get("regret", 0.0))
    regret_penalty = float(final_info.get("regret_penalty", 0.0))
    available_checkpoints = final_info.get("available_checkpoints", [])
    phase_gating_violations = 1 if "error" in final_info else 0  # Feature 5
    
    # Feature 6: Observation noise metrics
    noise_rate = total_noisy_observations / max(steps, 1)  # Noisy obs per step average
    
    # Feature 9: Trajectory diversity tracking (with contextual bonuses)
    matched_trajectory = final_info.get("matched_trajectory", "unknown")
    valid_path_count = int(final_info.get("valid_path_count", 1))
    trajectory_diversity_bonus = float(final_info.get("trajectory_diversity_bonus", 0.0))
    
    # Feature 9 (Refined): Contextual diversity - how rare was the chosen path?
    path_frequency = float(final_info.get("path_frequency", 0.0))
    contextual_bonus_scale = float(final_info.get("contextual_bonus_scale", 0.0))
    exploration_alpha = float(final_info.get("exploration_alpha", 1.0))
    
    # Feature 10: Policy entropy - measure diversity of path selection
    policy_entropy = float(final_info.get("policy_entropy", 0.0))
    entropy_normalized = float(final_info.get("entropy_normalized", 0.0))
    
    # Feature 9 (Refined++): Time decay factor for time-aware exploration
    time_decay_factor = float(final_info.get("time_decay_factor", 1.0))
    
    # Feature: Explicit Tradeoff Reasoning
    detected_tradeoffs = final_info.get("detected_tradeoffs", [])
    tradeoff_count = final_info.get("tradeoff_count", 0)
    tradeoff_reasoning_reward = float(final_info.get("tradeoff_reasoning_reward", 0.0))
    tradeoff_score = float(final_info.get("tradeoff_score", 0.0))
    
    # Partial success: continuous score based on coverage and reasoning quality
    # Agents get credit proportional to what they achieved
    # partial_success = coverage × oracle_score (0.0 to 1.0 continuous)
    partial_success = coverage * oracle_score

    # Binary success definition: HIGH oracle alignment at episode end.
    # FIXED: oracle_score is now continuous (0.0-1.0), not binarized
    # Success requires strong constraint alignment: oracle_score >= 0.8
    # This filters out lucky random guesses and generic patterns
    success = 1 if done and oracle_score >= 0.8 else 0
    failure_reason = "success" if success else infer_failure_reason(final_info, coverage, final_phase)
    
    # Composite Final Score: combines oracle accuracy with trajectory quality
    # final_score = 0.7 * oracle_score + 0.3 * trajectory_score
    # This rewards agents for achieving right answer (0.7) AND via good process (0.3)
    final_score = 0.7 * oracle_score + 0.3 * trajectory_score

    # Feature 9 (Refined): Update global path frequency for contextual bonuses
    # Track which architectures are being chosen for adaptive exploration rewards
    if matched_trajectory in global_path_frequency:
        global_path_frequency[matched_trajectory] += 1
    
    return {
        "task": task_id,
        "mode": mode,
        "agent": agent,
        "steps": steps,
        "total_reward": total_reward,
        "oracle_score": oracle_score,
        "coverage": coverage,
        "success": success,
        "partial_success": partial_success,
        "compromise_detected": compromise_detected,
        "phase": final_phase,
        "failure_reason": failure_reason,
        "counterfactual_gain_diff": counterfactual_gain_diff,
        "counterfactual_reward": counterfactual_reward,
        "efficiency_reward": efficiency_reward,
        "repeated_questions": int(final_info.get("repeated_questions", 0)),
        "uncertainty_resolved": int(final_info.get("uncertainty_resolved", 0)),
        "irrelevant_questions": int(final_info.get("irrelevant_questions", 0)),
        "logical_sequences": int(final_info.get("logical_sequences", 0)),
        "consistency_reward": consistency_reward,
        "consistency_violations": int(final_info.get("consistency_violations", 0)),
        "stable_constraints": int(final_info.get("stable_constraints", 0)),
        "consistency_score": float(final_info.get("consistency_score", 1.0)),
        "global_efficiency_score": global_efficiency_score,
        "global_efficiency_reward": global_efficiency_reward,
        "global_optimal_steps": int(final_info.get("global_optimal_steps", 9)),
        "recovery_score": recovery_score,
        "trajectory_score": trajectory_score,
        "exploration_completeness": exploration_completeness,
        "exploration_bonus": exploration_bonus,
        "discovered_constraints": discovered_constraints,
        "information_gain_score": information_gain_score,
        "early_discoveries": early_discoveries,
        "late_discoveries": late_discoveries,
        "utilization_score": utilization_score,
        "constraints_used": constraints_used,
        "constraints_observed": constraints_observed,
        "redundancy_score": redundancy_score,
        "total_questions": total_questions,
        "trajectory_quality_bonus": trajectory_quality_bonus,
        "trajectory_quality_reward": trajectory_quality_reward,
        "justification_score": justification_score,
        "justification_reward": justification_reward,
        "constraints_mentioned": int(final_info.get("constraints_mentioned", 0)),
        "tradeoffs_mentioned": int(final_info.get("tradeoffs_mentioned", 0)),
        "coverage_score": float(final_info.get("coverage_score", 0.0)),
        "tradeoff_awareness_score": float(final_info.get("tradeoff_awareness_score", 0.0)),
        "final_score": final_score,
        "contradiction_handling_reward": contradiction_handling_reward,
        "regret": regret,
        "regret_penalty": regret_penalty,
        "had_checkpoints": 1 if available_checkpoints else 0,
        "phase_gating_violations": phase_gating_violations,
        "total_noisy_observations": total_noisy_observations,
        "noise_rate": noise_rate,
        "matched_trajectory": matched_trajectory,
        "valid_path_count": valid_path_count,
        "trajectory_diversity_bonus": trajectory_diversity_bonus,
        "path_frequency": path_frequency,
        "contextual_bonus_scale": contextual_bonus_scale,
        "exploration_alpha": exploration_alpha,
        "policy_entropy": policy_entropy,
        "entropy_normalized": entropy_normalized,
        "time_decay_factor": time_decay_factor,
        "detected_tradeoffs": str(detected_tradeoffs),  # Convert list to string for CSV
        "tradeoff_count": tradeoff_count,
        "tradeoff_reasoning_reward": tradeoff_reasoning_reward,
        "tradeoff_score": tradeoff_score,
    }


def save_csv(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    out_path = out_dir / "episode_metrics.csv"
    fieldnames = [
        "task",
        "mode",
        "agent",
        "steps",
        "total_reward",
        "oracle_score",
        "coverage",
        "success",
        "partial_success",
        "compromise_detected",
        "phase",
        "failure_reason",
        "counterfactual_gain_diff",
        "counterfactual_reward",
        "efficiency_reward",
        "repeated_questions",
        "uncertainty_resolved",
        "irrelevant_questions",
        "logical_sequences",
        "consistency_reward",
        "consistency_violations",
        "stable_constraints",
        "consistency_score",
        "global_efficiency_score",
        "global_efficiency_reward",
        "global_optimal_steps",
        "recovery_score",
        "trajectory_score",
        "exploration_completeness",
        "exploration_bonus",
        "discovered_constraints",
        "information_gain_score",
        "early_discoveries",
        "late_discoveries",
        "utilization_score",
        "constraints_used",
        "constraints_observed",
        "redundancy_score",
        "total_questions",
        "trajectory_quality_bonus",
        "trajectory_quality_reward",
        "justification_score",
        "justification_reward",
        "constraints_mentioned",
        "tradeoffs_mentioned",
        "coverage_score",
        "tradeoff_awareness_score",
        "final_score",
        "contradiction_handling_reward",
        "regret",
        "regret_penalty",
        "had_checkpoints",
        "phase_gating_violations",
        "total_noisy_observations",
        "noise_rate",
        "matched_trajectory",
        "valid_path_count",
        "trajectory_diversity_bonus",
        "path_frequency",
        "contextual_bonus_scale",
        "exploration_alpha",
        "policy_entropy",
        "entropy_normalized",
        "time_decay_factor",
        "detected_tradeoffs",
        "tradeoff_count",
        "tradeoff_reasoning_reward",
        "tradeoff_score",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return out_path


def plot_reward_vs_mode(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(MODES))
    width = 0.35
    for i, agent in enumerate(AGENTS):
        means = []
        for mode in MODES:
            vals = [
                float(r["total_reward"])
                for r in records
                if r["agent"] == agent and r["mode"] == mode
            ]
            means.append(mean(vals) if vals else 0.0)
        offsets = [p + (i - 0.5) * width for p in x]
        ax.bar(offsets, means, width=width, label=agent)

    ax.set_xticks(list(x))
    ax.set_xticklabels(MODES)
    ax.set_ylabel("Mean Total Reward")
    ax.set_title("Reward vs Mode")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "reward_vs_mode.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_oracle_vs_steps(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(AGENTS), figsize=(6 * len(AGENTS), 5), sharey=True)
    if len(AGENTS) == 1:
        axes = [axes]
    colors = {"clean": "#2E7D32", "noisy": "#F9A825", "adversarial": "#C62828"}

    for idx, agent in enumerate(AGENTS):
        ax = axes[idx]
        for mode in MODES:
            subset = [r for r in records if r["agent"] == agent and r["mode"] == mode]
            ax.scatter(
                [int(r["steps"]) for r in subset],
                [float(r["oracle_score"]) for r in subset],
                alpha=0.65,
                s=20,
                color=colors[mode],
                label=mode,
            )
        ax.set_title(f"{agent.title()} Agent")
        ax.set_xlabel("Steps")
        if idx == 0:
            ax.set_ylabel("Oracle Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Oracle Score vs Steps", y=1.02)
    fig.tight_layout()

    out_path = out_dir / "oracle_score_vs_steps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_success_rate(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(MODES))
    width = 0.35

    for i, agent in enumerate(AGENTS):
        rates = []
        for mode in MODES:
            vals = [int(r["success"]) for r in records if r["agent"] == agent and r["mode"] == mode]
            rates.append((sum(vals) / len(vals)) if vals else 0.0)
        offsets = [p + (i - 0.5) * width for p in x]
        ax.bar(offsets, rates, width=width, label=agent)

    ax.set_xticks(list(x))
    ax.set_xticklabels(MODES)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success Rate by Mode")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "success_rate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_compromise_detection_rate(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(AGENTS))
    rates = []

    for agent in AGENTS:
        vals = [int(r["compromise_detected"]) for r in records if r["agent"] == agent]
        rates.append((sum(vals) / len(vals)) if vals else 0.0)

    bars = ax.bar(list(x), rates, color=["#546E7A", "#6A1B9A"], width=0.55)
    ax.set_xticks(list(x))
    ax.set_xticklabels([agent.title() for agent in AGENTS])
    ax.set_ylabel("Compromise Detection Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Compromise Detection Rate")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.0%}", ha="center", va="bottom")

    fig.tight_layout()

    out_path = out_dir / "compromise_detection_rate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_reward_vs_phase(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    """Plot average reward by phase for each agent."""
    fig, ax = plt.subplots(figsize=(8, 5))
    phases = ["exploration", "refinement", "decision"]
    x = range(len(phases))
    width = 0.25

    for i, agent in enumerate(AGENTS):
        means = []
        for phase in phases:
            vals = [float(r["total_reward"]) for r in records if r["agent"] == agent and r["phase"] == phase]
            means.append(sum(vals) / len(vals) if vals else 0.0)
        offsets = [p + (i - 1) * width for p in x]
        ax.bar(offsets, means, width=width, label=agent.title())

    ax.set_xticks(list(x))
    ax.set_xticklabels([p.title() for p in phases])
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward vs Phase")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "reward_vs_phase.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_failure_distribution_by_phase(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    """Plot failure distribution across phases."""
    fig, ax = plt.subplots(figsize=(10, 6))
    phases = ["exploration", "refinement", "decision"]
    failure_types = ["insufficient_exploration", "inefficient_refinement", "no_tradeoff_reasoning", "incomplete_coverage", "other", "success"]
    
    # Build data by phase
    phase_data = {phase: {} for phase in phases}
    for phase in phases:
        for failure_type in failure_types:
            count = sum(1 for r in records if r["phase"] == phase and r["failure_reason"] == failure_type)
            phase_data[phase][failure_type] = count

    x = range(len(phases))
    width = 0.12
    colors = ["#D32F2F", "#F57C00", "#1976D2", "#FDD835", "#757575", "#388E3C"]

    for i, failure_type in enumerate(failure_types):
        values = [phase_data[phase].get(failure_type, 0) for phase in phases]
        offsets = [p + (i - 2.5) * width for p in x]
        ax.bar(offsets, values, width=width, label=failure_type.replace("_", " ").title(), color=colors[i])

    ax.set_xticks(list(x))
    ax.set_xticklabels([p.title() for p in phases])
    ax.set_ylabel("Count")
    ax.set_title("Failure Distribution by Phase")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / "failure_distribution_by_phase.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_path_distribution_over_time(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    """Plot percentage of each architectural path selected over episode sequence."""
    import pandas as pd
    
    df = pd.DataFrame(records)
    df['episode'] = range(len(df))
    
    # Compute rolling percentage for each path
    window = max(10, len(df) // 20)  # 20 windows or 10 episodes
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    paths = df['matched_trajectory'].unique()
    colors = plt.cm.tab10(range(len(paths)))
    
    for i, path in enumerate(sorted(paths)):
        # Compute rolling percentage
        path_selected = (df['matched_trajectory'] == path).astype(float)
        rolling_pct = path_selected.rolling(window=window, center=True).mean() * 100
        
        ax.plot(df['episode'], rolling_pct, label=path, linewidth=2, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('% Path Selected (rolling avg)', fontsize=12)
    ax.set_title('Architectural Path Distribution Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    fig.tight_layout()
    out_path = out_dir / "path_distribution_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_entropy_vs_oracle_score(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    """Plot policy entropy vs oracle score to show exploration doesn't hurt correctness."""
    import pandas as pd
    
    df = pd.DataFrame(records)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Filter to episodes with valid entropy data
    df_valid = df[df['policy_entropy'] > 0].copy()
    
    # Color by agent
    agents = df_valid['agent'].unique()
    colors = {'random': '#ff7f0e', 'heuristic': '#1f77b4', 'improved': '#2ca02c'}
    
    for agent in sorted(agents):
        data = df_valid[df_valid['agent'] == agent]
        ax.scatter(data['policy_entropy'], data['oracle_score'], 
                   alpha=0.6, s=50, label=agent, color=colors.get(agent, '#7f7f7f'))
    
    # Add trend line
    if len(df_valid) > 1:
        z = np.polyfit(df_valid['policy_entropy'], df_valid['oracle_score'], 1)
        p = np.poly1d(z)
        entropy_range = np.linspace(df_valid['policy_entropy'].min(), df_valid['policy_entropy'].max(), 100)
        ax.plot(entropy_range, p(entropy_range), "k--", linewidth=2, alpha=0.5, label='Trend')
    
    ax.set_xlabel('Policy Entropy (path diversity)', fontsize=12)
    ax.set_ylabel('Oracle Score (correctness)', fontsize=12)
    ax.set_title('Diversity vs Performance: Exploration Doesn\'t Hurt Correctness', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    fig.tight_layout()
    out_path = out_dir / "entropy_vs_oracle_score.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_entropy_over_time(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    """Plot policy entropy over episode sequence to show exploration convergence."""
    import pandas as pd
    
    df = pd.DataFrame(records)
    df['episode'] = range(len(df))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Entropy over time (raw)
    window = max(5, len(df) // 30)
    rolling_entropy = df['policy_entropy'].rolling(window=window, center=True).mean()
    
    ax1.plot(df['episode'], df['policy_entropy'], alpha=0.3, label='Raw', color='lightblue')
    ax1.plot(df['episode'], rolling_entropy, linewidth=2, label=f'Rolling avg (w={window})', color='darkblue')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Policy Entropy', fontsize=11)
    ax1.set_title('Policy Entropy Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: By agent
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent].reset_index(drop=True)
        rolling = agent_data['policy_entropy'].rolling(window=window, center=True).mean()
        ax2.plot(agent_data.index, rolling, linewidth=2, label=agent)
    
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Policy Entropy', fontsize=11)
    ax2.set_title('Policy Entropy by Agent', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    out_path = out_dir / "entropy_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def print_summary(records: List[Dict[str, float | int | str]]) -> None:
    print("\n=== Evaluation Summary ===")
    print("success = 1 if done and oracle_score >= 0.8  (FIXED: Restored continuous scoring)")
    print("partial_success = coverage × oracle_score (continuous, 0.0 to 1.0)")
    print("\nFeature 9 (FIXED): Trajectory Diversity with Proper Oracle Gradient")
    print("- oracle_score: CONTINUOUS (0.0-1.0), not binarized to 1.0")
    print("  - 0.0-0.3: Generic/random architectures")
    print("  - 0.3-0.6: Partial matches, loose alignment")
    print("  - 0.6-0.8: Good matches, constraint-aware")
    print("  - 0.8-1.0: Excellent matches, precise and reasoning-backed")
    print("- trajectory_diversity_bonus: +0.05 reward for exploring alternatives (orthogonal to correctness)")
    print("- Diversity is a SECONDARY bonus, not a replacement for correctness\n")

    for agent in AGENTS:
        for mode in MODES:
            subset = [r for r in records if r["agent"] == agent and r["mode"] == mode]
            avg_reward = mean(float(r["total_reward"]) for r in subset)
            avg_oracle = mean(float(r["oracle_score"]) for r in subset)
            avg_steps = mean(int(r["steps"]) for r in subset)
            success_rate = sum(int(r["success"]) for r in subset) / len(subset)
            partial_success_avg = mean(float(r["partial_success"]) for r in subset)
            print(
                f"{agent:9s} | {mode:11s} | reward={avg_reward:6.3f} | "
                f"oracle={avg_oracle:5.3f} | partial={partial_success_avg:5.3f} | "
                f"steps={avg_steps:4.2f} | success={success_rate:5.2%}"
            )
    
    # Feature 9: Trajectory diversity analysis (FIXED - orthogonal signals)
    print("\n=== Feature 9: Trajectory Diversity (Contextual Bonuses) ===")
    print("Adaptive exploration: bonus = 0.05 * (1 - path_frequency)")
    print("More reward for rare paths, less for overused ones\n")
    print("Architectural path distribution:")
    
    trajectory_counts = {}
    total_episodes = len(records)
    for record in records:
        traj = str(record.get("matched_trajectory", "unknown"))
        trajectory_counts[traj] = trajectory_counts.get(traj, 0) + 1
    
    for traj, count in sorted(trajectory_counts.items()):
        pct = 100 * count / total_episodes if total_episodes > 0 else 0
        print(f"  {traj:20s}: {count:3d} episodes ({pct:5.1f}%)")
    
    # Oracle score analysis - should be similar across all paths with the fix
    print("\n=== Oracle Score by Chosen Path (should be ~equal) ===")
    paths = set(r["matched_trajectory"] for r in records)
    for path in sorted(paths):
        path_records = [r for r in records if r["matched_trajectory"] == path]
        avg_oracle = sum(float(r["oracle_score"]) for r in path_records) / len(path_records)
        print(f"  {path:20s}: avg_oracle_score={avg_oracle:.3f}")
    
    # Feature 9 (Refined): Contextual bonus analysis
    print("\n=== Contextual Diversity Bonus (Adapts to Frequency) ===")
    print("Path | Frequency | Avg Bonus Scale* | Avg Bonus Reward")
    print("-----|-----------|-----------------|------------------")
    
    paths = set(r["matched_trajectory"] for r in records if r["matched_trajectory"] != "unknown")
    for path in sorted(paths):
        path_records = [r for r in records if r["matched_trajectory"] == path]
        avg_freq = sum(float(r.get("path_frequency", 0.0)) for r in path_records) / len(path_records) if path_records else 0
        avg_scale = sum(float(r.get("contextual_bonus_scale", 0.0)) for r in path_records) / len(path_records) if path_records else 0
        avg_bonus = sum(float(r.get("trajectory_diversity_bonus", 0.0)) for r in path_records) / len(path_records) if path_records else 0
        freq_pct = 100 * avg_freq
        print(f"{path:5s} | {freq_pct:8.1f}% |      {avg_scale:6.2f}x       |     {avg_bonus:6.3f}")
    print("\n* Bonus scale = (1 - frequency), higher means rarer path")
    print("  Rare paths get +full 0.05, common paths get less exploration reward")
    
    # Show overall diversity bonus impact
    diversity_bonus_episodes = sum(1 for r in records if float(r.get("trajectory_diversity_bonus", 0.0)) > 0.0)
    print(f"\nEpisodes earning exploration bonus: {diversity_bonus_episodes}/{total_episodes} ({100*diversity_bonus_episodes/total_episodes:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mode-wise evaluation experiments")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per (agent, mode)")
    parser.add_argument("--task", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--out-dir", type=str, default="artifacts/evaluation")
    parser.add_argument("--exploration-alpha", type=float, default=1.0, help="Temperature control for exploration bonus (1.0=balanced, >1.0=stronger rarity push)")
    args = parser.parse_args()

    total_run_episodes = args.episodes * len(AGENTS) * len(MODES)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature 9 (Refined+): Global path frequency for contextual diversity bonuses with temperature control
    # Persists across all (agent, mode, episode) combinations
    global_path_frequency = {"primary": 0, "alternative_1": 0, "alternative_2": 0}
    
    records: List[Dict[str, float | int | str]] = []
    for agent in AGENTS:
        for mode in MODES:
            for _ in range(args.episodes):
                records.append(run_one_episode(
                    task_id=args.task, 
                    mode=mode, 
                    agent=agent,
                    global_path_frequency=global_path_frequency,  # Pass persistent frequency tracker
                    exploration_alpha=args.exploration_alpha  # Pass temperature control
                ))

    csv_path = save_csv(records, out_dir)
    plot1 = plot_reward_vs_mode(records, out_dir)
    plot2 = plot_oracle_vs_steps(records, out_dir)
    plot3 = plot_success_rate(records, out_dir)
    plot4 = plot_compromise_detection_rate(records, out_dir)
    plot5 = plot_reward_vs_phase(records, out_dir)
    plot6 = plot_failure_distribution_by_phase(records, out_dir)
    plot7 = plot_path_distribution_over_time(records, out_dir)
    plot8 = plot_entropy_over_time(records, out_dir)
    plot9 = plot_entropy_vs_oracle_score(records, out_dir)

    # Structured logging: evaluation end
    success_rate = sum(1 for r in records if r["success"]) / len(records) if records else 0.0
    avg_oracle_score = mean([r["oracle_score"] for r in records]) if records else 0.0

    print(f"\nEvaluation complete: {len(records)} episodes, {success_rate*100:.1f}% success rate")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {plot1}")
    print(f"Saved plot: {plot2}")
    print(f"Saved plot: {plot3}")
    print(f"Saved plot: {plot4}")
    print(f"Saved plot: {plot5}")
    print(f"Saved plot: {plot6}")
    print(f"Saved plot: {plot7}")
    print(f"Saved plot: {plot8}")
    print(f"Saved plot: {plot9}")


if __name__ == "__main__":
    main()
