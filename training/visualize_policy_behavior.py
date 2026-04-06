#!/usr/bin/env python3
"""
Policy Behavior Visualization

Visualizes:
1. Action distribution over time - Does agent stop repeating? Learn ordering?
2. Advantage per action type - Which actions are actually useful?
3. Learning dynamics - How does entropy and confidence evolve?
4. Action sequences - What patterns emerge?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sys

from env.environment import ArchitectEnv
from env.models import Action


class PolicyGradientAgent:
    """Simple policy gradient agent with detailed action tracking."""
    
    ACTION_MAP = [
        "ASK_BUDGET",
        "ASK_LATENCY", 
        "ASK_ACCURACY",
        "ASK_DATA_SIZE",
        "ASK_UPDATE_FREQUENCY",
        "FINALIZE"
    ]
    
    def __init__(self, state_dim=128, hidden_dim=64, lr=1e-3):
        self.action_head = nn.Linear(hidden_dim, len(self.ACTION_MAP))
        self.state_dim = state_dim
        
    def select_action(self, obs):
        """Select action using policy."""
        state_tensor = torch.randn(self.state_dim)
        
        # Simple random for now (just for illustration)
        action_logits = torch.randn(len(self.ACTION_MAP))
        action_dist = Categorical(logits=action_logits)
        action_idx = action_dist.sample()
        log_prob = action_dist.log_prob(action_idx)
        probs = action_dist.probs
        
        action_type = self.ACTION_MAP[action_idx.item()]
        
        return Action(type=action_type, content="Query"), log_prob, action_idx, probs


class TrainingMonitor:
    """Tracks detailed training metrics for visualization."""
    
    ACTION_MAP = [
        "ASK_BUDGET",
        "ASK_LATENCY", 
        "ASK_ACCURACY",
        "ASK_DATA_SIZE",
        "ASK_UPDATE_FREQUENCY",
        "FINALIZE"
    ]
    
    def __init__(self, num_episodes=100):
        self.num_episodes = num_episodes
        
        # Episode-level tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.entropy_patterns = []
        self.entropy_values = []
        
        # Action tracking
        self.action_sequences = []  # Per-episode action sequences
        self.action_counts_history = []  # Action counts per episode
        self.action_advantages = defaultdict(list)  # Per-action type advantages
        
        # Advantage tracking
        self.episode_advantages = []
        self.episode_advantages_by_action = defaultdict(list)
        
    def record_episode(self, trajectory, episode_num):
        """Record episode data for visualization."""
        self.episode_rewards.append(trajectory["episode_reward"])
        self.episode_lengths.append(trajectory["episode_length"])
        self.entropy_patterns.append(trajectory["entropy_pattern"])
        
        # Track action distribution
        actions = trajectory["actions"]
        self.action_sequences.append(actions)
        
        # Count actions this episode
        action_counts = {action: 0 for action in self.ACTION_MAP}
        for action in actions:
            action_counts[action] += 1
        self.action_counts_history.append(action_counts)
        
        # Track advantages by action
        advantages = trajectory["advantages"]
        for action, adv in zip(actions, advantages):
            self.episode_advantages_by_action[action].append(adv)
        
        self.episode_advantages.extend(advantages)
    
    def compute_statistics(self):
        """Compute statistics for visualization."""
        stats = {
            "avg_episode_reward": np.mean(self.episode_rewards),
            "avg_episode_length": np.mean(self.episode_lengths),
            "avg_advantage": np.mean(self.episode_advantages),
            
            # Per-action statistics
            "action_stats": {},
        }
        
        for action in self.ACTION_MAP:
            advantages = self.episode_advantages_by_action[action]
            if advantages:
                stats["action_stats"][action] = {
                    "avg_advantage": np.mean(advantages),
                    "std_advantage": np.std(advantages),
                    "count": len(advantages),
                }
            else:
                stats["action_stats"][action] = {
                    "avg_advantage": 0.0,
                    "std_advantage": 0.0,
                    "count": 0,
                }
        
        return stats
    
    def plot_action_distribution(self):
        """Plot 1: How does action distribution change over training?"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1a: Action frequency heatmap
        ax = axes[0]
        episodes = len(self.action_counts_history)
        action_evolution = np.zeros((len(self.ACTION_MAP), episodes))
        
        for ep, counts in enumerate(self.action_counts_history):
            for action_idx, action in enumerate(self.ACTION_MAP):
                action_evolution[action_idx, ep] = counts[action]
        
        # Normalize to probability
        action_evolution_pct = action_evolution / (action_evolution.sum(axis=0) + 1e-8) * 100
        
        im = ax.imshow(action_evolution_pct, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks(range(len(self.ACTION_MAP)))
        ax.set_yticklabels(self.ACTION_MAP)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Action Type")
        ax.set_title("Action Distribution Over Training (% of episode steps)", fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("% of steps")
        
        # Plot 1b: Action count trends (stacked area)
        ax = axes[1]
        categories = []
        for action_idx, action in enumerate(self.ACTION_MAP):
            counts = [self.action_counts_history[ep].get(action, 0) for ep in range(len(self.action_counts_history))]
            categories.append(counts)
        
        episodes_x = np.arange(len(self.action_counts_history))
        ax.stackplot(episodes_x, *categories, labels=self.ACTION_MAP, alpha=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Number of Actions")
        ax.set_title("Action Composition Over Training", fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_advantage_per_action(self):
        """Plot 2: Which actions are actually useful? (Advantage per action type)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 2a: Average advantage per action
        ax = axes[0, 0]
        actions = []
        avg_advantages = []
        action_counts = []
        
        for action in self.ACTION_MAP:
            advantages = self.episode_advantages_by_action[action]
            if advantages:
                actions.append(action)
                avg_advantages.append(np.mean(advantages))
                action_counts.append(len(advantages))
            else:
                actions.append(action)
                avg_advantages.append(0)
                action_counts.append(0)
        
        colors = ['green' if adv > 0 else 'red' for adv in avg_advantages]
        bars = ax.barh(actions, avg_advantages, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel("Average Advantage")
        ax.set_title("Average Advantage per Action Type", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, avg_advantages)):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                   va='center', ha='left' if val > 0 else 'right', fontsize=9)
        
        # Plot 2b: Advantage distribution (violin plot style)
        ax = axes[0, 1]
        advantages_data = []
        labels = []
        for action in self.ACTION_MAP:
            advantages = self.episode_advantages_by_action[action]
            if advantages:
                advantages_data.append(advantages)
                labels.append(action)
        
        bp = ax.boxplot(advantages_data, labels=labels, patch_artist=True, vert=False)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel("Advantage Value")
        ax.set_title("Advantage Distribution per Action", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 2c: Action utilization (frequency)
        ax = axes[1, 0]
        colors_util = plt.cm.Set3(np.linspace(0, 1, len(self.ACTION_MAP)))
        bars = ax.bar(range(len(self.ACTION_MAP)), action_counts, color=colors_util, alpha=0.7)
        ax.set_xticks(range(len(self.ACTION_MAP)))
        ax.set_xticklabels(self.ACTION_MAP, rotation=45, ha='right')
        ax.set_ylabel("Total Uses Across Training")
        ax.set_title("Action Frequency (Total Utilization)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2d: Advantage efficiency (avg advantage × frequency)
        ax = axes[1, 1]
        efficiency = [avg * count for avg, count in zip(avg_advantages, action_counts)]
        colors_eff = ['green' if e > 0 else 'red' for e in efficiency]
        bars = ax.bar(range(len(self.ACTION_MAP)), efficiency, color=colors_eff, alpha=0.7)
        ax.set_xticks(range(len(self.ACTION_MAP)))
        ax.set_xticklabels(self.ACTION_MAP, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel("Cumulative Advantage (Avg × Count)")
        ax.set_title("Total Contribution per Action Type", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_dynamics(self):
        """Plot 3: How do entropy and confidence evolve?"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 3a: Episode rewards over time
        ax = axes[0, 0]
        rewards = np.array(self.episode_rewards)
        episodes = np.arange(len(rewards))
        
        ax.plot(episodes, rewards, 'b-', alpha=0.5, label='Per-episode')
        
        # Add moving average
        window = min(5, len(rewards) // 2)
        if window > 1:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(rewards)), ma, 'r-', linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Episode Reward")
        ax.set_title("Reward Evolution During Training", fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3b: Episode length over time
        ax = axes[0, 1]
        lengths = np.array(self.episode_lengths)
        ax.plot(episodes, lengths, 'g-', alpha=0.5, label='Per-episode length')
        
        if window > 1:
            ma_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(lengths)), ma_len, 'darkgreen', linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Length (steps)")
        ax.set_title("Episode Length Evolution", fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3c: Entropy patterns over time
        ax = axes[1, 0]
        pattern_colors = {
            'confused': 'red',
            'adapting': 'orange',
            'learning': 'yellow',
            'overconfident': 'purple',
            'steady': 'green',
            'unknown': 'gray'
        }
        
        for ep, pattern in enumerate(self.entropy_patterns):
            color = pattern_colors.get(pattern, 'gray')
            ax.scatter(ep, 1, c=color, s=50, alpha=0.7)
        
        # Create legend
        legend_elements = [mpatches.Patch(facecolor=color, label=pattern) 
                          for pattern, color in pattern_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Entropy Pattern")
        ax.set_title("Behavioral Pattern Evolution", fontsize=11, fontweight='bold')
        ax.set_ylim([0.5, 1.5])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 3d: Pattern frequency distribution
        ax = axes[1, 1]
        pattern_counts = defaultdict(int)
        for pattern in self.entropy_patterns:
            pattern_counts[pattern] += 1
        
        patterns = list(pattern_counts.keys())
        counts = list(pattern_counts.values())
        colors = [pattern_colors.get(p, 'gray') for p in patterns]
        
        bars = ax.bar(range(len(patterns)), counts, color=colors, alpha=0.7)
        ax.set_xticks(range(len(patterns)))
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.set_ylabel("Number of Episodes")
        ax.set_title("Behavior Pattern Distribution", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_action_sequences(self):
        """Plot 4: Action sequence patterns ('ordering' learning)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 4a: First 10 episodes action sequences (timeline)
        ax = axes[0, 0]
        num_to_plot = min(10, len(self.action_sequences))
        
        action_to_idx = {action: i for i, action in enumerate(self.ACTION_MAP)}
        
        for ep_idx in range(num_to_plot):
            actions = self.action_sequences[ep_idx]
            action_indices = [action_to_idx[a] for a in actions]
            steps = np.arange(len(action_indices))
            ax.plot(steps, action_indices, 'o-', alpha=0.6, label=f'Ep {ep_idx}')
        
        ax.set_yticks(range(len(self.ACTION_MAP)))
        ax.set_yticklabels(self.ACTION_MAP)
        ax.set_xlabel("Step in Episode")
        ax.set_ylabel("Action Type")
        ax.set_title("Action Sequences: Early Training (Episodes 0-9)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Plot 4b: Last 10 episodes action sequences
        ax = axes[0, 1]
        start_ep = max(0, len(self.action_sequences) - 10)
        
        for ep_idx in range(start_ep, len(self.action_sequences)):
            actions = self.action_sequences[ep_idx]
            action_indices = [action_to_idx[a] for a in actions]
            steps = np.arange(len(action_indices))
            ax.plot(steps, action_indices, 'o-', alpha=0.6, label=f'Ep {ep_idx}')
        
        ax.set_yticks(range(len(self.ACTION_MAP)))
        ax.set_yticklabels(self.ACTION_MAP)
        ax.set_xlabel("Step in Episode")
        ax.set_ylabel("Action Type")
        ax.set_title(f"Action Sequences: Late Training (Episodes {start_ep}-{len(self.action_sequences)-1})", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Plot 4c: Action repetition analysis
        ax = axes[1, 0]
        early_repetition = []
        late_repetition = []
        
        for ep_idx in range(min(len(self.action_sequences) // 2, 20)):
            actions = self.action_sequences[ep_idx]
            # Count consecutive same actions
            reps = 0
            for i in range(len(actions) - 1):
                if actions[i] == actions[i+1]:
                    reps += 1
            early_repetition.append(reps / max(1, len(actions)-1))
        
        for ep_idx in range(max(0, len(self.action_sequences) - 20), len(self.action_sequences)):
            actions = self.action_sequences[ep_idx]
            reps = 0
            for i in range(len(actions) - 1):
                if actions[i] == actions[i+1]:
                    reps += 1
            late_repetition.append(reps / max(1, len(actions)-1))
        
        bp = ax.boxplot([early_repetition, late_repetition], 
                        labels=['Early Training', 'Late Training'],
                        patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_ylabel("Action Repetition Rate")
        ax.set_title("Does Agent Stop Repeating Actions?", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4d: First action patterns
        ax = axes[1, 1]
        first_actions_early = defaultdict(int)
        first_actions_late = defaultdict(int)
        
        for ep_idx in range(min(len(self.action_sequences) // 2, 25)):
            if self.action_sequences[ep_idx]:
                first_actions_early[self.action_sequences[ep_idx][0]] += 1
        
        for ep_idx in range(max(0, len(self.action_sequences) - 25), len(self.action_sequences)):
            if self.action_sequences[ep_idx]:
                first_actions_late[self.action_sequences[ep_idx][0]] += 1
        
        actions_set = set(self.ACTION_MAP)
        x = np.arange(len(actions_set))
        width = 0.35
        
        early_counts = [first_actions_early.get(a, 0) for a in self.ACTION_MAP]
        late_counts = [first_actions_late.get(a, 0) for a in self.ACTION_MAP]
        
        ax.bar(x - width/2, early_counts, width, label='Early', alpha=0.7)
        ax.bar(x + width/2, late_counts, width, label='Late', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(self.ACTION_MAP, rotation=45, ha='right')
        ax.set_ylabel("Frequency")
        ax.set_title("First Action Preference: Early vs Late Training", fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


def collect_trajectory_with_tracking(env, agent, monitor):
    """Collect one episode trajectory with detailed tracking."""
    obs = env.reset()
    trajectory = {
        "log_probs": [],
        "advantages": [],
        "rewards": [],
        "actions": [],
        "probs": [],
    }
    
    step_count = 0
    while True:
        # Select action
        action, log_prob, action_idx, probs = agent.select_action(obs)
        trajectory["log_probs"].append(log_prob)
        trajectory["actions"].append(action.type)
        trajectory["probs"].append(probs.detach().numpy())
        
        # Step environment
        obs, reward, done, info = env.step(action)
        trajectory["rewards"].append(reward)
        
        step_count += 1
        if step_count > 20:
            done = True
        
        if done:
            break
    
    # Extract learning signals from episode summary
    ep_summary = info.get("episode_summary", {})
    gae_advantages = ep_summary.get("gae_advantages", [])
    
    if not gae_advantages:
        gae_advantages = [0.0] * len(trajectory["rewards"])
    
    while len(gae_advantages) < len(trajectory["rewards"]):
        gae_advantages.append(0.0)
    gae_advantages = gae_advantages[:len(trajectory["rewards"])]
    
    trajectory["advantages"] = gae_advantages
    trajectory["episode_reward"] = sum(trajectory["rewards"])
    trajectory["episode_length"] = len(trajectory["rewards"])
    trajectory["entropy_pattern"] = ep_summary.get("entropy_behavior_pattern", "unknown")
    
    return trajectory


def run_visualization_training(num_episodes=100):
    """Run training with detailed tracking for visualization."""
    print("\n" + "="*70)
    print("POLICY BEHAVIOR VISUALIZATION WITH DETAILED TRACKING")
    print("="*70)
    print("\nCollecting data on:")
    print("  ✓ Action distribution evolution")
    print("  ✓ Advantage per action type")
    print("  ✓ Learning dynamics and entropy patterns")
    print("  ✓ Action sequence ordering patterns")
    print("="*70 + "\n")
    
    # Setup
    env = ArchitectEnv(task_id="easy")
    agent = PolicyGradientAgent(state_dim=128, hidden_dim=64, lr=5e-4)
    monitor = TrainingMonitor(num_episodes=num_episodes)
    
    print(f"{'Episode':<10} {'Length':<10} {'Reward':<12} {'Pattern':<15}")
    print("-" * 50)
    
    for episode in range(num_episodes):
        trajectory = collect_trajectory_with_tracking(env, agent, monitor)
        monitor.record_episode(trajectory, episode)
        
        if episode % 10 == 0 or episode == num_episodes - 1:
            print(f"{episode:<10} {trajectory['episode_length']:<10} "
                  f"{trajectory['episode_reward']:<12.3f} {trajectory['entropy_pattern']:<15}")
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Generate all plots
    fig1 = monitor.plot_action_distribution()
    print("✓ Action Distribution Visualization")
    
    fig2 = monitor.plot_advantage_per_action()
    print("✓ Advantage per Action Type Visualization")
    
    fig3 = monitor.plot_learning_dynamics()
    print("✓ Learning Dynamics Visualization")
    
    fig4 = monitor.plot_action_sequences()
    print("✓ Action Sequence Patterns Visualization")
    
    # Save figures
    fig1.savefig('artifacts/policy_action_distribution.png', dpi=150, bbox_inches='tight')
    print("\n📊 Saved: artifacts/policy_action_distribution.png")
    
    fig2.savefig('artifacts/policy_advantage_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: artifacts/policy_advantage_analysis.png")
    
    fig3.savefig('artifacts/policy_learning_dynamics.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: artifacts/policy_learning_dynamics.png")
    
    fig4.savefig('artifacts/policy_action_sequences.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: artifacts/policy_action_sequences.png")
    
    # Print statistics
    stats = monitor.compute_statistics()
    print("\n" + "="*70)
    print("BEHAVIOR INSIGHTS")
    print("="*70)
    print(f"\nAverage Episode Reward: {stats['avg_episode_reward']:.3f}")
    print(f"Average Episode Length: {stats['avg_episode_length']:.1f} steps")
    print(f"Average Advantage: {stats['avg_advantage']:.3f}")
    
    print("\nAction Quality Analysis:")
    print(f"{'Action':<25} {'Avg Advantage':<15} {'Utilization':<15} {'Impact':<15}")
    print("-" * 70)
    
    for action in monitor.ACTION_MAP:
        action_stat = stats["action_stats"][action]
        avg_adv = action_stat["avg_advantage"]
        count = action_stat["count"]
        impact = avg_adv * count
        
        status = "🟢 Useful" if avg_adv > 0.1 else ("🟡 Mixed" if avg_adv > -0.1 else "🔴 Harmful")
        
        print(f"{action:<25} {avg_adv:>7.3f}           {count:>5} times      {impact:>7.1f}  {status}")
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE - Open PNG files to view patterns")
    print("="*70 + "\n")
    
    plt.close('all')
    return monitor


if __name__ == "__main__":
    monitor = run_visualization_training(num_episodes=100)
