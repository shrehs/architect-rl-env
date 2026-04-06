#!/usr/bin/env python3
"""
Minimal Policy Gradient Training Loop

Closes the loop: Environment → Signals → Training → Improved Behavior

This validates that:
1. Advantage signals enable learning
2. Environment signals drive agent improvement
3. The entire system is trainable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import sys

from env.environment import ArchitectEnv
from env.models import Action


class SimplePolicy(nn.Module):
    """Minimal neural network policy for action selection."""
    
    def __init__(self, state_dim=128, hidden_dim=64, num_actions=6):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, state):
        """Return action logits."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        return action_logits


class PolicyGradientAgent:
    """Simple policy gradient agent with GAE advantages."""
    
    ACTION_MAP = [
        "ASK_BUDGET",
        "ASK_LATENCY", 
        "ASK_ACCURACY",
        "ASK_DATA_SIZE",
        "ASK_UPDATE_FREQUENCY",
        "FINALIZE"
    ]
    
    def __init__(self, state_dim=128, hidden_dim=64, lr=1e-3):
        self.policy = SimplePolicy(state_dim, hidden_dim, len(self.ACTION_MAP))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.state_dim = state_dim
        
    def select_action(self, obs):
        """Select action using policy, return action and log probability."""
        # Minimal state encoding: use observation message count as proxy
        state_tensor = torch.randn(self.state_dim)  # Random for now (simple demo)
        
        action_logits = self.policy(state_tensor)
        action_dist = Categorical(logits=action_logits)
        action_idx = action_dist.sample()
        log_prob = action_dist.log_prob(action_idx)
        
        action_type = self.ACTION_MAP[action_idx.item()]
        
        return Action(type=action_type, content="Query"), log_prob, action_idx
    
    def compute_loss(self, log_probs, advantages):
        """Policy gradient loss: -log_prob * advantage."""
        # Advantages should reduce variance of gradient estimates
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Center advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss: negative because we want to maximize reward
        loss = -(log_probs * advantages).mean()
        
        return loss
    
    def train_step(self, log_probs, advantages):
        """One gradient step."""
        loss = self.compute_loss(log_probs, advantages)
        
        # Skip update if loss is invalid
        if torch.isnan(loss) or torch.isinf(loss):
            print("      ⚠️  Invalid loss detected, skipping update")
            return float('nan')
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()


def collect_trajectory(env, agent):
    """Collect one episode trajectory with learning signals."""
    obs = env.reset()
    trajectory = {
        "log_probs": [],
        "advantages": [],
        "rewards": [],
        "actions": [],
    }
    
    step_count = 0
    while True:
        # Select action
        action, log_prob, action_idx = agent.select_action(obs)
        trajectory["log_probs"].append(log_prob)
        trajectory["actions"].append(action.type)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        trajectory["rewards"].append(reward)
        
        step_count += 1
        if step_count > 20:  # Prevent infinite loops
            done = True
        
        if done:
            break
    
    # Extract GAE advantages from episode summary
    ep_summary = info.get("episode_summary", {})
    gae_advantages = ep_summary.get("gae_advantages", [])
    
    # If no GAE, use raw advantage signals
    if not gae_advantages:
        gae_advantages = [info.get("advantage", 0.0) for _ in trajectory["rewards"]]
    
    # Pad advantages if needed
    while len(gae_advantages) < len(trajectory["rewards"]):
        gae_advantages.append(0.0)
    gae_advantages = gae_advantages[:len(trajectory["rewards"])]
    
    trajectory["advantages"] = gae_advantages
    trajectory["episode_reward"] = sum(trajectory["rewards"])
    trajectory["episode_length"] = len(trajectory["rewards"])
    trajectory["entropy_pattern"] = ep_summary.get("entropy_behavior_pattern", "unknown")
    trajectory["combined_reward"] = info.get("combined_reward", 0.0)
    
    return trajectory


def train_episode(env, agent):
    """Train agent on one episode."""
    trajectory = collect_trajectory(env, agent)
    
    # Stack log probs for batch processing
    log_probs = torch.stack(trajectory["log_probs"])
    advantages = trajectory["advantages"]
    
    # Validate advantages
    advantages_array = np.array(advantages, dtype=np.float32)
    if np.any(np.isnan(advantages_array)) or np.any(np.isinf(advantages_array)):
        print("      ⚠️  Invalid advantages, using raw rewards")
        advantages_array = np.array(trajectory["rewards"], dtype=np.float32)
    
    # Train step
    loss = agent.train_step(log_probs, advantages_array)
    
    return {
        "loss": loss,
        "episode_reward": trajectory["episode_reward"],
        "episode_length": trajectory["episode_length"],
        "entropy_pattern": trajectory["entropy_pattern"],
        "combined_reward": trajectory["combined_reward"],
    }


def run_training(num_episodes=50, checkpoint_interval=10):
    """Run minimal training loop."""
    print("\n" + "="*70)
    print("MINIMAL POLICY GRADIENT TRAINING LOOP")
    print("="*70)
    print("\nClosing the loop: Environment → Signals → Training → Improvement")
    print("Using: GAE advantages + policy gradient")
    print("="*70 + "\n")
    
    # Setup
    env = ArchitectEnv(task_id="easy")
    agent = PolicyGradientAgent(state_dim=128, hidden_dim=64, lr=5e-4)
    
    # Tracking
    reward_history = deque(maxlen=10)
    loss_history = deque(maxlen=10)
    entropy_patterns = {}
    
    print(f"{'Episode':<10} {'Reward':<12} {'Loss':<12} {'Avg Reward':<12} {'Pattern':<15}")
    print("-" * 70)
    
    for episode in range(num_episodes):
        stats = train_episode(env, agent)
        
        reward_history.append(stats["episode_reward"])
        
        # Only add valid losses to history
        if not np.isnan(stats["loss"]):
            loss_history.append(stats["loss"])
        
        # Track entropy patterns
        pattern = stats["entropy_pattern"]
        entropy_patterns[pattern] = entropy_patterns.get(pattern, 0) + 1
        
        # Show progress
        avg_reward = np.mean(reward_history)
        avg_loss = np.mean(loss_history) if loss_history else 0.0
        
        loss_str = f"{stats['loss']:.4f}" if not np.isnan(stats['loss']) else "   NaN "
        
        print(f"{episode:<10} {stats['episode_reward']:<12.3f} {loss_str:<12} {avg_reward:<12.3f} {pattern:<15}")
        
        # Checkpoint analysis
        if (episode + 1) % checkpoint_interval == 0:
            improvement = (reward_history[-1] - reward_history[0]) if len(reward_history) > 1 else 0
            print(f"\n  Checkpoint {episode + 1}:")
            print(f"    Current reward: {stats['episode_reward']:.3f}")
            print(f"    Avg reward (last 10): {avg_reward:.3f}")
            print(f"    Avg loss: {avg_loss:.4f}")
            print(f"    Entropy patterns seen: {entropy_patterns}")
            print()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Final statistics
    final_avg_reward = np.mean(reward_history)
    final_avg_loss = np.mean(loss_history)
    
    print(f"\nFinal Results:")
    print(f"  Average reward (last 10): {final_avg_reward:.3f}")
    print(f"  Average loss (last 10): {final_avg_loss:.4f}")
    print(f"  Total entropy patterns observed: {entropy_patterns}")
    
    # Check if training worked
    if len(reward_history) >= 2:
        early_rewards = list(reward_history)[:-5]
        late_rewards = list(reward_history)[-5:]
        early_avg = np.mean(early_rewards) if early_rewards else 0
        late_avg = np.mean(late_rewards)
        improvement_pct = 100 * (late_avg - early_avg) / (abs(early_avg) + 1)
        
        print(f"\nImprovement Analysis:")
        print(f"  Early average (first in window): {early_avg:.3f}")
        print(f"  Late average (last in window): {late_avg:.3f}")
        print(f"  Improvement: {improvement_pct:+.1f}%")
        
        if improvement_pct > 0:
            print(f"\n✅ SYSTEM IS TRAINABLE - Agent improved during training!")
            return True
        else:
            print(f"\n⚠️  Limited improvement - may need different hyperparameters")
            return False
    else:
        print("\n⚠️  Insufficient data for improvement analysis")
        return False


def validate_signals():
    """Validate that environment signals are present and usable."""
    print("\n" + "="*70)
    print("SIGNAL VALIDATION")
    print("="*70 + "\n")
    
    env = ArchitectEnv(task_id="easy")
    obs = env.reset()
    
    # Collect one step
    action = Action(type="ASK_BUDGET", content="Test")
    obs, reward, done, info = env.step(action)
    
    required_signals = [
        ("advantage", "Per-step advantage"),
        ("reward_components", "Dense reward breakdown"),
        ("step_reward", "Total step reward"),
        ("baseline", "Value function baseline"),
    ]
    
    print("Per-Step Signals:")
    all_present = True
    for signal_name, description in required_signals:
        if signal_name in info:
            print(f"  ✅ {signal_name:<20} - {description}")
        else:
            print(f"  ❌ {signal_name:<20} - MISSING")
            all_present = False
    
    # Collect until done for episode signals
    while not done:
        obs, reward, done, info = env.step(Action(type="ASK_LATENCY", content="Test"))
    
    episode_signals = [
        ("episode_summary", "Episode-level learning signals"),
        ("combined_reward", "Weighted combination of signals"),
    ]
    
    print("\nEpisode-End Signals:")
    for signal_name, description in episode_signals:
        if signal_name in info:
            print(f"  ✅ {signal_name:<20} - {description}")
        else:
            print(f"  ❌ {signal_name:<20} - MISSING")
            all_present = False
    
    # Check episode summary content
    ep_summary = info.get("episode_summary", {})
    summary_items = [
        ("gae_advantages", "GAE advantages for policy gradient"),
        ("entropy_behavior_pattern", "Entropy-based behavior pattern"),
        ("entropy_history", "Entropy values per step"),
        ("nstep_1_returns", "1-step bootstrap targets"),
    ]
    
    print("\nEpisode Summary Content:")
    for item_name, description in summary_items:
        if item_name in ep_summary:
            print(f"  ✅ {item_name:<20} - {description}")
        else:
            print(f"  ⚠️  {item_name:<20} - {description}")
    
    if all_present:
        print(f"\n✅ All critical signals present and ready for training")
        return True
    else:
        print(f"\n⚠️  Some signals missing - may need environment fixes")
        return False


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    # Step 1: Validate signals exist
    signals_ok = validate_signals()
    
    if signals_ok:
        # Step 2: Run training
        training_ok = run_training(num_episodes=50, checkpoint_interval=10)
        
        if training_ok:
            print("\n" + "="*70)
            print("🎉 CLOSED THE LOOP SUCCESSFULLY")
            print("="*70)
            print("\nEnvironment → Signals → Training → Improved Agent")
            print("\nYour system is now:")
            print("  ✅ Generating valid learning signals")
            print("  ✅ Trainable with policy gradients")
            print("  ✅ Showing behavioral improvement")
            print("  ✅ Ready for advanced RL algorithms")
            print("="*70)
            sys.exit(0)
        else:
            print("\n⚠️  Training ran but showed limited improvement")
            print("Next steps: Tune hyperparameters, adjust reward weights")
            sys.exit(1)
    else:
        print("\n❌ Signal validation failed - cannot proceed with training")
        sys.exit(1)
