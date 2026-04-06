# Advanced RL Features: GAE, N-Step Returns, Action Entropy

## Overview

The environment now provides advanced reinforcement learning features designed for production-grade RL training:

1. **Generalized Advantage Estimation (GAE)** - Reduces variance while maintaining unbiased targets
2. **N-Step Returns** - Multi-step bootstrapping with flexible horizons
3. **Action Entropy Tracking** - Monitor exploration patterns and identify behavioral modes
4. **Episode-Level Learning Signals** - Comprehensive trajectory analysis tools

---

## 1. Generalized Advantage Estimation (GAE)

### What is GAE?

GAE combines the advantages of TD learning (low variance) and Monte Carlo returns (unbiased):

$$A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where:
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (TD residual)
- $\lambda \in [0, 1]$ interpolates between TD and MC
- $\gamma$ is the discount factor

### Using GAE in Training

```python
# Access GAE advantages (computed automatically at episode end)
info = env.step(action)
if info["done"]:
    ep_summary = info["episode_summary"]
    gae_advantages = ep_summary["gae_advantages"]  # List of advantages
    gae_lambda = ep_summary["gae_lambda"]  # λ parameter used
    
    # Use in policy gradient
    for t, advantage in enumerate(gae_advantages):
        loss = -log_prob[t] * advantage
```

### Lambda Parameter Interpretation

| λ Value | Behavior | Bias | Variance | Use Case |
|---------|----------|------|----------|----------|
| 0.0 | 1-step TD | High | Low | Stable, simple |
| 0.5 | 2-step mixing | Medium | Medium | Balanced |
| 0.95 | **Default** | Low | Medium | Best for most tasks |
| 1.0 | Full Monte Carlo | None | High | Small batch sizes |

### Adjusting GAE Lambda

```python
env = ArchitectEnv(task_id="easy")
env.gae_lambda = 0.99  # Longer horizon (higher variance)

# Or per-episode
env.gae_lambda = 0.8 if episode < 100 else 0.95  # Curriculum
```

---

## 2. N-Step Returns

### What are N-Step Returns?

N-step returns bootstrap from the value function at step t+n:

$$G_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n})$$

This balances between:
- **n=1** (one-step): High bias, low variance (pure TD)
- **n=∞** (full trajectory): Unbiased but high variance (Monte Carlo)
- **n=3,5** (moderate): Practical balance

### Accessing N-Step Returns

```python
info = env.step(action)
if info["done"]:
    ep_summary = info["episode_summary"]
    
    # Environment computes 1-step, 3-step, and 5-step returns
    one_step = ep_summary["nstep_1_returns"]    # One-step TD targets
    three_step = ep_summary["nstep_3_returns"]  # 3-step bootstrap
    five_step = ep_summary["nstep_5_returns"]   # 5-step bootstrap
    
    # Use for value function training
    for t in range(len(one_step)):
        value_loss = (predicted_value[t] - five_step[t]) ** 2
```

### When to Use Each

```python
# Early training: use 1-step (stable)
if episode < 100:
    target_return = nstep_returns[1]
# Mid training: use 3-step (balanced)
elif episode < 500:
    target_return = nstep_returns[3]
# Late training: use 5-step (less bias)
else:
    target_return = nstep_returns[5]
```

---

## 3. Action Entropy Tracking

### What Does Entropy Measure?

Shannon entropy measures action distribution diversity:

$$H(\pi) = -\sum_a \pi(a) \log \pi(a)$$

- **High entropy** (≈1.0): Agent explores many actions uniformly
- **Low entropy** (≈0.0): Agent repeats same action (deterministic/stuck)
- **Medium entropy** (≈0.5): Mixed exploration-exploitation

### Accessing Entropy Signals

```python
info = env.step(action)

# Per-step entropy
entropy = info["action_entropy"]  # Raw Shannon entropy
entropy_info = info["entropy_info"]  # Detailed breakdown

print(f"Entropy: {entropy:.3f}")  # 0.0 to ~1.95
print(f"Normalized: {entropy_info['normalized_entropy']:.3f}")  # 0.0 to 1.0
print(f"Unique actions: {entropy_info['num_unique_actions']}")
print(f"Most common: {entropy_info['most_common_action']} ({entropy_info['most_common_pct']:.1f}%)")

# Exploration detection
is_exploring = entropy_info["is_exploring"]  # True if high entropy
is_stuck = entropy_info["is_deterministic"]  # True if low entropy
```

### Episode-level Entropy

```python
if info["done"]:
    ep_summary = info["episode_summary"]
    
    # Overall episode entropy
    total_entropy = ep_summary["action_entropy"]
    entropy_details = ep_summary["entropy_info"]
    
    # Action distribution
    dist = entropy_details["action_distribution"]
    print(f"Distribution: {dist}")
    # Output: {'ASK_BUDGET': 0.4, 'ASK_LATENCY': 0.2, ...}
```

### Using Entropy for Training

```python
# Encourage exploration in early stages
if entropy < 0.3:  # Too deterministic
    print("Agent stuck in local policy - increase entropy bonus")
    exploration_bonus = 0.1  # Add bonus

# Exploit reliably discovered policies
if entropy > 0.8:  # Too random
    print("Agent exploring too much - reduce entropy bonus")
    exploration_bonus = 0.0  # No bonus
```

---

## 4. Episode Summary: Complete Learning Signals

### What's in the Episode Summary?

```python
if info["done"]:
    ep_summary = info["episode_summary"]
    
    # GAE-based advantages
    {
        "gae_advantages": [float],  # List per timestep
        "gae_lambda": 0.95,
    }
    
    # N-step returns (multiple horizons)
    {
        "nstep_1_returns": [float],  # One-step TD
        "nstep_3_returns": [float],  # Three-step bootstrap
        "nstep_5_returns": [float],  # Five-step bootstrap
    }
    
    # Action diversity
    {
        "action_entropy": float,      # Total Shannon entropy
        "entropy_info": {
            "action_distribution": {...},
            "most_common_action": "ASK_BUDGET",
            "most_common_pct": 40.0,
            "is_exploring": True,
            "is_deterministic": False,
        }
    }
    
    # Episode statistics
    {
        "episode_total_reward": float,
        "episode_avg_reward": float,
        "episode_max_reward": float,
        "episode_min_reward": float,
        "episode_length": int,
        "total_actions_taken": int,
    }
```

---

## 5. Practical Training Examples

### Actor-Critic with GAE

```python
def train_episode_with_gae(env, actor, critic):
    """Train using GAE-based policy gradient."""
    obs = env.reset()
    trajectory = []
    
    # Collect episode
    while True:
        action = actor.select_action(obs)
        obs, reward, done, info = env.step(action)
        trajectory.append(info)
        if done:
            break
    
    # Get GAE advantages at episode end
    ep_summary = trajectory[-1].get("episode_summary", {})
    gae_advantages = ep_summary.get("gae_advantages", [])
    
    # Train actor with GAE advantages
    actor_loss = 0
    for t, step_info in enumerate(trajectory[:-1]):  # Exclude last step
        action_logits = actor.policy(step_info["obs"])
        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = action_dist.log_prob(step_info["action"])
        
        # GAE advantage reduces variance!
        advantage = gae_advantages[t] if t < len(gae_advantages) else 0
        actor_loss += -log_prob * advantage
    
    # Train critic with n-step returns
    nstep_returns = ep_summary.get("nstep_5_returns", [])
    critic_loss = 0
    for t, step_info in enumerate(trajectory[:-1]):
        predicted_value = critic(step_info["obs"])
        target_value = nstep_returns[t] if t < len(nstep_returns) else 0
        critic_loss += (predicted_value - target_value) ** 2
    
    # Update
    (actor_loss + critic_loss).backward()
    optimizer.step()
```

### Entropy-Adaptive Curriculum

```python
class AdaptiveCurriculum:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.entropy_history = []
    
    def train_episode(self, episode_num):
        obs = self.env.reset()
        
        while True:
            action = self.agent.select_action(obs)
            obs, reward, done, info = self.env.step(action)
            
            if done:
                ep_summary = info.get("episode_summary", {})
                entropy = ep_summary.get("action_entropy", 0)
                self.entropy_history.append(entropy)
                
                # Adjust curriculum based on entropy
                if entropy < 0.3:
                    # Agent is too deterministic - increase exploration bonus
                    self.agent.exploration_bonus *= 1.1
                elif entropy > 0.8:
                    # Agent is too random - decrease exploration
                    self.agent.exploration_bonus *= 0.9
                
                break
```

### Multi-Horizon Value Targets

```python
# Use different n-step returns based on task difficulty
def get_value_target(ep_summary, task_difficulty):
    if task_difficulty == "easy":
        return ep_summary["nstep_1_returns"]  # Low variance
    elif task_difficulty == "medium":
        return ep_summary["nstep_3_returns"]  # Balanced
    else:
        return ep_summary["nstep_5_returns"]  # Low bias
```

---

## 6. Configuration & Tuning

### Key Parameters

```python
env = ArchitectEnv(task_id="easy")

# GAE lambda interpolation
env.gae_lambda = 0.95  # Default: balanced

# Discount factor
env.gamma = 0.99  # Standard for most RL

# These control episode behavior
env.step_reward_weight = 1.0      # Weight on per-step rewards
env.terminal_reward_weight = 0.5  # Weight on episode-end rewards
```

### Recommendation by Task Type

| Task | λ | n-step | Why |
|------|---|--------|-----|
| **Quick tasks** (easy) | 0.95 | 1-3 | Fast convergence, high variance tolerable |
| **Medium complexity** | 0.95 | 3-5 | Balanced bias-variance tradeoff |
| **Hard/long horizons** | 0.99 | 5 | Low bias critical, variance acceptable |
| **Exploration-heavy** | 0.90 | 1 | Stable, emphasize immediate signals |

---

## 7. Monitoring & Debugging

### Check for Common Issues

```python
ep_summary = info["episode_summary"]

# 1. GAE advantages should be roughly centered at 0 with reasonable variance
gae_advs = ep_summary["gae_advantages"]
print(f"GAE mean: {np.mean(gae_advs):.3f}")  # Should be ~0
print(f"GAE std: {np.std(gae_advs):.3f}")   # Should be reasonable

# 2. N-step returns should increase smoothly
for n in [1, 3, 5]:
    returns = ep_summary[f"nstep_{n}_returns"]
    print(f"n={n}: min={min(returns):.3f}, max={max(returns):.3f}")

# 3. Entropy should show exploration progression
entropy = ep_summary["action_entropy"]
if entropy < 0.1:
    print("⚠️ Agent very deterministic (stuck?) - check reward design")
elif entropy > 1.5:
    print("⚠️ Agent very random - check exploration bonus")
else:
    print(f"✓ Good exploration level: {entropy:.3f}")
```

### Plotting for Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_signals(episodes_data):
    """Plot key signals across episodes."""
    
    gae_stds = [np.std(ep["gae_advantages"]) for ep in episodes_data]
    entropies = [ep["action_entropy"] for ep in episodes_data]
    rewards = [ep["episode_total_reward"] for ep in episodes_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # GAE variance over time
    axes[0].plot(gae_stds)
    axes[0].set_title("GAE Advantage Variance")
    axes[0].set_ylabel("Std Dev")
    
    # Action entropy over time
    axes[1].plot(entropies)
    axes[1].axhline(0.5, linestyle="--", color="red", alpha=0.5)
    axes[1].set_title("Action Entropy")
    axes[1].set_ylabel("Entropy")
    
    # Reward progress
    axes[2].plot(rewards)
    axes[2].set_title("Episode Reward")
    axes[2].set_ylabel("Total Reward")
    
    plt.tight_layout()
    plt.savefig("learning_signals.png")
```

---

## Summary: Which Feature To Use When

| Feature | When | Why |
|---------|------|-----|
| **GAE (λ=0.95)** | Always | Reduces variance, almost always beneficial |
| **1-step returns** | Unstable training | Low variance, steady learning |
| **3-step returns** | Balanced task | Good default choice |
| **5-step returns** | Long horizons | Better credit assignment |
| **Entropy tracking** | Debug stuck agents | Identify exploration problems |
| **Episode summary** | Analysis & tuning | Complete view of trajectory quality |

All of these are **automatically computed** and returned in `info["episode_summary"]` at episode end!

