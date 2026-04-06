# Learning Signal Perspective: RL Training Integration

## Overview

The environment now provides **dense reward shaping** and **advantage signals** specifically designed for reinforcement learning algorithms. This enables you to train agents more effectively by:

1. **Dense Reward Breakdown** - See exactly which components of the reward signal drove behavior
2. **Advantage Signals** - Normalize rewards relative to a learned baseline for lower variance
3. **Component-Level Analysis** - Diagnose which reward signals were effective during training

---

## Dense Reward Components

Each step returns rewards broken down by component in `info["reward_components"]`:

```python
reward_components = {
    "information_gain":              float,  # Core reward: info gain from constraint discovery
    "exploration_reward":            float,  # Phase-specific: bonuses during exploration
    "refinement_reward":             float,  # Phase-specific: bonuses during refinement
    "belief_calibration_reward":     float,  # Reward for well-calibrated confidence
    "contradiction_handling_reward": float,  # Bonus for handling adversarial noise
    "efficiency_reward":             float,  # Reward for efficient question sequences
    "consistency_reward":            float,  # Reward for stable constraint reasoning
    "counterfactual_reward":         float,  # Penalty if better alternatives existed
    "step_penalties":                float,  # Negative: no progress, duplicates, unsafe
    "phase":                         str,    # "exploration" | "refinement" | "decision"
    "action_type":                   str,    # The action taken (ASK_*, FINALIZE, etc.)
    "total_step_reward":             float,  # Sum of all components (before weighting)
}
```

### Example Usage

```python
observation, reward, done, info = env.step(action)

# Inspect which signals drove the reward
components = info["reward_components"]
print(f"Information gain reward: {components['information_gain']:.3f}")
print(f"Efficiency bonus: {components['efficiency_reward']:.3f}")
print(f"Counterfactual penalty: {components['counterfactual_reward']:.3f}")

# Debug why an action got a low reward
step_reward = info["step_reward"]  # Before terminal scaling
print(f"Total step reward: {step_reward:.3f}")
```

---

## Advantage Signals for Policy Gradient Algorithms

The environment tracks a **running baseline** to compute advantage signals:

$$A(s,t) = R(s,t) - V(s)$$

Where:
- $R(s,t)$ = actual reward received at step $t$
- $V(s)$ = baseline (exponential moving average of past rewards)
- $A(s,t)$ = advantage signal

### Baseline Computation

The baseline is updated with exponential moving average (EMA):

$$V_t = (1 - \alpha) \cdot V_{t-1} + \alpha \cdot R(s,t)$$

Where $\alpha = 0.01$ (smoothing factor).

### Info Dict Entries

```python
info["advantage"]           # Raw advantage: reward - baseline
info["advantage_signal"] = {
    "baseline":              float,  # V(s) - running baseline
    "advantage_raw":         float,  # A(s,t) - absolute advantage
    "advantage_normalized":  float,  # Advantage / std(recent_rewards)
    "step_reward":           float,  # Actual reward received
    "global_step":           int,    # Cumulative step count (cross-episode)
}
info["rolling_baseline"]    # Current baseline value for inspection
```

### Why Use Advantage Signals?

**Problem**: Raw rewards have high variance
- Some steps are inherently easier than others
- Leads to high-variance policy gradients
- Causes training instability

**Solution**: Normalize by baseline
- Positive advantage → action better than average
- Negative advantage → action worse than average
- Reduces variance while preserving learning signal

### Example Usage (Policy Gradient Training)

```python
for episode in range(num_episodes):
    obs = env.reset()
    trajectory = []
    
    while True:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        # Collect advantage signals for training
        advantage = info["advantage"]
        trajectory.append({
            "action": action,
            "reward": info["step_reward"],
            "advantage": advantage,
            "baseline": info["baseline"],
        })
        
        if done:
            break
    
    # Train policy using advantages (lower variance!)
    policy_loss = compute_loss(trajectory, agent)
    policy_loss.backward()
```

---

## Component-Level Reward Analysis

### When Information Gain is the Main Signal

```python
components = info["reward_components"]
if components["information_gain"] > 0.2:
    print("Agent discovered multiple constraints")
elif components["information_gain"] > 0.05:
    print("Agent found 1-2 new constraints")
else:
    print("Agent didn't learn anything new")
```

### When Counterfactual Penalty Indicates Inefficiency

```python
# Negative counterfactual reward = better questions existed
if components["counterfactual_reward"] < -0.05:
    print("This question was suboptimal")
    print(f"Lost potential: {-components['counterfactual_reward']:.3f}")
```

### When Efficiency Signals Good Process

```python
# High efficiency reward = logical, non-repetitive questions
if components["efficiency_reward"] > 0.1:
    print("High-quality question sequence")
    print(f"Repeated questions: {info.get('repeated_questions', 0)}")
    print(f"Logical sequences: {info.get('logical_sequences', 0)}")
```

### Phase-Specific Signal Inspection

```python
phase = info["phase"]
if phase == "exploration":
    print(f"Exploration reward: {components['exploration_reward']:.3f}")
    print(f"Phase-specific bonus for discovering new constraints")
elif phase == "refinement":
    print(f"Refinement reward: {components['refinement_reward']:.3f}")
    print(f"Phase-specific bonus for clarifying constraints")
```

---

## RL Training Integration Checklist

### For Policy Gradient Methods (PPO, A3C, etc.)

- [x] Use `advantage` for policy gradient
- [x] Track baseline updates (EMA automatic)
- [x] Monitor `rolling_baseline` to detect distribution shift
- [x] Inspect `reward_components` to debug reward hacking

### For Value-Based Methods (DQN, etc.)

- [x] Use `step_reward` as bellman target
- [x] Track `baseline` as value function estimate
- [x] Use `advantage_normalized` for value error

### For Multi-Task RL

```python
# Track which components matter per task
task_signal_importance = {
    "easy": [("information_gain", 0.4), ("efficiency_reward", 0.2)],
    "hard": [("consistency_reward", 0.3), ("contradiction_handling_reward", 0.25)],
}
```

### For Curriculum Learning

```python
# Early training: emphasize exploration signals
# Late training: emphasize consistency and efficiency
if episode < warm_up_episodes:
    mask_components(["consistency_reward", "efficiency_reward"], weight=0.5)
else:
    mask_components(["exploration_reward"], weight=0.5)
```

---

## Monitoring Training

### Key Metrics to Track

```python
# Per episode
ep_total_reward = sum(info["step_reward"] for info in trajectory)
ep_avg_advantage = sum(info["advantage"] for info in trajectory) / len(trajectory)
ep_final_baseline = info["rolling_baseline"]

# Component contribution
component_sums = {
    k: sum(info["reward_components"].get(k, 0) for info in trajectory)
    for k in ["information_gain", "efficiency_reward", "consistency_reward"]
}

# Failure diagnosis
if done and not success:
    failure_analysis = trajectory[-1].get("failure_analysis")
    print(f"Failure type: {failure_analysis['failure_type']}")
```

### Expected Ranges

| Signal | Range | Interpretation |
|--------|-------|-----------------|
| `information_gain` | [0, 0.25] | New constraints discovered |
| `exploration_reward` | [-0.05, 0.15] | Exploration phase bonuses |
| `efficiency_reward` | [-0.1, 0.1] | Question sequence quality |
| `counterfactual_reward` | [-0.15, 0.0] | Penalty for suboptimal choices |
| `advantage` | [-0.5, 0.5] | Relative to baseline |

---

## Terminal (Episode-End) Learning Signals

At episode end, terminal rewards are also available but scaled by `terminal_reward_weight=0.5`:

```python
if info.get("oracle_score") is not None:
    # Oracle matching reward (scaled by coverage)
    terminal_reward = info["oracle_score"] * info["coverage"] * 0.5
    
    # Efficiency bonus
    terminal_reward += info["efficiency_reward"]
    
    # Quality bonuses
    if info.get("tradeoff_score") > 0.5:
        terminal_reward += 0.1  # Understood constraint tradeoffs
    
    print(f"Terminal reward: {terminal_reward:.3f}")
```

The terminal reward weight is explicitly **lower** (0.5) than step reward weight (1.0) to prevent agents from gaming the system with "slow but safe" strategies.

---

## Advanced: Reconstructing Episode Reward Flow

```python
episode_data = []
obs = env.reset()

while True:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    
    episode_data.append({
        "step": info["global_step"],
        "action": action.type,
        "components": info["reward_components"],
        "advantage": info["advantage"],
        "baseline": info["baseline"],
    })
    
    if done:
        break

# Analyze which components were most impactful
import pandas as pd
df = pd.DataFrame([
    {
        "step": d["step"],
        "info_gain": d["components"]["information_gain"],
        "efficiency": d["components"]["efficiency_reward"],
        "advantage": d["advantage"],
    }
    for d in episode_data
])
print(df.describe())
```

---

## Next Steps

1. **Use advantage signals in training** - Replace raw rewards with advantages for lower variance
2. **Monitor component contributions** - Track which signals drive agent behavior
3. **Adapt curriculum** - Weight components differently at different training stages
4. **Debug failures** - Use `reward_components` breakdown to identify why agents fail
5. **Validate learning** - Plot advantage trends to confirm policy is improving

