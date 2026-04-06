# Learning Signals: Practical RL Training Examples

## Quick Start: Using Advantage Signals in Policy Gradient Training

### Basic PPO-Style Update

```python
import numpy as np
from collections import deque

class RLAgentTrainer:
    def __init__(self, agent, env, learning_rate=1e-3):
        self.agent = agent
        self.env = env
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        
    def collect_trajectory(self, num_steps=30):
        """Collect one episode with learning signals."""
        obs = self.env.reset()
        trajectory = []
        
        while True:
            # Get action from agent
            action = self.agent.select_action(obs)
            obs, reward, done, info = self.env.step(action)
            
            # Extract learning signals
            trajectory.append({
                "obs": obs,
                "action": action,
                "step_reward": info["step_reward"],      # ← Dense reward
                "advantage": info["advantage"],          # ← Key signal!
                "baseline": info["baseline"],
                "components": info["reward_components"], # ← For debugging
            })
            
            if done:
                break
        
        return trajectory
    
    def train_episode(self):
        """Train on one episode using advantage signals."""
        trajectory = self.collect_trajectory()
        
        # Compute policy loss using advantages (lower variance!)
        policy_loss = 0.0
        for step in trajectory:
            # Forward pass
            action_logits = self.agent.policy_head(step["obs"])
            action_dist = torch.distributions.Categorical(logits=action_logits)
            log_prob = action_dist.log_prob(step["action"])
            
            # Policy gradient with advantage baseline
            advantage = step["advantage"]  # ← From environment!
            policy_loss += -log_prob * advantage  # Negative = gradient direction
        
        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()

# Usage
trainer = RLAgentTrainer(agent, env)
for episode in range(1000):
    loss = trainer.train_episode()
    if episode % 100 == 0:
        print(f"Episode {episode}: Loss = {loss:.4f}")
```

---

## Component-Level Reward Analysis

### Inspect Which Signals Drive Your Agent

```python
def analyze_episode(env, agent):
    """Collect one episode and analyze reward components."""
    obs = env.reset()
    component_sums = {}
    total_reward = 0
    
    while True:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        
        # Accumulate components
        for key, val in info["reward_components"].items():
            if isinstance(val, (int, float)):
                component_sums[key] = component_sums.get(key, 0) + val
        
        total_reward += info["step_reward"]
        
        if done:
            break
    
    # Print breakdown
    print("\n=== Reward Component Breakdown ===")
    print(f"Total reward: {total_reward:.3f}")
    print("\nComponent contributions:")
    for component, value in sorted(component_sums.items(), key=lambda x: abs(x[1]), reverse=True):
        if isinstance(value, float):
            pct = 100 * value / max(abs(total_reward), 0.01)
            print(f"  {component:30s}: {value:7.3f} ({pct:5.1f}%)")
    
    return component_sums, total_reward

# Run analysis
components, total = analyze_episode(env, agent)
```

Output example:
```
=== Reward Component Breakdown ===
Total reward: 2.15

Component contributions:
  information_gain                  :   1.230 ( 57.2%)
  efficiency_reward                 :   0.450 ( 20.9%)
  belief_calibration_reward         :   0.320 ( 14.9%)
  consistency_reward                :   0.100 (  4.7%)
  counterfactual_reward             :  -0.140 ( -6.5%)
```

---

## Advantage Signal Monitoring

### Track Baseline Evolution During Training

```python
class AdvantageMonitor:
    def __init__(self):
        self.baseline_history = []
        self.advantage_history = []
        self.reward_history = []
    
    def update(self, info):
        """Track one step."""
        self.baseline_history.append(info["baseline"])
        self.advantage_history.append(info["advantage"])
        self.reward_history.append(info["step_reward"])
    
    def episode_summary(self):
        """Print episode-level statistics."""
        if not self.reward_history:
            return
        
        avg_reward = np.mean(self.reward_history)
        avg_advantage = np.mean(self.advantage_history)
        final_baseline = self.baseline_history[-1]
        
        print(f"Avg reward:     {avg_reward:7.3f}")
        print(f"Avg advantage:  {avg_advantage:7.3f}")
        print(f"Final baseline: {final_baseline:7.3f}")
        
        # Check for distribution shift
        if final_baseline > 0.5:
            print("⚠️  Baseline rising - agent getting too good or rewards inflating")
        if final_baseline < -0.3:
            print("⚠️  Baseline dropping - increasing task difficulty?")
    
    def plot_trends(self):
        """Visualize baseline and advantage evolution."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.reward_history, label="Step reward", alpha=0.7)
        plt.axhline(np.mean(self.reward_history), label="Mean", linestyle="--")
        plt.legend()
        plt.title("Step Rewards Over Episode")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        
        plt.subplot(1, 2, 2)
        plt.plot(self.advantage_history, label="Advantage", alpha=0.7)
        plt.axhline(0, linestyle="--", color="red")
        plt.legend()
        plt.title("Advantage Signals Over Episode")
        plt.xlabel("Step")
        plt.ylabel("Advantage")
        
        plt.tight_layout()
        plt.savefig("advantage_trends.png")
        plt.show()

# Usage
for episode in range(100):
    monitor = AdvantageMonitor()
    obs = env.reset()
    
    while True:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        monitor.update(info)
        
        if done:
            monitor.episode_summary()
            if episode % 10 == 0:
                monitor.plot_trends()
            break
```

---

## Curriculum Learning with Component Weights

### Dynamically Adjust Which Signals to Emphasize

```python
class CurriculumAgent:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.episode = 0
    
    def get_component_weights(self):
        """Dynamically adjust which components matter."""
        if self.episode < 100:
            # Early training: learn to discover constraints
            return {
                "information_gain": 1.0,
                "exploration_reward": 1.0,
                "efficiency_reward": 0.5,  # Less important
                "consistency_reward": 0.2,  # Not yet
            }
        elif self.episode < 500:
            # Mid training: learn efficiency
            return {
                "information_gain": 1.0,
                "efficiency_reward": 1.0,  # Now important
                "consistency_reward": 0.5,
                "counterfactual_reward": 1.0,  # Check against alternatives
            }
        else:
            # Late training: all signals equal
            return {
                "information_gain": 1.0,
                "efficiency_reward": 1.0,
                "consistency_reward": 1.0,
                "counterfactual_reward": 1.0,
            }
    
    def adjust_reward(self, component_dict):
        """Mask/weight components based on curriculum."""
        weights = self.get_component_weights()
        adjusted_reward = 0.0
        
        for key, weight in weights.items():
            if key in component_dict:
                adjusted_reward += component_dict[key] * weight
        
        return adjusted_reward
    
    def train_episode(self):
        """Train with curriculum-adjusted rewards."""
        obs = self.env.reset()
        total_loss = 0
        
        while True:
            action = self.agent.select_action(obs)
            obs, reward, done, info = self.env.step(action)
            
            # Adjust reward based on curriculum stage
            curriculum_reward = self.adjust_reward(info["reward_components"])
            
            # Use curriculum reward instead of raw reward
            loss = self.agent.update(curriculum_reward)
            total_loss += loss
            
            if done:
                break
        
        self.episode += 1
        return total_loss

# Usage
trainer = CurriculumAgent(agent, env)
for _ in range(1000):
    loss = trainer.train_episode()
```

---

## Failure Mode Detection Using Components

### Identify Why Your Agent Fails

```python
def diagnose_failure(env, agent, num_trials=10):
    """Run multiple episodes and detect failure patterns."""
    failure_patterns = {
        "weak_exploration": 0,
        "poor_efficiency": 0,
        "inconsistent": 0,
        "missed_tradeoffs": 0,
    }
    
    for trial in range(num_trials):
        obs = env.reset()
        episode_data = []
        
        while True:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            episode_data.append(info)
            
            if done:
                break
        
        # Analyze final episode
        final_info = episode_data[-1]
        
        # Check for failure signatures
        if final_info["reward_components"]["information_gain"] < 0.1:
            failure_patterns["weak_exploration"] += 1
        
        if final_info["reward_components"]["efficiency_reward"] < 0:
            failure_patterns["poor_efficiency"] += 1
        
        if final_info["reward_components"]["consistency_reward"] < 0:
            failure_patterns["inconsistent"] += 1
        
        if final_info.get("failure_analysis") is not None:
            failure_analysis = final_info["failure_analysis"]
            if "reasoning" in str(failure_analysis.get("failure_type", "")):
                failure_patterns["missed_tradeoffs"] += 1
    
    # Report findings
    print("\n=== Failure Mode Analysis ===")
    for pattern, count in sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / num_trials
        print(f"  {pattern:25s}: {count:2d}/{num_trials} ({pct:5.1f}%)")
    
    return failure_patterns

# Usage
diagnose_failure(env, agent)
```

---

## Value Function Training with Baselines

### Train Separate Value Function Alongside Policy

```python
class ActorCriticAgent:
    def __init__(self, agent, critic_network, env):
        self.agent = agent
        self.critic = critic_network
        self.env = env
        self.optimizer_actor = torch.optim.Adam(agent.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(critic_network.parameters(), lr=1e-3)
    
    def train_episode(self):
        """Train actor and critic using environment baselines."""
        obs = self.env.reset()
        trajectory = []
        
        while True:
            action = self.agent.select_action(obs)
            obs, reward, done, info = self.env.step(action)
            
            trajectory.append({
                "obs": obs,
                "action": action,
                "reward": info["step_reward"],
                "advantage": info["advantage"],           # ← From env!
                "baseline": info["baseline"],            # ← From env!
            })
            
            if done:
                break
        
        # Train critic to predict better baselines
        critic_loss = 0
        for step in trajectory:
            predicted_value = self.critic(step["obs"])
            target_value = step["baseline"]  # Use env's baseline as target
            critic_loss += (predicted_value - target_value) ** 2
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # Train actor with advantage signals
        actor_loss = 0
        for step in trajectory:
            action_logits = self.agent.policy_head(step["obs"])
            action_dist = torch.distributions.Categorical(logits=action_logits)
            log_prob = action_dist.log_prob(step["action"])
            
            # Use advantage from environment (already baseline-normalized!)
            actor_loss += -log_prob * step["advantage"]
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        return critic_loss.item(), actor_loss.item()

# Usage
trainer = ActorCriticAgent(agent, critic, env)
for episode in range(1000):
    critic_loss, actor_loss = trainer.train_episode()
    if episode % 100 == 0:
        print(f"Episode {episode}: Critic={critic_loss:.4f}, Actor={actor_loss:.4f}")
```

---

## Summary: Key Integration Points

| Component | Use For | Example |
|-----------|---------|---------|
| `step_reward` | Direct reward signal | summing episode rewards |
| `advantage` | Policy gradient | `-log_prob * advantage` |
| `baseline` | Value function target | `(pred - baseline)^2` |
| `reward_components` | Debugging | identifying which signals drive behavior |
| `global_step` | Curriculum scheduling | adjusting weights based on step count |

All of these are automatically computed and provided in `info` at each step!

