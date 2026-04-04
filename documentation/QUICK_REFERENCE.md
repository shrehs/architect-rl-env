# Quick Reference: Complete Feature Set (Updated April 4, 2026)

## 🚨 CRITICAL: Oracle Gradient Now Restored!

**What Changed:** Oracle scoring is now continuous (0.0-1.0), not binarized

**Impact on Results:**
- Random agents: Now correctly score ~0.0 (not lucky 1.0 anymore)
- Heuristic agents: Show degradation with noise (0.6-1.0 range)
- Improved agents: Show robustness (0.4-1.0 range)
- Success threshold: Raised from ≥0.6 to ≥0.8 (much harder to achieve!)

**To understand this fix:** See [ORACLE_GRADIENT_RESTORATION.md](ORACLE_GRADIENT_RESTORATION.md)

---

## 🎯 TL;DR - Run This

### Default (Balanced, Recommended)
```bash
python experiments/run_evaluation.py --episodes 30 --task easy
```
- Uses `exploration_alpha=1.0` (smoothing + temperature enabled)
- Balanced exploration/exploitation
- Reproduces original behavior with smoothing improvement

### Explore More (Rare Path Discovery)
```bash
python experiments/run_evaluation.py --episodes 50 --task easy --exploration-alpha 1.5
```
- Stronger push to undiscovered architectures
- Good for research and diversity analysis
- Takes longer to converge

### Converge Faster (Stable Training)  
```bash
python experiments/run_evaluation.py --episodes 30 --task easy --exploration-alpha 0.8
```
- Softer exploration incentive
- Quicker convergence
- Less diversity

---

## 📊 Full Parameter List

```bash
python experiments/run_evaluation.py \
  --episodes 100              # Default: 100. Episodes per (agent, mode)
  --task easy                 # Default: easy. Choices: easy|medium|hard
  --out-dir artifacts/eval    # Default: artifacts/evaluation
  --exploration-alpha 1.0     # Default: 1.0. Temperature (new!)
```

---

## 🧪 Common Experiments

### 1. Quick Test (5 min)
```bash
python experiments/run_evaluation.py --episodes 5 --task easy
# Result: 45 total episodes (3 agents × 3 modes × 5 episodes)
```

### 2. Benchmark Run (default settings)
```bash
python experiments/run_evaluation.py --episodes 30 --task easy
# Result: 270 episodes, publishable results
```

### 3. Comprehensive Benchmark
```bash
python experiments/run_evaluation.py --episodes 100 --task easy
# Result: 900 episodes, high-quality metrics
```

### 4. Compare Alpha Values (Ablation)
```bash
# Run once for each temperature
for alpha in 0.5 1.0 1.5 2.0; do
    echo "Running alpha=$alpha..."
    python experiments/run_evaluation.py \
      --episodes 20 \
      --task easy \
      --exploration-alpha $alpha \
      --out-dir artifacts/ablation_alpha_$alpha
done

# Then compare results
python compare_alpha_values.py
```

### 5. Test on Hard Task
```bash
python experiments/run_evaluation.py \
  --episodes 30 \
  --task hard \
  --exploration-alpha 1.5  # Encourage exploration for hard task
```

### 6. Stability Test (Same conditions, different seeds)
```bash
# Run 3 times with same alpha, different random seeds
for run in 1 2 3; do
    python experiments/run_evaluation.py \
      --episodes 25 \
      --task medium \
      --exploration-alpha 1.0 \
      --out-dir artifacts/stability_run_$run
done
```

---

## 🔍 Verification Commands

### Verify Installation
```bash
# Run comprehensive verification
python verify_improvements.py
# Should see: "ALL VERIFICATIONS PASSED!"

# Show mathematical behavior
python test_smoothing_temperature.py
# Demonstrates smoothing and temperature formulas

# Compare two alpha runs
python compare_alpha_values.py
# Side-by-side metrics comparison
```

### Check Test Suite
```bash
# Run all environment tests
python -m pytest tests/test_environment.py -v

# Should see: "14 passed in 0.25s"
```

---

## 📈 Analysis Commands

### Load and Analyze Results
```python
import pandas as pd

# Read evaluation CSV
df = pd.read_csv('artifacts/evaluation/episode_metrics.csv')

# Check smoothing (should all be > 0)
print("Path frequency range:", df['path_frequency'].min(), "-", df['path_frequency'].max())
assert (df['path_frequency'] > 0).all()  # ✅ Verify smoothing

# Check temperature effect
print("Unique alphas:", df['exploration_alpha'].unique())
print("Bonus statistics:")
print(df.groupby('matched_trajectory')['trajectory_diversity_bonus'].describe())

# Compare distribution
df.groupby('exploration_alpha')['matched_trajectory'].value_counts()
```

### Plot Diversity Metrics
```python
import matplotlib.pyplot as plt
import pandas as pd

df_1_0 = pd.read_csv('artifacts/evaluation/episode_metrics.csv')
df_1_5 = pd.read_csv('artifacts/evaluation_alpha_1.5/episode_metrics.csv')

# Compare average bonus by path
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, df, alpha in zip(axes, [df_1_0, df_1_5], [1.0, 1.5]):
    df.groupby('matched_trajectory')['trajectory_diversity_bonus'].mean().plot(
        kind='bar', ax=ax, title=f'Alpha = {alpha}'
    )
    ax.set_ylabel('Avg Bonus ($)')
    ax.set_xlabel('Architecture Path')

plt.tight_layout()
plt.show()
```

---

## 🎓 Understanding Alpha Values

```
Alpha   Meaning                    When to Use
────────────────────────────────────────────────────────────
0.5     Gentle exploration         Stable/safe training
0.8     Soft exploration           Focused learning  
1.0     Balanced (DEFAULT)         General purpose
1.2     Moderate exploration       Extended runs
1.5     Strong exploration         Diversity research
2.0     Aggressive exploration     Ablation studies
```

### Visual Comparison
```
Bonus vs Path Frequency (5% frequency example):

    0.05 │     
    0.04 │  α=0.5 ━━━━━━━━
         │        α=1.0 ━╱────
    0.03 │       α=1.5 ╱━┴────
         │      α=2.0╱  
    0.02 │________╱________
         └─────────────────
              Rare Path
```

---

## 📂 Output Files

### Generated by Evaluation Script
```
artifacts/evaluation/
├── episode_metrics.csv          # Full episode data (26 columns)
├── reward_vs_mode.png          # Reward by chaos mode
├── oracle_score_vs_steps.png   # Oracle alignment vs efficiency
├── success_rate.png            # Success by agent/mode
├── compromise_detection_rate.png # Tradeoff awareness
├── reward_vs_phase.png         # Rewards by episode phase
└── failure_distribution_by_phase.png # Failure analysis
```

### Key Columns (CSV)
```
episode_id                        # 0-indexed episode number
task_id                          # easy|medium|hard
mode                             # clean|noisy|adversarial
agent                            # random|heuristic|improved
reward                           # Total episode reward
oracle_score                     # 0.0-1.0 correctness
success                          # Binary success (1=yes, 0=no)
path_frequency                   # Smoothed freq (0.0-1.0) ← NEW!
contextual_bonus_scale           # Temperature effect       ← NEW!
exploration_alpha                # Alpha value used        ← NEW!
trajectory_diversity_bonus       # $ reward for exploration
matched_trajectory               # Which path chosen
... (and 16 more columns)
```

---

## 💡 Tips & Tricks

### Tip 1: Quick Iteration
```bash
# Run quick 5-episode test to verify setup
python experiments/run_evaluation.py --episodes 5 --task easy

# Confirms paths are loading, bonuses computing, CSV writing
# Takes < 1 minute
```

### Tip 2: Compare Modes Isolation
```bash
# In Python, create env directly to test one mode
from env.environment import ArchitectEnv

env = ArchitectEnv(task_id="easy", exploration_alpha=1.5)
obs = env.reset()

# Manually set mode (normally random)
env.mode = "clean"

# Run episode
for _ in range(10):
    obs, reward, done, info = env.step(...)
    print(f"Alpha={info['exploration_alpha']}, Bonus={info['trajectory_diversity_bonus']}")
```

### Tip 3: Extract Specific Paths
```python
import pandas as pd

df = pd.read_csv('artifacts/evaluation/episode_metrics.csv')

# See only streaming path episodes
streaming = df[df['matched_trajectory'] == 'alternative_2']
print(f"Streaming episodes: {len(streaming)}")
print(f"Average bonus: ${streaming['trajectory_diversity_bonus'].mean():.5f}")
print(f"Frequency range: {streaming['path_frequency'].describe()}")
```

### Tip 4: Validate Smoothing
```python
import pandas as pd

df = pd.read_csv('artifacts/evaluation/episode_metrics.csv')

# This should ALWAYS be true with smoothing
assert (df['path_frequency'] > 0).all(), "Smoothing failed!"

# Check it's not too high (sanity check)
assert (df['path_frequency'] < 1.0).all(), "Smoothing too aggressive!"

# Verify Laplace formula: (n+1)/(N+num_paths)
# For first episode with never-tried path:
# (0+1)/(1+5) = 0.1667
assert df['path_frequency'].max() <= 1.0

print("✅ Smoothing verification passed!")
```

### Tip 5: Mass Processing
```bash
# Run 4 different temperature settings
mkdir -p artifacts/temperatures

for alpha in 0.8 1.0 1.5 2.0; do
    echo "Temperature: $alpha"
    python experiments/run_evaluation.py \
      --episodes 15 \
      --task easy \
      --exploration-alpha $alpha \
      --out-dir artifacts/temperatures/$alpha
done

# Compare all at once
ls artifacts/temperatures/*/episode_metrics.csv
# Now you have 4 CSV files to analyze
```

---

## 🛠️ Troubleshooting

### Q: "exploration_alpha not recognized"
```bash
# Might be using older code. Verify changes:
grep "exploration_alpha" env/environment.py

# Should see:
# - def __init__(..., exploration_alpha: float = 1.0)
# - self.exploration_alpha = exploration_alpha
```

### Q: "path_frequency is zero"
```bash
# Smoothing may not be applied. Check:
grep "path_count + 1" env/environment.py

# Should show the (count+1) formula, not just count
```

### Q: "CSV doesn't have contextual_bonus_scale column"
```bash
# Evaluation script might be outdated. Verify:
grep "contextual_bonus_scale" experiments/run_evaluation.py

# Should be present in the info dict
```

### Q: "All tests pass except test_api"
```bash
# That's expected - test_api failure is pre-existing
# Run only environment tests:
python -m pytest tests/test_environment.py -v

# Should see: "14 passed"
```

---

## 🚀 Production Deployment

### Recommended Settings by Workload

**Light Load (5-10 episodes)**
```bash
--exploration-alpha 1.0  # Standard, no tuning needed
```

**Standard Load (30-100 episodes)**
```bash
--exploration-alpha 1.0  # Default balanced
# OR
--exploration-alpha 1.5  # If more diversity desired
```

**Heavy Research (100+ episodes)**
```bash
--exploration-alpha 1.5  # Maintain discovery
# OR
--exploration-alpha 2.0  # Extreme diversity
```

**Production Benchmarks**
```bash
--exploration-alpha 1.0  # Reproducible, standard
# Compare across agents/modes with same alpha
```

---

## 📞 Quick Help

```bash
# Show all available options
python experiments/run_evaluation.py --help

# Output:
# usage: run_evaluation.py [-h] [--episodes EPISODES] [--task {easy,medium,hard}]
#                          [--out-dir OUT_DIR] [--exploration-alpha EXPLORATION_ALPHA]
#
# options:
#   -h, --help                    show help
#   --episodes EPISODES           Episodes per (agent, mode) (default: 100)
#   --task {easy,medium,hard}     Task difficulty (default: easy)
#   --out-dir OUT_DIR             Output directory (default: artifacts/evaluation)
#   --exploration-alpha EXPLORATION_ALPHA
#                                 Temperature for exploration bonus (default: 1.0)
```

---

## ✨ Summary

| Need | Command |
|------|---------|
| Quick test | `python experiments/run_evaluation.py --episodes 5` |
| Standard eval | `python experiments/run_evaluation.py --episodes 30` |
| Benchmark | `python experiments/run_evaluation.py --episodes 100` |
| Explore more | `... --exploration-alpha 1.5` |
| Converge faster | `... --exploration-alpha 0.8` |
| Ablation study | Run with for loop over alpha values |
| Verify setup | `python verify_improvements.py` |
| See math | `python test_smoothing_temperature.py` |
| Compare alphas | `python compare_alpha_values.py` |

**Default that "just works":** `python experiments/run_evaluation.py --episodes 30`
