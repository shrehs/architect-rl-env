# 🧠 ArchitectEnv

### *Process-Aware Reinforcement Learning for Multi-Step System Design*

> **Train agents to think like system architects — not just output answers.**

---

## 🚀 TL;DR

ArchitectEnv is a **next-generation RL environment** where agents learn to:

* Ask the *right questions* before acting
* Navigate **multiple valid solutions** (not single-answer tasks)
* Handle **uncertainty, noise, and adversarial inputs**
* Optimize **reasoning process**, not just final answers

> 💡 **Core Idea:** Real-world problems don’t have one correct solution — they have **tradeoffs**.

---

## 🎯 Why ArchitectEnv?

Traditional RL benchmarks:

* ❌ Single correct answer
* ❌ Binary rewards
* ❌ No reasoning evaluation

ArchitectEnv:

* ✅ **Multiple valid architectures**
* ✅ **Continuous reward gradients**
* ✅ **Process-aware evaluation**
* ✅ **Adversarial robustness testing**

---

## 🧩 What Makes This Novel?

### 1. 🧠 Multi-Solution Oracle (NOT single-answer)

```python
oracle_recommend(constraints) → {
  "primary": {...},
  "alternatives": [...],
  "valid_paths": [...]
}
```

✔ Agents are rewarded for **any valid solution**
✔ Encourages **understanding tradeoffs**, not memorization

---

### 2. 📊 Process-Aware Rewards (Core Innovation)

Agents are evaluated on **how they solve**, not just *what they output*:

| Signal            | Purpose                         |
| ----------------- | ------------------------------- |
| Information Gain  | Ask useful questions            |
| Efficiency Reward | Avoid redundant steps           |
| Consistency Score | Avoid flip-flopping             |
| Recovery Score    | Handle adversarial inputs       |
| Trajectory Score  | Evaluate full reasoning process |

---

### 3. 🌍 Real-World Uncertainty Modes

| Mode            | Behavior                |
| --------------- | ----------------------- |
| **Clean**       | Perfect information     |
| **Noisy**       | Partial / vague answers |
| **Adversarial** | Misleading signals      |

✔ Tests **robust reasoning**, not just correctness

---

### 4. 🔀 Trajectory Diversity (Exploration Innovation)

Agents are rewarded for exploring **different valid architectures**:

```
bonus = 0.05 × (1 - path_frequency)^α
```

✔ Prevents policy collapse
✔ Encourages discovering **alternative solutions**
✔ Orthogonal to correctness (no gaming)

---

## 🏗️ The Problem Setting

Given partial constraints:

```python
{
  "latency": "real-time",
  "accuracy": "high",
  "data_size": "large",
  "budget": "medium"
}
```

Agents must:

1. Ask questions 🧩
2. Infer missing constraints 🔍
3. Propose architecture ⚙️
4. Justify tradeoffs 🧠

---

## 🔀 Multiple Valid Solutions

| Architecture | Strength          |
| ------------ | ----------------- |
| Streaming    | Low latency       |
| Batch        | High accuracy     |
| Hybrid       | Balanced          |
| Edge         | Ultra-low latency |

✔ **All can be correct** depending on reasoning

---

## 🧠 Reward System (Key Insight)

### Final Score

```
final_score = 0.7 × oracle_score + 0.3 × trajectory_score
```

### Trajectory Score

```
trajectory = 0.4 × consistency
           + 0.3 × efficiency
           + 0.3 × recovery
```

✔ Encourages:

* Stable reasoning
* Fast convergence
* Robust decision-making

---

## 📊 Proven Learning (Empirical Results)

| Metric           | Heuristic | Improved Agent |
| ---------------- | --------- | -------------- |
| Oracle Score     | 0.53      | **0.73**       |
| Trajectory Score | 0.80      | **0.96**       |
| Recovery         | 0.87      | **1.00**       |
| Steps            | 9.3       | **6.3**        |
| Success Rate     | 53%       | **73%**        |

---

## 📈 Behavioral Learning Evidence

✔ Agents learn **which actions matter**
✔ Reduce **random repetition by 50%**
✔ Develop **action ordering strategies**
✔ Shift from **confused → adaptive behavior**

---

## 🧪 Environment Design

### State

```python
state = {
  "observed_constraints": {},
  "hidden_constraints": {},
  "belief": {},
  "phase": "exploration",
  "mode": "noisy"
}
```

### Actions

```
ASK_USE_CASE
ASK_LATENCY
ASK_ACCURACY
ASK_DATA_SIZE
ASK_UPDATE_FREQUENCY
ASK_BUDGET
FINALIZE
FINALIZE_WITH_COMPROMISE
```

---

## 🔄 Episode Flow

```
Exploration → Refinement → Decision
```

✔ Ask → Learn → Adapt → Decide

---

## 🤖 Baseline Agents

| Agent     | Behavior                     |
| --------- | ---------------------------- |
| Random    | Fails (≈0 score)             |
| Heuristic | Rigid, single-path           |
| Improved  | Adaptive, explores tradeoffs |

---

## 📊 Outputs

* `oracle_score` → correctness
* `trajectory_score` → reasoning quality
* `diversity_bonus` → exploration
* `episode_metrics.csv` → 50+ metrics

---

## ⚙️ Quick Start

```bash
git clone <repo>
cd <repo>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Run Evaluation

```bash
PYTHONPATH=. python experiments/run_evaluation.py \
  --episodes 30 \
  --task easy \
  --out-dir artifacts/evaluation
```

---

## 🌐 Deployment

* Dockerized ✅
* Hugging Face Spaces ✅
* OpenEnv API:

  * `/reset`
  * `/step`
  * `/state`

---

## 🧠 Advanced RL Features

✔ Generalized Advantage Estimation (GAE)
✔ N-step returns
✔ Action entropy tracking
✔ Dense reward decomposition
✔ Advantage-based learning signals

---

## 🔬 Research Contributions

1. Process-Aware Reward Shaping
2. Multi-Solution RL Evaluation
3. Trajectory-Level Scoring
4. Adversarial Robustness in RL
5. Behavioral Pattern Detection via Entropy

---

## 📄 Paper (In Progress)

> *Process-Aware Reinforcement Learning for Multi-Step Reasoning Under Uncertainty*

✔ Full system
✔ Empirical validation
✔ Visualization suite

---

## 🔮 Future Work

* Information-theoretic question selection
* Stronger adversarial simulations
* Belief calibration (uncertainty-aware agents)
* Curriculum learning

---

## 🧠 Key Insight

> **Good agents don’t just give answers.
> They ask, adapt, and reason under uncertainty.**

---

## ⭐ Final Takeaway

ArchitectEnv transforms RL from:

❌ “Find the correct answer”

to

✅ **“Understand the solution space.”**
