---
title: ArchitectEnv
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: api/server.py
pinned: false
---

# 🧠 ArchitectEnv: Tradeoff-Aware RL Environment for AI System Design

## 📌 Overview

ArchitectEnv is a **real-world reinforcement learning environment** for training agents to perform **AI system design under uncertainty**.

Unlike typical environments that reward simple correctness, ArchitectEnv evaluates whether agents can:

* Elicit missing constraints from users
* Handle noisy and adversarial inputs
* Operate under partial observability
* Detect **infeasible constraint combinations**
* Produce **tradeoff-aware architecture decisions**

---

## 🚀 Key Contribution

> **We introduce Tradeoff Awareness as a measurable capability in RL environments.**

### 📊 Tradeoff Awareness Rate (TAR)

TAR measures whether an agent recognizes when no solution satisfies all constraints and responds with a **compromise architecture**.

**Hard Task Results (100 episodes):**

| Agent                | Tradeoff Awareness Rate |
| -------------------- | ----------------------- |
| Random               | 0%                      |
| Heuristic            | 0%                      |
| Tradeoff-Aware Agent | 50%                     |

👉 This shows that standard policies fail to reason under infeasibility, while tradeoff-aware behavior can be learned.

---

## 🧩 Environment Design

### State Representation

```python
state = {
  "observed_constraints": {},   # constraints elicited by agent
  "hidden_constraints": {},     # ground truth (not visible)
  "derived_constraints": {}     # inferred signals
}
```

---

### Action Space (Typed)

```python
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

### Observation Space

```python
Observation(
  last_assistant_message: str,
  constraints_collected: dict,
  missing_constraints: list,
  mode: str,
  step_count: int
)
```

---

### Episode Flow

1. Agent selects an action
2. Environment simulates user response
3. Constraints are extracted
4. Reward is computed
5. Episode terminates on:

   * `FINALIZE`
   * `FINALIZE_WITH_COMPROMISE`
   * max steps reached

---

## 🌍 Real-World Complexity

### Chaos Modes

* **clean** → direct answers
* **noisy** → vague / incomplete responses
* **adversarial** → misleading signals
* **dynamic (optional)** → constraints change mid-episode

---

### Hard Tasks (Infeasible Scenarios)

Example:

```python
{
  "latency": "real-time",
  "budget": "low",
  "data_size": "large",
  "accuracy": "high"
}
```

👉 No architecture satisfies all constraints
👉 Requires **explicit tradeoff reasoning**

---

## 🧠 Oracle (Expert Policy)

The environment uses a **structured oracle** derived from practitioner reasoning.

```python
oracle_output = {
  "model": "...",
  "deployment": "...",
  "architecture": "...",
  "reasoning": "tradeoff explanation"
}
```

For infeasible scenarios, the oracle returns a **compromise solution**, not an optimal one.

---

## 🎯 Reward Function

### Step Reward

* Information gain (uncertainty reduction)
* Penalty for redundant or irrelevant actions

### Terminal Reward

* Similarity to oracle output
* Coverage of constraints
* Penalty for missing critical constraints
* Bonus for **explicit tradeoff recognition** (`FINALIZE_WITH_COMPROMISE`)

---

## 🧪 Tasks

| Task   | Description                                 |
| ------ | ------------------------------------------- |
| Easy   | Fully satisfiable constraints               |
| Medium | Partial conflicts, manageable tradeoffs     |
| Hard   | Infeasible constraints requiring compromise |

Each task includes:

* Ground truth constraints
* Grader returning score in **[0.0 – 1.0]**

---

## 🤖 Baseline Agents

* **Random Agent** → random actions
* **Heuristic Agent** → deterministic constraint collection
* **Tradeoff-Aware Agent** → detects infeasibility and finalizes with compromise

---

## 📊 Evaluation Outputs

Generated via `experiments/run_evaluation.py`:

* `reward_vs_mode.png`
* `oracle_score_vs_steps.png`
* `success_rate.png`
* `compromise_detection_rate.png`
* `episode_metrics.csv`

---

## 💥 Failure Case: Overconfident Agent Under Uncertainty

Mode: Adversarial
Agent actions: ASK_LATENCY -> ASK_BUDGET -> FINALIZE

Result:

* Constraints collected: 0
* Oracle score: 0.1
* Reward: -0.96

Failure:
The agent prematurely finalized without sufficient information,
leading to a misaligned architecture recommendation.

Failure reason: overconfident_no_tradeoff

This demonstrates that the environment penalizes overconfident decision-making under uncertainty, a critical property of real-world AI system design.

---

## ⚙️ Setup Instructions

```bash
git clone <repo>
cd <repo>

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate (Windows)

pip install -r requirements.txt
```

---

## ▶️ Run Evaluation

```bash
PYTHONPATH=. python experiments/run_evaluation.py \
  --episodes 100 \
  --task hard \
  --out-dir artifacts/evaluation
```

---

## 🤖 Inference Script

The required inference script is provided as:

```
inference.py
```

It:

* Loads the environment
* Interacts via `reset()` and `step()`
* Uses OpenAI client for action generation
* Produces reproducible scores

---

## 🌐 Deployment

* Dockerized environment provided via `Dockerfile`
* Deployed on Hugging Face Spaces
* Supports OpenEnv API:

  * `/reset`
  * `/step`
  * `/state`

---

## 🔐 Required Environment Variables

```bash
API_BASE_URL=<your_api_url>
MODEL_NAME=<model_name>
HF_TOKEN=<your_token>
```

---

## 📦 OpenEnv Compliance

* ✅ Typed models (Action, Observation)
* ✅ `step() / reset() / state()` implemented
* ✅ `openenv.yaml` included
* ✅ Docker build passes
* ✅ API returns valid responses

---

## ⏱️ Constraints

* Runtime < 20 minutes
* Compatible with:

  * 2 vCPU
  * 8GB RAM

---

## ✅ Pre-Submission Checklist

* [x] HF Space deploys and responds to `/reset`
* [x] Dockerfile builds successfully
* [x] `openenv.yaml` validated
* [x] `inference.py` runs without errors
* [x] 3 tasks implemented with graders
* [x] Scores in valid range [0.0 – 1.0]

---

## 🧠 Key Insight

> Even high-performing heuristic agents achieve **0% Tradeoff Awareness Rate**.

This highlights a critical gap:

> **Agents can collect information but fail to reason under constraint conflict.**

ArchitectEnv provides a framework to **measure and improve this capability**.
