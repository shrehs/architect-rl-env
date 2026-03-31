---
title: Architect RL Environment
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: api/server.py
pinned: false
---

# Architect RL Environment

## Overview

Architect RL Environment is a real-world, multi-turn AI system design simulation built using the OpenEnv framework. It evaluates an agent's ability to gather system constraints and recommend appropriate AI architectures through structured interaction.

Unlike toy environments, this models a practical workflow used by ML engineers and AI architects when designing systems such as RAG pipelines, chatbots, or real-time AI services.

---

## Problem Motivation

Designing AI systems requires:
- Understanding user requirements
- Handling ambiguity and trade-offs
- Making architecture decisions under constraints

This environment simulates that process by requiring an agent to:
1. Collect key constraints
2. Interpret user responses
3. Recommend appropriate system architectures

---

## Real-World Utility

This environment reflects tasks performed in industry at companies like:
- AI product teams
- ML infrastructure teams
- Consulting and solution architecture roles

It can be used for:
- Evaluating LLM reasoning ability
- Training agents for structured decision-making
- Benchmarking multi-turn planning systems

---

## Environment Design

### Core API (OpenEnv compliant)

- reset() -> Observation
- step(action) -> (Observation, reward, done, info)
- state() -> dict

### Determinism

The environment is fully deterministic:
Same initial state + same action sequence -> identical outputs

No randomness is used.

---

## Observation Space

```json
{
  "last_assistant_message": "string",
  "constraints_collected": "object",
  "missing_constraints": ["string"],
  "mode": "string",
  "step_count": "integer"
}
```

---

## Action Space

```json
{
  "user_reply": "string"
}
```

---

## Task Design

Three tasks with increasing difficulty:

### Easy

- Clear user responses
- Direct constraint extraction
- Minimal ambiguity

### Medium

- Partial or vague responses
- Requires interpretation
- Some ambiguity handling

### Hard

- Conflicting or incomplete constraints
- Requires trade-off reasoning
- Multi-step inference

Each task includes a deterministic grader scoring from 0.0 to 1.0.

---

## Reward Design (Important)

The reward function provides dense feedback across the episode:

### Positive Signals

- Reward for collecting new constraints
- Reward for completing all constraints efficiently

### Penalties

- -0.05 for no progress in a step
- -0.1 for duplicate submissions (anti-exploit)
- Prevents reward hacking via random spam

### Final Score

- Task grader outputs normalized score in [0.0, 1.0]

---

## Anti-Exploit Mechanisms

- Duplicate input penalty
- No-progress penalty
- Step efficiency tracking
- Deterministic behavior prevents stochastic exploitation

---

## Episode Rules

- Max steps: 8
- Episode ends when:
  - All constraints are collected
  - Or max steps reached

### Strict Enforcement

- Calling step() after done=True raises RuntimeError
- API maps this to HTTP 409 Conflict

---

## Baseline Inference

Run:

```bash
python inference.py
```

Supports:

- Interactive CLI mode
- JSON input/output mode

### Environment Variables Required

- OPENAI_API_KEY
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

---

## Baseline Behavior

The baseline agent:

- Iteratively queries missing constraints
- Uses LLM responses to decide next actions
- Produces reproducible scores across tasks

---

## Deployment

### Docker

```bash
docker build -t architect-rl .
docker run --rm -p 7860:7860 architect-rl
```

### Hugging Face Space

- Containerized deployment
- API endpoints:
  - /reset
  - /step

---

## API Endpoints

### Reset

GET /reset
POST /reset

Returns initial observation.

---

### Step

POST /step

Request:

```json
{
  "user_reply": "string"
}
```

Response:

```json
{
  "observation": {...},
  "reward": "float",
  "done": "bool",
  "info": {...}
}
```

---

## Testing and Validation

Includes tests for:

- Determinism
- API contract compliance
- Post-done behavior
- Reward stability
- Anti-exploit resistance
- YAML-task alignment

---

## Retrospective Insights

- Determinism is critical for evaluation environments
- Reward shaping must prevent exploitation
- API correctness matters as much as core logic
- Deployment failures often stem from metadata, not code
- Lifecycle invariants (for example, done-state) must be strictly enforced

---

## Outcome

This environment is:

- Deterministic
- OpenEnv-compliant
- Reward-shaped with anti-exploit mechanisms
- Fully containerized
- Deployable on Hugging Face Spaces
- Backed by reproducible evaluation

---

## Summary

Architect RL Environment provides a realistic benchmark for evaluating multi-turn AI reasoning in system design, bridging the gap between academic RL environments and real-world AI workflows.

## Safety Considerations

This environment includes basic safeguards to ensure stable and reliable agent evaluation:

### Implemented Safety Measures

- Deterministic transitions to ensure reproducibility
- Reward shaping to prevent exploitative behaviors (e.g. spam, no-progress loops)
- Strict episode lifecycle enforcement (post-done calls raise errors)
- Bounded reward signals to prevent instability

### Limitations

- The environment uses rule-based constraint extraction and does not validate semantic correctness of architectural recommendations
- No explicit filtering of harmful or adversarial user inputs is implemented
- Recommendations are not evaluated for real-world safety, compliance, or ethical considerations

### Future Improvements

- Add constraint validation checks for conflicting or unsafe configurations
- Introduce safety-aware grading (e.g. penalizing risky architectures)
- Integrate content moderation for adversarial or harmful inputs
