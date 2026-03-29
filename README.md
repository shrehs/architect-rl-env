---
title: Architect RL Environment
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: api/server.py
pinned: false
---

# ArchitectRL: AI System Design Consultant Environment

A deterministic OpenEnv-style environment where an agent gathers architecture constraints and recommends a system design.

## Problem

This environment simulates a real-world AI system design consultation workflow.

An AI agent acts as a technical architect assistant and interacts with a user to:

- gather system constraints (use case, latency, accuracy, data scale, update frequency)
- infer requirements from natural language
- recommend appropriate AI architectures (for example, RAG variants, fine-tuning, and agentic systems)

This mirrors practical workflows in ML system design interviews, AI consulting, and enterprise solution architecture.

## Why This Is Real-World

Designing AI systems is a multi-step reasoning task that requires:

- multi-turn interaction
- extraction from ambiguous input
- trade-off analysis
- iterative decision-making under uncertainty

Unlike toy tasks, this setup includes incomplete information, noisy responses, and iterative refinement. It evaluates agent reasoning and planning, not only one-shot generation.

## Environment Design

### Observation Space

Each step returns:

- `last_assistant_message`: prompt/question asked to user
- `constraints_collected`: dictionary of extracted constraints
- `missing_constraints`: remaining constraints to collect
- `mode`: `consultant` or `architect`
- `step_count`: current step index

Observations are intentionally constrained to avoid hidden-state leakage.

### Action Space

Agent provides:

- `user_reply` (string)

This represents simulated user input in response to assistant prompts.

## Reward Design

The reward function is dense and shaped.

Positive signal:

- +0.3 for extracting a new constraint
- +final completion grade when all constraints are gathered

Penalties:

- -0.05 for no progress
- -0.1 for duplicate/repeated submission
- step cost pressure via shaping for efficiency

Stability:

- reward is clamped to `[-1.0, 2.0]`

This combination encourages efficient multi-step reasoning, reduces reward hacking, and provides continuous learning signal.

## Task Design

The environment includes three tasks with increasing difficulty:

- Easy: clear and structured user signals
- Medium: moderate ambiguity and partial noise
- Hard: incomplete and ambiguous signals requiring robust multi-turn reasoning

## Grader Design

Each task uses a deterministic grader that outputs a score in `[0.0, 1.0]` based on:

- completeness of collected constraints
- consistency/correctness
- efficiency

This supports reproducible and fair evaluation.

## Baseline Behavior

`inference.py` provides an interactive and JSON baseline runner that can be used for quick smoke checks and trajectory generation.

Difficulty trends are intentional: easy should generally be solved faster than medium and hard under similar policies.

## Determinism

- no randomness in baseline transition logic
- same input sequence produces same outputs
- determinism and contract behavior are covered by tests

## Strict Contract

The environment in `env/environment.py` implements these exact methods:

- `reset(self) -> Observation`
- `step(self, action: Action) -> Tuple[Observation, float, bool, dict]`
- `state(self) -> dict`

Rules enforced:

- deterministic transitions (same input => same output)
- Pydantic typed models (`Observation`, `Action`, optional `Reward`)
- no side effects outside internal state
- no randomness in baseline path

## Setup

```bash
pip install -r requirements.txt
```

## Run Locally With Docker

```bash
docker build -t architect-rl .
docker run -p 7860:7860 architect-rl
```

## Run API

```bash
uvicorn api.server:app --reload
```

### API Endpoints

- `GET /reset`: initialize environment
- `POST /step`: apply one action
- `GET /state`: inspect full internal state
- `GET /health`: health probe
- `GET /tasks`: list task ids

## Run Baseline Inference

Interactive mode:

```bash
python inference.py --task easy
```

JSON mode:

```bash
python inference.py --json-input input.json --json-output output.json
```

Example input JSON:

```json
{
  "task": "easy",
  "actions": [
    "Our use case is fraud detection with low latency under 20ms",
    "Need 99.9% accuracy and we process 5TB dataset daily"
  ]
}
```

## Tests

```bash
pytest -q
```

## Why This Environment Is Challenging

- requires multi-step planning rather than one-shot answers
- handles ambiguous natural language constraints
- balances speed versus completeness
- includes anti-exploit shaping penalties

This makes it a strong candidate for evaluating advanced agent behavior in realistic architecture-assistant settings.
