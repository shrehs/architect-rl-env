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

## Run API

```bash
uvicorn api.server:app --reload
```

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
