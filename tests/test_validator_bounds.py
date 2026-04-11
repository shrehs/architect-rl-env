#!/usr/bin/env python3
"""
Verify that all trajectory scores are strictly within (0, 1) bounds.
This test mimics what the validator checks.
"""
import os
os.environ["DEBUG"] = "false"

from env.environment import ArchitectEnv
from env.models import Action
import random

print("=" * 80)
print("VALIDATOR COMPLIANCE CHECK: Score Bounds")
print("=" * 80)

# Create environment and run a minimal episode
env = ArchitectEnv()
state = env.reset()

CRITICAL_FIELDS = ["oracle_score", "trajectory_score", "combined_reward"]
BOUNDARY_ERRORS = []

print("\nRunning episode and checking trajectory scores...\n")

# Simple action list (from inference.py)
ACTION_SPACE = [
    "ASK_USE_CASE", "ASK_LATENCY", "ASK_ACCURACY", "ASK_DATA_SIZE",
    "ASK_UPDATE_FREQUENCY", "ASK_BUDGET", "FINALIZE", "FINALIZE_WITH_COMPROMISE",
]

step = 0
done = False
while not done and step < 15:
    # Pick a random action
    action_type = random.choice(ACTION_SPACE)
    action = Action(type=action_type)
    state, reward, done, info = env.step(action)
    step += 1
    
    # Check each critical field
    for field in CRITICAL_FIELDS:
        if field in info:
            value = info[field]
            if not isinstance(value, (int, float)):
                continue
            
            # Check strict bounds: must be > 0 and < 1
            if value <= 0.0 or value >= 1.0:
                BOUNDARY_ERRORS.append({
                    "step": step,
                    "field": field,
                    "value": value,
                    "error": "OUT OF BOUNDS"
                })
                print(f"  ❌ STEP {step}: {field} = {value} (OUT OF BOUNDS)")

# After episode completes, check final trajectory scores
print(f"\nFinal trajectory info (last step):")
for field in CRITICAL_FIELDS:
    if field in info:
        value = info[field]
        if isinstance(value, (int, float)):
            status = "✅" if 0 < value < 1 else "❌"
            print(f"  {status} {field}: {value:.4f}")

print("\n" + "=" * 80)
if BOUNDARY_ERRORS:
    print(f"VALIDATOR WOULD FAIL: {len(BOUNDARY_ERRORS)} boundary violations found")
    for error in BOUNDARY_ERRORS[:5]:
        print(f"   - Step {error['step']}: {error['field']} = {error['value']}")
else:
    print("VALIDATOR WOULD PASS: All scores strictly in (0, 1)")
print("=" * 80)
