import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required for core functionality

from openai import OpenAI

from env.agents import choose_action
from env.environment import ArchitectEnv
from env.models import Action

load_dotenv()

# Mandatory environment variables with fallbacks
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def run_episode(task_id: str, action_types: List[str]) -> Dict[str, Any]:
    env = ArchitectEnv(task_id=task_id)
    first_observation = env.reset()

    trajectory: List[Dict[str, Any]] = [
        {"observation": first_observation.model_dump(), "reward": 0.0, "done": False, "info": {}}
    ]

    for action_type in action_types:
        observation, reward, done, info = env.step(Action(type=action_type))
        trajectory.append(
            {
                "observation": observation.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        if done:
            break

    return {"task_id": task_id, "trajectory": trajectory, "final_state": env.state()}


def run_compliant_episode(task_id: str = "easy", agent: str = "heuristic", verbose: bool = False) -> Dict[str, Any]:
    """Run episode with compliant [START]/[STEP]/[END] logging."""
    env = ArchitectEnv(task_id=task_id)
    observation = env.reset()
    
    rewards: List[float] = []
    oracle_scores: List[float] = []
    steps = 0
    final_info = {}
    
    for step_num in range(env.max_steps):
        action_type = choose_action(agent, observation)
        observation, reward, done, info = env.step(Action(type=action_type))
        
        rewards.append(reward)
        steps += 1
        final_info = info
        
        # Extract oracle score for this step (0.0-1.0)
        step_oracle_score = float(info.get("oracle_score", 0.0))
        oracle_scores.append(step_oracle_score)
        
        # Output [STEP] log with normalized values
        if verbose:
            normalized_reward = min(reward / 2.0, 1.0)  # Normalize to ~[0,1]
            print(f"[STEP] step={steps} action={action_type} reward={normalized_reward:.2f} oracle_score={step_oracle_score:.2f}")
        
        if done:
            break
    
    # Final oracle score from last step info
    oracle_score = float(final_info.get("oracle_score", 0.0))
    success = oracle_score >= 0.8
    
    # Normalize rewards for output (they can be up to 2.0 internally)
    normalized_rewards = [min(r / 2.0, 1.0) for r in rewards]
    
    return {
        "task_id": task_id,
        "agent": agent,
        "steps": steps,
        "success": success,
        "oracle_score": oracle_score,
        "rewards": normalized_rewards,  # Normalized rewards for output
        "raw_rewards": rewards,  # Raw rewards for debugging  
        "final_state": env.state()
    }


def run_policy_episode(task_id: str, agent: str) -> Dict[str, Any]:
    env = ArchitectEnv(task_id=task_id)
    observation = env.reset()

    trajectory: List[Dict[str, Any]] = [
        {"observation": observation.model_dump(), "reward": 0.0, "done": False, "info": {}}
    ]

    for _ in range(env.max_steps):
        action_type = choose_action(agent, observation)
        observation, reward, done, info = env.step(Action(type=action_type))
        trajectory.append(
            {
                "action": action_type,
                "observation": observation.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        if done:
            break

    return {"task_id": task_id, "agent": agent, "trajectory": trajectory, "final_state": env.state()}


def interactive_mode(task_id: str) -> None:
    env = ArchitectEnv(task_id=task_id)
    print("ArchitectRL interactive mode. Type 'exit' to stop.")
    print(env.reset().last_assistant_message)
    print("Available actions: ASK_USE_CASE, ASK_LATENCY, ASK_ACCURACY, ASK_DATA_SIZE, ASK_UPDATE_FREQUENCY, ASK_BUDGET, FINALIZE, FINALIZE_WITH_COMPROMISE")

    while True:
        action_type = input("Action type: ").strip()
        if action_type.lower() in {"exit", "quit"}:
            break

        observation, reward, done, info = env.step(Action(type=action_type))
        print(f"Assistant: {observation.last_assistant_message}")
        print(f"Reward: {reward:.3f} | Done: {done} | Info: {json.dumps(info)}")

        if done:
            print("Episode complete.")
            break


def json_mode(input_path: str, output_path: str) -> None:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    task_id = str(payload.get("task", "easy"))
    actions = [str(item) for item in payload.get("actions", [])]

    result = run_episode(task_id=task_id, action_types=actions)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ArchitectRL baseline inference runner")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--json-input", default="")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--agent", default="heuristic", choices=["manual", "random", "heuristic"])
    parser.add_argument("--compliant", action="store_true", help="Output compliant [START]/[STEP]/[END] logs")
    args = parser.parse_args()

    if args.compliant:
        print(f"[START] task={args.task} env=architectenv model={MODEL_NAME}")
        try:
            result = run_compliant_episode(task_id=args.task, agent=args.agent, verbose=True)
            rewards_str = ",".join(f"{r:.2f}" for r in result["rewards"])
            print(f"[END] success={str(result['success']).lower()} steps={result['steps']} score={result['oracle_score']:.2f} rewards={rewards_str}")
        except Exception as e:
            print(f"[END] error={str(e)}")
            raise
        return

    if args.agent in {"random", "heuristic"}:
        result = run_policy_episode(task_id=args.task, agent=args.agent)
        print(json.dumps(result, indent=2))
        return

    if args.json_input and args.json_output:
        json_mode(args.json_input, args.json_output)
        return

    interactive_mode(args.task)


if __name__ == "__main__":
    main()
