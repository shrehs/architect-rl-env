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

# Optional environment variables for local runs.
API_BASE = os.getenv("API_BASE_URL") or os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
USE_LLM = API_KEY is not None and API_BASE is not None


PRIORITY_CONSTRAINTS = [
    ("use_case", "ASK_USE_CASE"),
    ("latency", "ASK_LATENCY"),
    ("accuracy", "ASK_ACCURACY"),
    ("data_size", "ASK_DATA_SIZE"),
    ("update_frequency", "ASK_UPDATE_FREQUENCY"),
    ("budget", "ASK_BUDGET"),
]


def ping_llm(observation: Any) -> str:
    if not USE_LLM:
        return None

    try:
        client = OpenAI(
            base_url=API_BASE,
            api_key=API_KEY,
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Observation: {observation}\\nChoose next action.",
                }
            ],
            temperature=0,
        )

        return response.choices[0].message.content

    except Exception:
        return None


def prioritized_constraint_action(observation: Any) -> str:
    missing = set(observation.missing_constraints or [])
    collected = observation.constraints_collected or {}

    for key, action in PRIORITY_CONSTRAINTS:
        value = str(collected.get(key, "")).strip()
        if key in missing or not value:
            return action

    return ""


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
    last_actions: List[str] = []
    steps = 0
    final_info = {}
    
    for step_num in range(env.max_steps):
        _ = ping_llm(observation)
        action_type = prioritized_constraint_action(observation)
        if not action_type:
            action_type = choose_action(agent, observation)
        # Prevent infinite repetition by forcing exploratory fallback.
        if len(last_actions) >= 2 and all(a == action_type for a in last_actions[-2:]):
            action_type = "ASK_LATENCY"
        last_actions.append(action_type)
        observation, reward, done, info = env.step(Action(type=action_type))
        
        # Normalize reward ONCE and append: single source of truth
        normalized_reward = min(reward / 2.0, 1.0)
        rewards.append(normalized_reward)
        steps += 1
        final_info = info
        
        # Output minimal validator format: [STEP] step, action, reward.
        print(
            f"[STEP] step={steps} action={action_type} reward={normalized_reward:.2f}",
            flush=True,
        )
        
        if done:
            break
    
    # Final oracle score from last step info
    constraints_collected_count = int(final_info.get("constraints_collected_count", 0))
    oracle_score = float(final_info.get("oracle_score", 0.0))
    if constraints_collected_count == 0:
        oracle_score = 0.0
    oracle_score = min(max(oracle_score, 0.0), 1.0)  # Defensive: clamp to [0.0, 1.0]
    success = oracle_score >= 0.8
    
    combined_reward = final_info.get("combined_reward") or (
        0.7 * float(final_info.get("oracle_score", 0.0))
        + 0.3 * float(final_info.get("trajectory_score", 0.0))
    )

    return {
        "task_id": task_id,
        "agent": agent,
        "steps": steps,
        "success": success,
        "oracle_score": oracle_score,
        "combined_reward": float(combined_reward),
        "rewards": rewards,  # Already normalized [0,1]
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
    parser = argparse.ArgumentParser(description="ArchitectRL inference runner")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--agent", default="heuristic", choices=["manual", "random", "heuristic"])
    parser.add_argument("--num-episodes", type=int, default=1)
    args = parser.parse_args()

    num_episodes = max(1, args.num_episodes)

    for _ in range(num_episodes):
        print(f"[START] task={args.task}", flush=True)
        try:
            result = run_compliant_episode(task_id=args.task, agent=args.agent, verbose=True)
            final_score = float(result.get("combined_reward", result.get("oracle_score", 0.0)))
            final_score = min(final_score, 1.0)
            print(
                f"[END] task={args.task} score={final_score:.2f} steps={result['steps']}",
                flush=True,
            )
        except Exception:
            print(
                f"[END] task={args.task} score=0.00 steps=0",
                flush=True,
            )
            raise


if __name__ == "__main__":
    main()
