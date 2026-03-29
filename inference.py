import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import ArchitectEnv
from env.models import Action

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


def run_episode(task_id: str, user_replies: List[str]) -> Dict[str, Any]:
    env = ArchitectEnv(task_id=task_id)
    first_observation = env.reset()

    trajectory: List[Dict[str, Any]] = [
        {"observation": first_observation.model_dump(), "reward": 0.0, "done": False, "info": {}}
    ]

    for reply in user_replies:
        observation, reward, done, info = env.step(Action(user_reply=reply))
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


def interactive_mode(task_id: str) -> None:
    env = ArchitectEnv(task_id=task_id)
    print("ArchitectRL interactive mode. Type 'exit' to stop.")
    print(env.reset().last_assistant_message)

    while True:
        user_reply = input("You: ").strip()
        if user_reply.lower() in {"exit", "quit"}:
            break

        observation, reward, done, info = env.step(Action(user_reply=user_reply))
        print(f"Assistant: {observation.last_assistant_message}")
        print(f"Reward: {reward:.3f} | Done: {done} | Info: {json.dumps(info)}")

        if done:
            print("Episode complete.")
            break


def json_mode(input_path: str, output_path: str) -> None:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    task_id = str(payload.get("task", "easy"))
    actions = [str(item) for item in payload.get("actions", [])]

    result = run_episode(task_id=task_id, user_replies=actions)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ArchitectRL baseline inference runner")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--json-input", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    if args.json_input and args.json_output:
        json_mode(args.json_input, args.json_output)
        return

    interactive_mode(args.task)


if __name__ == "__main__":
    main()
