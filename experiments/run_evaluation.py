import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List
from unittest.mock import patch

import matplotlib.pyplot as plt

from env.agents import choose_action
from env.environment import ArchitectEnv
from env.models import Action
from env.utils import REQUIRED_CONSTRAINTS

MODES = ["clean", "noisy", "adversarial"]
AGENTS = ["random", "heuristic", "improved"]


def run_one_episode(task_id: str, mode: str, agent: str) -> Dict[str, float | int | str]:
    with patch("env.environment.random.choice", return_value=mode):
        env = ArchitectEnv(task_id=task_id)
        observation = env.reset()

        total_reward = 0.0
        final_info: Dict[str, object] = {}
        done = False

        for _ in range(env.max_steps):
            action_type = choose_action(agent, observation)
            observation, reward, done, info = env.step(Action(type=action_type))
            total_reward += float(reward)
            final_info = info
            if done:
                break

    steps = int(observation.step_count)
    oracle_score = float(final_info.get("oracle_score", 0.0))
    compromise_detected = int(bool(final_info.get("compromise_detected", False)))
    coverage = float(
        final_info.get(
            "coverage",
            float(len(observation.constraints_collected)) / float(len(REQUIRED_CONSTRAINTS)),
        )
    )

    # Success definition: reasonable oracle alignment at episode end.
    success = 1 if done and oracle_score >= 0.6 else 0

    return {
        "task": task_id,
        "mode": mode,
        "agent": agent,
        "steps": steps,
        "total_reward": total_reward,
        "oracle_score": oracle_score,
        "coverage": coverage,
        "success": success,
        "compromise_detected": compromise_detected,
    }


def save_csv(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    out_path = out_dir / "episode_metrics.csv"
    fieldnames = [
        "task",
        "mode",
        "agent",
        "steps",
        "total_reward",
        "oracle_score",
        "coverage",
        "success",
        "compromise_detected",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return out_path


def plot_reward_vs_mode(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(MODES))
    width = 0.35
    for i, agent in enumerate(AGENTS):
        means = []
        for mode in MODES:
            vals = [
                float(r["total_reward"])
                for r in records
                if r["agent"] == agent and r["mode"] == mode
            ]
            means.append(mean(vals) if vals else 0.0)
        offsets = [p + (i - 0.5) * width for p in x]
        ax.bar(offsets, means, width=width, label=agent)

    ax.set_xticks(list(x))
    ax.set_xticklabels(MODES)
    ax.set_ylabel("Mean Total Reward")
    ax.set_title("Reward vs Mode")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "reward_vs_mode.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_oracle_vs_steps(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(AGENTS), figsize=(6 * len(AGENTS), 5), sharey=True)
    if len(AGENTS) == 1:
        axes = [axes]
    colors = {"clean": "#2E7D32", "noisy": "#F9A825", "adversarial": "#C62828"}

    for idx, agent in enumerate(AGENTS):
        ax = axes[idx]
        for mode in MODES:
            subset = [r for r in records if r["agent"] == agent and r["mode"] == mode]
            ax.scatter(
                [int(r["steps"]) for r in subset],
                [float(r["oracle_score"]) for r in subset],
                alpha=0.65,
                s=20,
                color=colors[mode],
                label=mode,
            )
        ax.set_title(f"{agent.title()} Agent")
        ax.set_xlabel("Steps")
        if idx == 0:
            ax.set_ylabel("Oracle Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Oracle Score vs Steps", y=1.02)
    fig.tight_layout()

    out_path = out_dir / "oracle_score_vs_steps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_success_rate(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(MODES))
    width = 0.35

    for i, agent in enumerate(AGENTS):
        rates = []
        for mode in MODES:
            vals = [int(r["success"]) for r in records if r["agent"] == agent and r["mode"] == mode]
            rates.append((sum(vals) / len(vals)) if vals else 0.0)
        offsets = [p + (i - 0.5) * width for p in x]
        ax.bar(offsets, rates, width=width, label=agent)

    ax.set_xticks(list(x))
    ax.set_xticklabels(MODES)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success Rate by Mode")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "success_rate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_compromise_detection_rate(records: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(AGENTS))
    rates = []

    for agent in AGENTS:
        vals = [int(r["compromise_detected"]) for r in records if r["agent"] == agent]
        rates.append((sum(vals) / len(vals)) if vals else 0.0)

    bars = ax.bar(list(x), rates, color=["#546E7A", "#6A1B9A"], width=0.55)
    ax.set_xticks(list(x))
    ax.set_xticklabels([agent.title() for agent in AGENTS])
    ax.set_ylabel("Compromise Detection Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Compromise Detection Rate")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.0%}", ha="center", va="bottom")

    fig.tight_layout()

    out_path = out_dir / "compromise_detection_rate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def print_summary(records: List[Dict[str, float | int | str]]) -> None:
    print("\n=== Evaluation Summary ===")
    print("success = 1 if done and oracle_score >= 0.6")

    for agent in AGENTS:
        for mode in MODES:
            subset = [r for r in records if r["agent"] == agent and r["mode"] == mode]
            avg_reward = mean(float(r["total_reward"]) for r in subset)
            avg_oracle = mean(float(r["oracle_score"]) for r in subset)
            avg_steps = mean(int(r["steps"]) for r in subset)
            success_rate = sum(int(r["success"]) for r in subset) / len(subset)
            print(
                f"{agent:9s} | {mode:11s} | reward={avg_reward:6.3f} | "
                f"oracle={avg_oracle:5.3f} | steps={avg_steps:4.2f} | success={success_rate:5.2%}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mode-wise evaluation experiments")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per (agent, mode)")
    parser.add_argument("--task", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--out-dir", type=str, default="artifacts/evaluation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, float | int | str]] = []
    for agent in AGENTS:
        for mode in MODES:
            for _ in range(args.episodes):
                records.append(run_one_episode(task_id=args.task, mode=mode, agent=agent))

    csv_path = save_csv(records, out_dir)
    plot1 = plot_reward_vs_mode(records, out_dir)
    plot2 = plot_oracle_vs_steps(records, out_dir)
    plot3 = plot_success_rate(records, out_dir)
    plot4 = plot_compromise_detection_rate(records, out_dir)

    print_summary(records)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {plot1}")
    print(f"Saved plot: {plot2}")
    print(f"Saved plot: {plot3}")
    print(f"Saved plot: {plot4}")


if __name__ == "__main__":
    main()
