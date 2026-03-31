from copy import deepcopy
from typing import Dict, Tuple

from .models import Action, Message, Observation
from .reward import shaped_step_reward
from .tasks import TASKS, grade_constraints
from .utils import REQUIRED_CONSTRAINTS, extract_constraints, generate_recommendation, missing_constraints


class ArchitectEnv:
    """Strict OpenEnv contract implementation.

    Required signatures:
    - reset(self) -> Observation
    - step(self, action: Action) -> Tuple[Observation, float, bool, dict]
    - state(self) -> dict
    """

    def __init__(self, task_id: str = "easy", max_steps: int = 8):
        self.task_id = task_id if task_id in TASKS else "easy"
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> Observation:
        self.state_data: Dict[str, object] = {
            "messages": [],
            "constraints": {},
            "mode": "consultant",
            "step_count": 0,
            "done": False,
            "task_id": self.task_id,
            "last_assistant_message": "What are your use case, latency, accuracy, data size, and update frequency constraints?",
        }
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.state_data["done"]:
            raise RuntimeError("Episode already finished. Call reset().")

        user_action = Action.model_validate(action)
        self._append_message("user", user_action.user_reply)

        before = dict(self.state_data["constraints"])
        after = extract_constraints(user_action.user_reply, before)
        self.state_data["constraints"] = after
        new_constraints_count = len(after) - len(before)
        duplicate_submission = user_action.user_reply in before.values()

        self.state_data["step_count"] = int(self.state_data["step_count"]) + 1
        missing = missing_constraints(after)

        done = len(missing) == 0 or int(self.state_data["step_count"]) >= self.max_steps
        self.state_data["done"] = done

        if done:
            recommendation = generate_recommendation(after)
            assistant_message = f"Recommendation: {recommendation}"
        else:
            next_key = missing[0]
            assistant_message = f"Please provide details for: {next_key}."

        self.state_data["last_assistant_message"] = assistant_message
        self._append_message("assistant", assistant_message)

        reward = shaped_step_reward(new_constraints_count, repeated_turn=(new_constraints_count == 0))
        if new_constraints_count == 0:
            reward -= 0.05
        if duplicate_submission:
            reward -= 0.1
        lowered_reply = user_action.user_reply.lower()
        unsafe_input = "hack" in lowered_reply or "bypass" in lowered_reply
        if unsafe_input:
            reward -= 0.1

        constraints_count = len(after)
        step_count = int(self.state_data["step_count"])
        info: Dict[str, object] = {
            "task_id": self.task_id,
            "new_constraints_count": new_constraints_count,
            "missing_constraints": missing,
            "constraints_collected_count": constraints_count,
            "progress": float(constraints_count) / float(len(REQUIRED_CONSTRAINTS)),
            "step_efficiency": float(step_count) / float(self.max_steps),
            "mode": self.state_data["mode"],
            "duplicate_submission": duplicate_submission,
        }
        if unsafe_input:
            info["unsafe_input"] = True

        if done:
            final_score = grade_constraints(after, self.task_id)
            reward += final_score
            info["final_score"] = final_score

        reward = float(max(min(reward, 2.0), -1.0))
        observation = self._build_observation()
        return observation, reward, bool(self.state_data["done"]), info

    def state(self) -> dict:
        return deepcopy(self.state_data)

    def _build_observation(self) -> Observation:
        return Observation(
            last_assistant_message=str(self.state_data["last_assistant_message"]),
            constraints_collected=dict(self.state_data["constraints"]),
            missing_constraints=[k for k in REQUIRED_CONSTRAINTS if k not in self.state_data["constraints"]],
            mode=str(self.state_data["mode"]),
            step_count=int(self.state_data["step_count"]),
        )

    def _append_message(self, role: str, content: str) -> None:
        msg = Message(role=role, content=content)
        messages = self.state_data["messages"]
        if isinstance(messages, list):
            messages.append(msg.model_dump())
