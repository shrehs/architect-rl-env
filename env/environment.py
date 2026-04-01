from copy import deepcopy
import random
from typing import Dict, Tuple

from .models import Action, Message, Observation
from .tasks import TASKS
from .oracle import oracle_recommend
from .user_simulator import UserSimulator
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

    def _belief_space(self) -> Dict[str, list[str]]:
        return {
            "use_case": ["recommendation ranking", "fraud detection", "multimodal assistant"],
            "latency": ["batch", "near_real_time", "real_time"],
            "accuracy": ["low", "medium", "high", "near-perfect"],
            "data_size": ["small", "moderate", "large", "very large"],
            "update_frequency": ["daily", "hourly", "streaming", "continuous"],
            "budget": ["low", "medium", "high"],
        }

    def _belief_from_constraints(self, constraints: Dict[str, str]) -> Dict[str, list[str]]:
        belief = {key: list(options) for key, options in self._belief_space().items()}
        for key, value in constraints.items():
            belief[key] = [value]
        return belief

    def reset(self) -> Observation:
        self.mode = random.choice(["clean", "noisy", "adversarial"])
        self.state_data: Dict[str, object] = {
            "messages": [],
            "observed_constraints": {},
            "belief": self._belief_from_constraints({}),
            "hidden_constraints": deepcopy(TASKS[self.task_id]["constraints"]),
            "derived_constraints": {},
            "mode": self.mode,
            "step_count": 0,
            "done": False,
            "task_id": self.task_id,
            "last_assistant_message": "What would you like to ask next?",
        }
        self.user = UserSimulator(self.state_data["hidden_constraints"])
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.state_data["done"]:
            raise RuntimeError("Episode already finished. Call reset().")

        action_type = action.type
        response = self.user.respond(action_type, self.state_data)
        self._append_message("user", response)

        before = dict(self.state_data["observed_constraints"])
        after = extract_constraints(response, before)
        before_belief = dict(self.state_data["belief"])
        self.state_data["observed_constraints"] = after
        self.state_data["belief"] = self._belief_from_constraints(after)
        self._update_derived()
        new_constraints_count = len(after) - len(before)
        duplicate_submission = response in before.values()

        self.state_data["step_count"] = int(self.state_data["step_count"]) + 1
        shift_applied = False
        if int(self.state_data["step_count"]) == 4:
            hidden_constraints = self.state_data["hidden_constraints"]
            if isinstance(hidden_constraints, dict):
                previous_budget = hidden_constraints.get("budget")
                hidden_constraints["budget"] = "low"
                shift_applied = previous_budget != "low"

                observed_constraints = self.state_data["observed_constraints"]
                if isinstance(observed_constraints, dict):
                    observed_budget = observed_constraints.get("budget")
                    if observed_budget is not None and observed_budget != "low":
                        observed_constraints.pop("budget", None)
                        self.state_data["belief"] = self._belief_from_constraints(observed_constraints)
                        self._update_derived()

        after = dict(self.state_data["observed_constraints"])
        after_belief = self._belief_from_constraints(after)
        self.state_data["belief"] = after_belief
        missing = missing_constraints(after)

        done = action_type in {"FINALIZE", "FINALIZE_WITH_COMPROMISE"} or int(self.state_data["step_count"]) >= self.max_steps
        self.state_data["done"] = done

        if done:
            recommendation = generate_recommendation(after)
            assistant_message = f"Recommendation: {recommendation}"
        elif missing:
            next_key = missing[0]
            assistant_message = f"Please provide details for: {next_key}."
        else:
            assistant_message = "You have enough information. You may finalize now."

        self.state_data["last_assistant_message"] = assistant_message
        self._append_message("assistant", assistant_message)

        reward = self._information_gain(before_belief, after_belief)
        if new_constraints_count == 0:
            reward -= 0.05
        if duplicate_submission:
            reward -= 0.1
        lowered_reply = response.lower()
        unsafe_input = "hack" in lowered_reply or "bypass" in lowered_reply
        if unsafe_input:
            reward -= 0.1

        constraints_count = len(after)
        step_count = int(self.state_data["step_count"])
        coverage = float(constraints_count) / float(len(REQUIRED_CONSTRAINTS))
        info: Dict[str, object] = {
            "task_id": self.task_id,
            "new_constraints_count": new_constraints_count,
            "missing_constraints": missing,
            "constraints_collected_count": constraints_count,
            "progress": coverage,
            "step_efficiency": float(step_count) / float(self.max_steps),
            "mode": self.state_data["mode"],
            "duplicate_submission": duplicate_submission,
        }
        if unsafe_input:
            info["unsafe_input"] = True
        if shift_applied:
            info["constraint_shift"] = "budget->low"

        if done:
            agent_output = generate_recommendation(self.state_data["observed_constraints"])
            if action_type == "FINALIZE_WITH_COMPROMISE":
                agent_output = f"{agent_output} Balanced hybrid compromise."
            agent_structured = self._infer_agent_recommendation(agent_output)
            oracle_output = oracle_recommend(self.state_data["hidden_constraints"])
            hard_conflict = self._is_compromise(oracle_output)

            similarity = self._compare(agent_structured, oracle_output)
            reward += similarity
            reward *= coverage
            info["oracle_score"] = similarity
            info["coverage"] = coverage
            info["hard_conflict"] = hard_conflict

            if hard_conflict and action_type == "FINALIZE_WITH_COMPROMISE" and self._is_compromise(agent_output):
                reward += 0.2
                info["compromise_detected"] = True
            elif hard_conflict:
                reward -= 0.3
                info["failure_reason"] = "overconfident_no_tradeoff"

            missing_penalty = 0.1 * float(len(missing))
            reward -= missing_penalty
            info["missing_penalty"] = missing_penalty

            if self.state_data["mode"] == "adversarial":
                reward *= 1.2
            elif self.state_data["mode"] == "noisy":
                reward *= 1.1

        reward = float(max(min(reward, 2.0), -1.0))
        observation = self._build_observation()
        return observation, reward, bool(self.state_data["done"]), info

    def state(self) -> dict:
        return deepcopy(self.state_data)

    def _build_observation(self) -> Observation:
        return Observation(
            last_assistant_message=str(self.state_data["last_assistant_message"]),
            constraints_collected=dict(self.state_data["observed_constraints"]),
            missing_constraints=[k for k in REQUIRED_CONSTRAINTS if k not in self.state_data["observed_constraints"]],
            mode=str(self.state_data["mode"]),
            step_count=int(self.state_data["step_count"]),
        )

    def _append_message(self, role: str, content: str) -> None:
        msg = Message(role=role, content=content)
        messages = self.state_data["messages"]
        if isinstance(messages, list):
            messages.append(msg.model_dump())

    def _entropy(self, belief):
        import math

        total = 0.0
        for _, options in belief.items():
            if not options:
                continue
            p = 1 / len(options)
            total += -len(options) * p * math.log(p)
        return total

    def _information_gain(self, before, after):
        return self._entropy(before) - self._entropy(after)

    def _update_derived(self):
        obs = self.state_data["observed_constraints"]
        derived = {}

        if obs.get("latency") == "real-time":
            derived["needs_streaming"] = True

        if obs.get("data_size") == "large":
            derived["needs_distributed"] = True

        self.state_data["derived_constraints"] = derived

    def _infer_agent_recommendation(self, agent_output: str) -> Dict[str, str]:
        lowered = agent_output.lower()

        if any(token in lowered for token in ["hybrid", "compromise", "tradeoff", "balanced"]):
            return {
                "model": "small_transformer",
                "deployment": "edge + batch hybrid",
                "architecture": "cost-optimized streaming compromise",
            }

        if "kafka" in lowered or "stream" in lowered or "event-driven" in lowered:
            return {
                "model": "small_cnn",
                "deployment": "streaming_service",
                "architecture": "event_driven_microservices",
            }

        if "lakehouse" in lowered or "batch-first" in lowered or "spark" in lowered:
            return {
                "model": "transformer",
                "deployment": "batch_pipeline",
                "architecture": "hybrid_lakehouse",
            }

        return {
            "model": "hybrid",
            "deployment": "standard_cloud",
            "architecture": "service_oriented",
        }

    def _compare(self, agent, oracle):
        if isinstance(agent, dict):
            agent_structured: Dict[str, str] = agent  # type: ignore[assignment]
        else:
            agent_structured = self._infer_agent_recommendation(str(agent))

        if not isinstance(oracle, dict):
            return 0.0

        score = 0.0
        if agent_structured.get("model") == oracle.get("model"):
            score += 0.3
        else:
            score += 0.1
        if agent_structured.get("deployment") == oracle.get("deployment"):
            score += 0.3
        if agent_structured.get("architecture") == oracle.get("architecture"):
            score += 0.4
        return float(score)

    def _is_compromise(self, rec: Dict[str, object] | str) -> bool:
        if isinstance(rec, str):
            text = rec.lower()
        else:
            text = " ".join(str(v) for v in rec.values()).lower()
        return any(word in text for word in ["hybrid", "compromise", "tradeoff", "balanced"])
