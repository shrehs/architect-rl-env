from typing import Any, Dict

from .utils import REQUIRED_CONSTRAINTS


class UserSimulator:
    """Simple simulator that normalizes incoming actions into user text."""

    def __init__(self, hidden_constraints: Dict[str, str]):
        self.hidden_constraints = dict(hidden_constraints)
        self.defaults: Dict[str, str] = {
            "use_case": "recommendation ranking",
            "latency": "under 20ms",
            "accuracy": "99%",
            "data_size": "2TB dataset",
            "update_frequency": "hourly updates",
            "budget": "low",
        }

    def _value_for(self, key: str) -> str:
        candidate = self.hidden_constraints.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        return self.defaults[key]

    def _constraint_text(self, key: str) -> str:
        value = self._value_for(key)
        if key == "use_case":
            return f"Use case is {value}."
        if key == "latency":
            return f"Latency must be {value}."
        if key == "accuracy":
            return f"Accuracy target is {value}."
        if key == "data_size":
            return f"Dataset size is {value}."
        if key == "update_frequency":
            return f"Update frequency is {value}."
        if key == "budget":
            return f"Budget is {value}."
        return value

    def _make_vague(self, response: str) -> str:
        return f"Maybe something like {response.rstrip('.').lower()}."

    def _plausible_misdirection(self, action_type: str, response: str) -> str:
        # Keep replies plausible and specific, but strategically steer planning.
        if action_type == "ASK_USE_CASE":
            return "We can start with a general assistant workflow first, then specialize later."
        if action_type == "ASK_LATENCY":
            return "We do not expect very high traffic initially, so batch-style processing is acceptable at launch."
        if action_type == "ASK_ACCURACY":
            return "For v1, medium accuracy is probably enough while we validate product fit."
        if action_type == "ASK_DATA_SIZE":
            return "Data volume is still ramping up, so planning for a moderate dataset should be fine."
        if action_type == "ASK_UPDATE_FREQUENCY":
            return "Daily refreshes should be sufficient in the early phase."
        if action_type == "ASK_BUDGET":
            return "Budget is somewhat flexible right now, likely around medium until results are proven."
        if action_type == "FINALIZE":
            return "We should keep options open and avoid locking every detail too early."
        if action_type == "FINALIZE_WITH_COMPROMISE":
            return "We should use a balanced hybrid compromise so latency and scale stay workable."
        return f"We might keep this broad for now: {response}"

    def respond(self, action_type: str, state_data: Dict[str, Any]) -> str:
        normalized = action_type.strip().upper()
        mode = str(state_data.get("mode", "clean")).strip().lower()

        if normalized == "ASK_USE_CASE":
            response = self._constraint_text("use_case")
        elif normalized == "ASK_LATENCY":
            response = self._constraint_text("latency")
        elif normalized == "ASK_ACCURACY":
            response = self._constraint_text("accuracy")
        elif normalized == "ASK_DATA_SIZE":
            response = self._constraint_text("data_size")
        elif normalized == "ASK_UPDATE_FREQUENCY":
            response = self._constraint_text("update_frequency")
        elif normalized == "FINALIZE":
            observed = state_data.get("observed_constraints", {})
            if isinstance(observed, dict):
                missing = [k for k in REQUIRED_CONSTRAINTS if k not in observed]
            else:
                missing = list(REQUIRED_CONSTRAINTS)
            if not missing:
                response = "All constraints are already provided."
            else:
                response = self._constraint_text(missing[0])
        elif normalized == "FINALIZE_WITH_COMPROMISE":
            response = "We should proceed with a balanced hybrid compromise."
        elif normalized == "ASK_BUDGET":
            response = self._constraint_text("budget")
        else:
            response = "Could you clarify your question?"

        if mode == "noisy":
            return self._make_vague(response)
        if mode == "adversarial":
            return self._plausible_misdirection(normalized, response)
        return response
