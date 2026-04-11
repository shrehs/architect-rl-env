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
        # Adversarial mode: track what was said to create contradictions
        self.adversarial_history: Dict[str, list] = {}  # key -> [value1, value2, ...]
        self.step_count = 0

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
    
    def _adversarial_conflicting_answer(self, key: str) -> str:
        """
        Feature: Adversarial Conflicting Answers
        
        Give contradictory values for the same constraint on repeated asks.
        Requires agent to:
        1. Detect the contradiction
        2. Ask follow-up clarifications
        3. Revise their solution
        
        Simulates real-world scenario: requirements change or stakeholders disagree.
        """
        # Track asking history for this key
        if key not in self.adversarial_history:
            self.adversarial_history[key] = []
        
        # Return true value first time, contradiction on follow-ups
        if len(self.adversarial_history[key]) == 0:
            value = self._value_for(key)
            self.adversarial_history[key].append(value)
            return self._constraint_text(key)
        else:
            # On follow-ups, give conflicting answer (partial lie)
            true_value = self._value_for(key)
            conflicting_values = {
                "latency": ["batch processing is fine", "actually we need real-time"],
                "accuracy": ["medium is enough", "wait, we need 99.99% actually"],
                "data_size": ["couple hundred GB max", "actually, expecting petabytes"],
                "update_frequency": ["daily is fine", "no wait, we need continuous updates"],
                "budget": ["low budget", "actually, budget is flexible/high"],
            }
            
            conflict = conflicting_values.get(key, [true_value])[self.step_count % 2]
            self.adversarial_history[key].append(conflict)
            
            # Frame as correction to test agent's revision behavior
            if len(self.adversarial_history[key]) > 1:
                return f"Actually, I need to correct myself earlier. {conflict}"
            return conflict
    
    def _adversarial_delayed_correction(self, key: str, step_num: int) -> str:
        """
        Feature: Delayed Corrections
        
        Say one thing early (wrong), then correct it later when agent is committed.
        Tests:
        1. Can agent revise their architecture mid-solution?
        2. Do they ask clarifying questions?
        3. Can they isolate impact of late changes?
        
        This simulates: "We forgot to mention..." or "New info just came in..."
        """
        # After step 8+, introduce a delayed contradiction
        if step_num >= 8 and key not in self.adversarial_history:
            # Simulate late discovery of requirement
            return f"Oh, one more thing I forgot to mention earlier: {self._constraint_text(key)} (This is critical!)"
        
        return self._constraint_text(key)

    def respond(self, action_type: str, state_data: Dict[str, Any]) -> str:
        normalized = action_type.strip().upper()
        mode = str(state_data.get("mode", "clean")).strip().lower()
        step_num = int(state_data.get("step_count", 0))

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

        if mode == "adversarial":
            # Use advanced adversarial techniques
            if normalized in ["ASK_LATENCY", "ASK_ACCURACY", "ASK_DATA_SIZE", "ASK_UPDATE_FREQUENCY", "ASK_BUDGET"]:
                # Map action to constraint key
                key_map = {
                    "ASK_LATENCY": "latency",
                    "ASK_ACCURACY": "accuracy",
                    "ASK_DATA_SIZE": "data_size",
                    "ASK_UPDATE_FREQUENCY": "update_frequency",
                    "ASK_BUDGET": "budget",
                }
                constraint_key = key_map.get(normalized)
                if constraint_key:
                    # Mix adversarial strategies
                    if step_num >= 8:
                        # Later in episode: inject delayed correction
                        return self._adversarial_delayed_correction(constraint_key, step_num)
                    else:
                        # Early: give conflicting answers on repeats
                        return self._adversarial_conflicting_answer(constraint_key)
            
            # Fallback to adversarial misdirection for other actions.
            return self._make_vague(self._plausible_misdirection(normalized, response))
        return response
