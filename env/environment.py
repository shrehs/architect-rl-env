from copy import deepcopy
import random
import math
from typing import Dict, Tuple, Any, List
from enum import Enum

from .models import Action, Message, Observation
from .tasks import TASKS
from .oracle import oracle_recommend
from .user_simulator import UserSimulator
from .utils import REQUIRED_CONSTRAINTS, extract_constraints, generate_recommendation, missing_constraints


MAX_CONTEXT_TOKENS = 300  # simulate LLM context limit


class ArchitectEnv:
    """Strict OpenEnv contract implementation.

    Required signatures:
    - reset(self) -> Observation
    - step(self, action: Action) -> Tuple[Observation, float, bool, dict]
    - state(self) -> dict
    """

    def __init__(self, task_id: str = "easy", max_steps: int = 30, exploration_alpha: float = 1.0):
        self.task_id = task_id if task_id in TASKS else "easy"
        self.max_steps = max_steps
        self.optimal_steps = {"easy": 6, "medium": 9, "hard": 12}  # Target efficient paths
        self.checkpoints = {}  # step_id -> state snapshot for branching
        self.episode_counter = 0  # Track episode number for structured logging
        
        # Reward weighting to reduce terminal dominance
        self.step_reward_weight = 1.0    # Full weight on step-level signals
        self.terminal_reward_weight = 0.5  # Reduced weight on terminal rewards
        
        # Feature 9: Trajectory diversity tracking
        self._matched_path_idx = -1  # Which valid path was matched (-1 = no match yet)
        
        # Feature 9 (Refined): Contextual diversity - track path frequency across episodes
        # Enables: bonus = 0.05 * (1 - frequency) ** alpha for adaptive exploration incentives
        self._path_frequency = {"primary": 0, "alternative_1": 0, "alternative_2": 0, "alternative_3": 0, "alternative_4": 0}
        self._total_episodes = 0  # Track episode count for frequency computation
        self._num_paths = len(self._path_frequency)  # Number of valid architectural paths
        self.exploration_alpha = exploration_alpha  # Temperature control: 1.0=standard, >1.0=stronger rare-path push
        
        # Learning Signal Perspective: RL Training Support
        # Track reward components per step for dense reward shaping
        self._episode_step_rewards: List[float] = []  # All step-level rewards in current episode
        self._episode_reward_components: List[Dict[str, float]] = []  # Breakdown of components
        self._global_step_count = 0  # Total steps across all episodes
        self._rolling_baseline = 0.0  # Running average of step rewards (for advantage computation)
        self._baseline_alpha = 0.01  # EMA smoothing factor for rolling baseline
        
        # Advanced RL features
        self._episode_baselines: List[float] = []  # Baseline values for GAE computation
        self._episode_advantages: List[float] = []  # Raw advantages for n-step returns
        self.gae_lambda = 0.95  # GAE lambda parameter (0=TD, 1=monte-carlo)
        self.gamma = 0.99  # Discount factor for multi-step returns
        self._action_history: List[str] = []  # Action log for entropy computation
        self._action_counts: Dict[str, int] = {}  # Count of each action type
        
        # Entropy behavior detection
        self._episode_entropy_history: List[float] = []  # Entropy values at each step
        self._entropy_behavior_pattern = "unknown"  # Pattern detected: "overconfident", "confused", "learning", "steady"
        self._entropy_decay_rate = 0.0  # Rate of entropy change over episode
        
        # Reward composition control (for combining learning signals)
        self.oracle_weight = 0.4  # Weight for oracle matching reward
        self.trajectory_weight = 0.3  # Weight for trajectory quality score
        self.process_weight = 0.3  # Weight for process rewards (efficiency, consistency, etc.)
        
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
    
    def _init_belief_state_with_confidence(self) -> Dict[str, Dict[str, any]]:
        """
        Feature: Calibrated Belief State Tracking
        
        Instead of treating answers as facts, track confidence in each constraint.
        Allows agents to be uncertain and update beliefs based on evidence.
        
        Format:
        {
            "use_case": {"value": "fraud detection", "confidence": 0.95},
            "latency": {"value": None, "confidence": 0.0},
            ...
        }
        
        Confidence interpretation:
        - 0.0-0.3: Very uncertain (might be noisy/adversarial)
        - 0.3-0.7: Moderate confidence (typical observation)
        - 0.7-1.0: High confidence (multiple confirmations or clear signal)
        """
        return {
            key: {"value": None, "confidence": 0.0}
            for key in self._belief_space().keys()
        }
    
    def _update_belief_with_confidence(
        self, 
        belief_state: Dict[str, Dict[str, any]], 
        constraint_key: str, 
        new_value: str,
        mode: str = "clean"
    ) -> tuple[Dict[str, Dict[str, any]], float]:
        """
        Update belief state with new observation.
        
        Returns: (updated_belief_state, confidence_change_signal)
        
        Confidence change based on:
        - First observation: +0.9 (strong signal)
        - Same value: +0.05 (reinforcement)
        - Conflicting value: -0.3 (reduce trust in new info)
        - Noisy mode: capped at +0.6 (less confident)
        - Adversarial mode: capped at +0.4 (very skeptical)
        """
        # Handle missing constraint keys (for optional system design constraints)
        if constraint_key not in belief_state:
            belief_state[constraint_key] = {"value": None, "confidence": 0.0}
        
        old_state = belief_state[constraint_key]
        old_value = old_state.get("value")
        old_confidence = old_state.get("confidence", 0.0)
        
        # Determine confidence boost based on consistency
        if old_value is None:
            # First observation
            confidence_boost = 0.9
        elif old_value == new_value:
            # Consistent observation (reinforcement)
            confidence_boost = 0.05
        else:
            # Conflicting observation (contradictory)
            confidence_boost = -0.3
        
        # Apply mode-dependent dampening
        if mode == "noisy":
            confidence_boost *= 0.7  # Less confident in noisy mode
        elif mode == "adversarial":
            confidence_boost *= 0.4  # Much less confident in adversarial
        
        # Update confidence and value
        new_confidence = min(max(old_confidence + confidence_boost, 0.0), 1.0)
        
        # If confidence drops below threshold after conflict, keep old value (skepticism)
        if new_confidence < old_confidence and old_value is not None:
            # Don't override with low-confidence contradictory info
            final_value = old_value
            final_confidence = old_confidence - 0.1  # Slight penalty for conflicting info
        else:
            final_value = new_value
            final_confidence = new_confidence
        
        belief_state[constraint_key] = {
            "value": final_value,
            "confidence": min(max(final_confidence, 0.0), 1.0)
        }
        
        confidence_signal = confidence_boost  # Return signal for reward
        return belief_state, confidence_signal

    def reset(self) -> Observation:
        self.episode_counter += 1
        self.mode = random.choice(["clean", "noisy", "adversarial"])
        self.state_data: Dict[str, object] = {
            "messages": [],
            "observed_constraints": {},
            "belief": self._belief_from_constraints({}),
            "belief_state": self._init_belief_state_with_confidence(),  # Feature: Calibrated belief tracking
            "hidden_constraints": deepcopy(TASKS[self.task_id]["constraints"]),
            "derived_constraints": {},
            "phase": "exploration",
            "mode": self.mode,
            "step_count": 0,
            "done": False,
            "task_id": self.task_id,
            "last_assistant_message": "What would you like to ask next?",
            "achieved_reward": 0.0,  # Track cumulative reward for regret
            "oracle_best_reward": 0.0,  # Track oracle benchmark
            "question_history": {},  # Track how many times each question asked {action_type: count}
            "constraint_discovery_order": [],  # Track order constraints discovered (logical progression)
            "previous_uncertainty": {},  # Track previous confidence levels for each constraint
            "constraint_value_history": {},  # Feature: Consistency tracking - all values each constraint has had
            "flip_flop_count": 0,  # Feature: Count of consistency violations (constraint value changes)
            "final_reasoning": "",  # Feature: Justification - capture agent's reasoning when finalizing
            "final_recommendation": "",  # Feature: Store the final recommendation for analysis
        }
        self.checkpoints = {}  # Reset checkpoints for new episode
        self.user = UserSimulator(self.state_data["hidden_constraints"])
        
        # Feature: Failure Case Analysis
        # Track detailed information about when and why failures occur
        self.constraint_first_discovery_step = {}  # {constraint: step_number}
        self.step_oracle_history = []  # Oracle score at each step
        self.critical_failure_step = None  # Step where oracle collapsed
        self.constraint_missing_at_end = []  # Constraints never discovered
        self.failure_type = "success"  # success, early_mistake, late_misjudgment, incomplete_exploration
        
        # Feature: Learning Signal Perspective - Reset episode-level tracking
        # For RL training infrastructure
        self._episode_step_rewards = []  # All step-level rewards in this episode
        self._episode_reward_components = []  # Component breakdown per step
        self._episode_baselines = []  # Baseline values for GAE
        self._episode_advantages = []  # Raw advantages
        self._action_history = []  # Reset action log
        self._action_counts = {}  # Reset action counts
        
        # Entropy tracking for behavior detection
        self._episode_entropy_history = []  # Entropy at each step
        self._entropy_behavior_pattern = "unknown"
        self._entropy_decay_rate = 0.0
        
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.state_data["done"]:
            raise RuntimeError("Episode already finished. Call reset().")

        # Feature 5: Phase-dependent action gating
        action_type = action.type
        
        # Advanced RL: Track action entropy for exploration monitoring
        action_entropy, entropy_info = self._compute_action_entropy(action_type)
        
        phase = str(self.state_data.get("phase", "exploration"))
        if self._is_action_gated_in_phase(action_type, phase):
            # Soften penalty for phase violations: -0.15 instead of -0.3
            observation = self._build_observation()
            return observation, -0.15, False, {"error": f"Action {action_type} not allowed in {phase}"}
        
        response = self.user.respond(action_type, self.state_data)
        self._append_message("user", response)

        before = dict(self.state_data["observed_constraints"])
        after = extract_constraints(response, before)
        
        # Feature 6: Stochastic observation noise (20% chance per constraint corrupted)
        # Makes environment realistic: sensors fail, misinterpretations occur
        after, noisy_constraints = self._apply_observation_noise(after)
        
        before_belief = dict(self.state_data["belief"])
        self.state_data["observed_constraints"] = after
        self.state_data["belief"] = self._belief_from_constraints(after)
        
        # Feature: Update calibrated belief state with confidence tracking
        belief_state = self.state_data.get("belief_state", self._init_belief_state_with_confidence())
        mode = str(self.state_data.get("mode", "clean"))
        for constraint_key in after.keys():
            if constraint_key in after:
                constraint_value = after[constraint_key]
                # Update belief for ALL constraints (new or re-observed)
                belief_state, _signal = self._update_belief_with_confidence(
                    belief_state, constraint_key, constraint_value, mode
                )
        self.state_data["belief_state"] = belief_state
        
        self._update_derived()
        new_constraints_count = len(after) - len(before)
        duplicate_submission = response in before.values()

        self.state_data["step_count"] = int(self.state_data["step_count"]) + 1
        shift_applied = False
        if int(self.state_data["step_count"]) == 5:
            hidden_constraints = self.state_data["hidden_constraints"]
            if isinstance(hidden_constraints, dict):
                previous_budget = hidden_constraints.get("budget")
                hidden_constraints["budget"] = "low"
                shift_applied = previous_budget != "low"
                if shift_applied:
                    self._append_message("user", "Actually, budget is tighter than expected.")

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
        
        # Early termination: insufficient exploration
        step_count = int(self.state_data["step_count"])
        missing_count = len(missing)
        if missing_count > 4 and step_count > 15:
            done = True
            early_termination = "insufficient_exploration"
        else:
            early_termination = None
        
        self.state_data["done"] = done
        constraints_count = len(after)
        previous_phase = str(self.state_data.get("phase", "exploration"))
        if constraints_count >= 3:
            self.state_data["phase"] = "refinement"
        if done:
            self.state_data["phase"] = "decision"
        phase_transition = previous_phase != str(self.state_data["phase"])

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

        visible_messages = self._prune_messages(self.state_data["messages"])

        step_count = int(self.state_data["step_count"])
        coverage = float(constraints_count) / float(len(REQUIRED_CONSTRAINTS))
        
        # Compute counterfactuals EARLY for reward calculation
        counterfactuals: Dict[str, float] = {}
        for sim_action in ["ASK_LATENCY", "ASK_ACCURACY"]:
            sim_state = deepcopy(self.state_data)
            sim_response = self.user.respond(sim_action, sim_state)
            sim_before = dict(sim_state["observed_constraints"])
            sim_after = extract_constraints(sim_response, sim_before)
            sim_before_belief = self._belief_from_constraints(sim_before)
            sim_after_belief = self._belief_from_constraints(sim_after)
            simulated_gain = self._information_gain(sim_before_belief, sim_after_belief)
            counterfactuals[sim_action] = float(simulated_gain)
        
        # Now compute actual gain and counterfactual reward
        actual_gain = self._information_gain(before_belief, after_belief)
        counterfactual_gains = list(counterfactuals.values())
        best_counterfactual_gain = max(counterfactual_gains) if counterfactual_gains else 0.0

        # === Learning Signal Perspective: Track reward components separately ===
        # This enables dense reward shaping and advantage signal computation for RL
        information_gain = self._information_gain(before_belief, after_belief)
        
        # Phase-specific per-step rewards
        phase = str(self.state_data.get("phase", "exploration"))
        exploration_reward = 0.0
        refinement_reward = 0.0
        
        if phase == "exploration":
            exploration_reward = self._exploration_reward(new_constraints_count, action_type)
        elif phase == "refinement":
            refinement_reward = self._refinement_reward(self._information_gain(before_belief, after_belief), duplicate_submission)
        
        # Feature: Belief Calibration Reward (new constraint update tracking)
        belief_state = self.state_data.get("belief_state", {})
        belief_calibration_reward = self._evaluate_belief_calibration(belief_state) * 0.5  # Scale down
        
        # Adversarial resilience bonus
        mode = str(self.state_data.get("mode", "clean"))
        adversarial_bonus = 0.0
        if mode == "adversarial" and new_constraints_count > 0:
            adversarial_bonus = 0.15
        
        # Feature: Reward agents for handling contradictory information
        contradiction_handling_reward = self._detect_and_reward_contradiction_handling(action_type, self.state_data)
        
        # Feature: Reward process quality - efficient, logical question sequences
        efficiency_reward, efficiency_metrics = self._evaluate_question_efficiency(
            action_type, 
            self.state_data.get("messages", []),
            belief_state,
            self.state_data.get("hidden_constraints", {})
        )
        
        # Feature: Consistency Score - reward stable reasoning
        consistency_reward, consistency_metrics = self._evaluate_consistency(
            self.state_data.get("observed_constraints", {})
        )
        
        # Update previous_uncertainty for next step
        previous_uncertainty = self.state_data.get("previous_uncertainty", {})
        for key, state in belief_state.items():
            previous_uncertainty[key] = state.get("confidence", 0.0)
        
        # Accumulate step-level penalties
        step_penalties = 0.0
        
        # Soften step-level penalties: use smaller continuous adjustments
        if new_constraints_count == 0:
            if phase == "refinement":
                step_penalties -= 0.05  # Softer penalty
            else:
                step_penalties -= 0.02  # Very soft penalty for exploration
        
        if duplicate_submission:
            step_penalties -= 0.05  # Softer duplicate penalty
        
        # Soften context pruning penalty
        if len(self.state_data["messages"]) > len(visible_messages):
            step_penalties -= 0.02  # Very soft penalty
        
        # Safety penalty
        lowered_reply = response.lower()
        unsafe_input = "hack" in lowered_reply or "bypass" in lowered_reply
        if unsafe_input:
            step_penalties -= 0.05  # Softer safety penalty
        
        # Early termination penalty
        if action_type in {"FINALIZE", "FINALIZE_WITH_COMPROMISE"} and int(self.state_data["step_count"]) < 10:
            step_penalties -= 0.2
        
        # Budget reward bonus
        if "budget" in after and before.get("budget") != after.get("budget"):
            exploration_reward += 0.1  # Track as exploration bonus
        
        # Feature 1: Counterfactual reward (softened scaling)
        counterfactual_reward = 0.10 * (best_counterfactual_gain - actual_gain)
        
        # === Compute Dense Rewards with Component Breakdown ===
        reward_components, step_reward = self._compute_dense_rewards(
            information_gain=information_gain,
            exploration_reward=exploration_reward,
            refinement_reward=refinement_reward,
            belief_calibration_reward=belief_calibration_reward,
            contradiction_handling_reward=contradiction_handling_reward,
            efficiency_reward=efficiency_reward,
            consistency_reward=consistency_reward,
            counterfactual_reward=counterfactual_reward,
            step_penalties=step_penalties + adversarial_bonus,  # Include adversarial bonus with penalties
            phase=phase,
            action_type=action_type
        )
        
        # === Compute Advantage Signal for RL ===
        advantage, advantage_signal = self._compute_advantage_signal(step_reward)
        
        # Track for this episode (for baseline updates)
        self._episode_step_rewards.append(step_reward)
        self._episode_reward_components.append(reward_components)
        self._episode_baselines.append(float(advantage_signal["baseline"]))  # For GAE/n-step
        self._episode_advantages.append(float(advantage))  # For advanced RL analysis
        self._global_step_count += 1
        
        # Apply step-level reward weight
        reward = step_reward * self.step_reward_weight
        
        # Feature 4: Lightweight checkpoint system at step 3 for branching visibility
        if step_count == 3:
            self.checkpoints[3] = deepcopy(self.state_data)
            info_msg = "Checkpoint created at exploration stage (branch point)"
        else:
            info_msg = None
        
        info: Dict[str, object] = {
            "task_id": self.task_id,
            "new_constraints_count": new_constraints_count,
            "missing_constraints": missing,
            "constraints_collected_count": constraints_count,
            "progress": coverage,
            "step_efficiency": float(step_count) / float(self.max_steps),
            "mode": self.state_data["mode"],
            "phase": self.state_data["phase"],
            "phase_transition": phase_transition,
            "duplicate_submission": duplicate_submission,
            "counterfactual_gain_diff": float(best_counterfactual_gain - actual_gain),
            "counterfactual_reward": float(counterfactual_reward),
            "contradiction_handling_reward": float(contradiction_handling_reward),
            "efficiency_reward": float(efficiency_reward),
            "repeated_questions": int(efficiency_metrics.get("repeated_questions", 0)),
            "uncertainty_resolved": int(efficiency_metrics.get("uncertainty_resolved", 0)),
            "irrelevant_questions": int(efficiency_metrics.get("irrelevant_questions", 0)),
            "logical_sequences": int(efficiency_metrics.get("logical_sequences", 0)),
            "consistency_reward": float(consistency_reward),
            "consistency_violations": int(consistency_metrics.get("consistency_violations", 0)),
            "stable_constraints": int(consistency_metrics.get("stable_constraints", 0)),
            "consistency_score": float(consistency_metrics.get("consistency_score", 1.0)),
            "noisy_constraints": noisy_constraints,
            "num_noisy_observations": len(noisy_constraints),
        }
        if info_msg:
            info["checkpoint"] = info_msg
        if early_termination:
            info["early_termination"] = early_termination
        
        info["counterfactuals"] = counterfactuals

        if unsafe_input:
            info["unsafe_input"] = True
        if shift_applied:
            info["constraint_shift"] = "budget->low"
        
        # === Add Learning Signal Information for RL Training ===
        # Dense reward breakdown - enables component-level analysis
        info["reward_components"] = reward_components
        info["step_reward"] = float(step_reward)
        
        # Advantage signals - for policy gradient algorithms
        info["advantage"] = float(advantage)
        info["advantage_signal"] = advantage_signal
        info["baseline"] = float(advantage_signal["baseline"])
        
        # Support for training infrastructure
        info["global_step"] = self._global_step_count
        info["rolling_baseline"] = float(self._rolling_baseline)

        if done:
            agent_output = generate_recommendation(self.state_data["observed_constraints"])
            if action_type == "FINALIZE_WITH_COMPROMISE":
                agent_output = f"{agent_output} Balanced hybrid compromise."
            
            # Feature: Capture final reasoning and recommendation for justification scoring
            self.state_data["final_reasoning"] = f"Action: {action_type}"
            self.state_data["final_recommendation"] = agent_output
            
            agent_structured = self._infer_agent_recommendation(agent_output)
            oracle_output = oracle_recommend(self.state_data["hidden_constraints"])
            hard_conflict = len(self.state_data["observed_constraints"]) >= 3 and self._is_compromise(oracle_output)
            
            # Feature 2: Trajectory efficiency reward (scaled by terminal weight)
            # Agents that solve faster get bonus
            optimal_target = self.optimal_steps.get(self.task_id, 9)
            efficiency_reward = 0.5 * (optimal_target / max(step_count, 1)) * self.terminal_reward_weight
            info["efficiency_reward"] = float(efficiency_reward)
            reward = step_reward + efficiency_reward

            similarity = self._compare(agent_structured, oracle_output)
            terminal_similarity = (similarity * coverage) * self.terminal_reward_weight
            reward += terminal_similarity
            info["oracle_score"] = similarity
            
            # Feature: Failure Analysis - Diagnose why oracle_score might be low
            if similarity < 1.0:
                failure_analysis = self._analyze_failure(
                    agent_structured=agent_structured,
                    oracle_output=oracle_output,
                    similarity=similarity,
                    coverage=coverage,
                    observed_constraints=self.state_data["observed_constraints"],
                    hidden_constraints=self.state_data["hidden_constraints"],
                    final_reasoning=self.state_data.get("final_reasoning", ""),
                    agent_output=agent_output
                )
                info["failure_analysis"] = failure_analysis
            
            info["coverage"] = coverage
            info["hard_conflict"] = hard_conflict
            
            # Feature: Explicit Tradeoff Reasoning
            # Reward agents for detecting and reasoning about constraint conflicts
            tradeoff_reward = self._evaluate_constraint_conflicts(self.state_data["observed_constraints"])
            reward += tradeoff_reward * self.terminal_reward_weight
            conflicts, tradeoff_score = self._detect_tradeoffs(self.state_data["observed_constraints"])
            info["detected_tradeoffs"] = conflicts
            info["tradeoff_count"] = len(conflicts)
            info["tradeoff_reasoning_reward"] = float(tradeoff_reward)
            info["tradeoff_score"] = float(tradeoff_score)
            
            # Feature: Global Efficiency Score (Terminal-level)
            # Prevent "slow but safe" agents from gaming the system
            global_efficiency_reward, global_efficiency_metrics = self._evaluate_global_efficiency(step_count, self.task_id)
            reward += global_efficiency_reward * self.terminal_reward_weight
            info["global_efficiency_score"] = float(global_efficiency_metrics.get("efficiency_score", 0.0))
            info["global_efficiency_reward"] = float(global_efficiency_reward)
            info["global_optimal_steps"] = int(global_efficiency_metrics.get("optimal_steps", 9))
            
            # Feature: Calibrated Belief State
            # Reward agents for having well-calibrated confidence in their beliefs
            belief_state = self.state_data.get("belief_state", {})
            calibration_reward, calibration_metrics = self._compute_calibration_reward(
                belief_state, 
                self.state_data["hidden_constraints"]
            )
            reward += calibration_reward
            info["calibration_reward"] = float(calibration_reward)
            info["avg_confidence"] = float(calibration_metrics.get("avg_confidence", 0.0))
            info["brier_score"] = float(calibration_metrics.get("brier_score", 0.0))
            info["correct_high_confidence"] = int(calibration_metrics.get("correct_high_confidence", 0))
            info["incorrect_high_confidence"] = int(calibration_metrics.get("incorrect_high_confidence", 0))
            
            # Feature 9: Track which valid path was matched
            if self._matched_path_idx >= 0:
                path_names = ["primary", "alternative_1", "alternative_2", "alternative_3", "alternative_4"]
                path_name = path_names[self._matched_path_idx] if self._matched_path_idx < len(path_names) else f"alternative_{self._matched_path_idx}"
                info["matched_trajectory"] = path_name
                info["trajectory_diversity_index"] = self._matched_path_idx
                
                # Feature 9 (Refined+): Contextual diversity bonus with Laplace smoothing & temperature control
                # Laplace smoothing: frequency = (count + 1) / (total + num_paths)
                # Temperature: bonus = 0.05 * (1 - smoothed_frequency) ** alpha
                # Time decay: bonus *= (1 / sqrt(total + 1))
                # - alpha=1.0: standard behavior (explores rarer paths)
                # - alpha>1.0: stronger push to rare paths (more aggressive exploration)
                # - alpha<1.0: softer effect (less exploration emphasis)
                # - Time decay: early episodes get higher bonus, naturally decays over time
                if self._matched_path_idx > 0 and isinstance(oracle_output, dict):
                    # Compute smoothed frequency of this path (Laplace smoothing)
                    path_count = self._path_frequency.get(path_name, 0)
                    total = self._total_episodes + 1  # Include current episode
                    # Laplace smoothing: ensures no path gets zero probability
                    path_frequency = (path_count + 1) / (total + self._num_paths)
                    
                    # Contextual bonus with temperature control
                    # bonus scales from 0.05 (new) to near-0 (very common), shaped by alpha
                    frequency_penalty = (1.0 - path_frequency) ** self.exploration_alpha
                    
                    # Time-aware decay: early episodes encourage exploration, later episodes settle down
                    # This prevents exploration bonuses from dominating in late training
                    time_decay = 1.0 / math.sqrt(total + 1)
                    contextual_bonus = 0.05 * frequency_penalty * time_decay
                    alternative_bonus = contextual_bonus * self.terminal_reward_weight
                    reward += alternative_bonus
                    
                    info["trajectory_diversity_bonus"] = alternative_bonus
                    info["path_frequency"] = float(path_frequency)
                    info["time_decay_factor"] = float(time_decay)
                    info["contextual_bonus_scale"] = float(frequency_penalty)  # Multiplier applied with temperature
                    info["exploration_alpha"] = float(self.exploration_alpha)
                    
                    # Feature 10: Policy entropy - measure diversity of path selection
                    # entropy = -sum(p * log(p)) across all paths
                    # High entropy = diverse exploration, Low entropy = converged policy
                    path_probs = [
                        (self._path_frequency.get(path, 0) + 1) / (self._total_episodes + 1 + self._num_paths)
                        for path in ["primary", "alternative_1", "alternative_2", "alternative_3", "alternative_4"]
                    ]
                    entropy = -sum(p * math.log(p + 1e-8) for p in path_probs)
                    info["policy_entropy"] = float(entropy)
                    info["entropy_normalized"] = float(entropy / math.log(self._num_paths + 1e-8))  # Normalized 0-1
            
            # Track available valid paths for learning
            if isinstance(oracle_output, dict) and "valid_paths" in oracle_output:
                info["valid_path_count"] = oracle_output.get("path_count", 1)
            
            # Feature 3: Regret signal (scaled by terminal weight)
            # Oracle best reward = optimal similarity + efficiency bonus + coverage scaling
            oracle_best_similarity = 0.8  # Assume oracle gets near-perfect match
            oracle_best_efficiency = 0.5 * (optimal_target / optimal_target) if optimal_target > 0 else 0.0
            oracle_best_reward = (oracle_best_similarity + oracle_best_efficiency) * coverage
            achieved_reward = reward
            regret = oracle_best_reward - achieved_reward
            beta = 0.1  # Regret weight
            regret_penalty = beta * regret * self.terminal_reward_weight  # Scale by terminal weight
            reward -= regret_penalty
            info["regret"] = float(regret)
            info["oracle_best_reward"] = float(oracle_best_reward)
            info["regret_penalty"] = float(regret_penalty)

            # Improved compromise detection: explicit action OR tradeoff reasoning in text
            recommendation_has_tradeoff = self._is_compromise(agent_output)
            explicit_compromise_action = action_type == "FINALIZE_WITH_COMPROMISE"
            compromise_detected = hard_conflict and recommendation_has_tradeoff
            
            # Add decision-phase specific rewards (scaled by terminal weight)
            decision_bonus = self._decision_reward(agent_structured, oracle_output, hard_conflict, compromise_detected)
            decision_bonus *= self.terminal_reward_weight  # Scale terminal rewards
            reward += decision_bonus
            
            if compromise_detected:
                info["compromise_detected"] = True
                # Bonus for explicit action vs implicit reasoning
                if explicit_compromise_action:
                    info["compromise_type"] = "explicit"
                else:
                    info["compromise_type"] = "implicit_reasoning"
                    implicit_bonus = 0.3 * self.terminal_reward_weight
                    reward += implicit_bonus
            elif hard_conflict:
                info["failure_reason"] = "overconfident_no_tradeoff"

            # Continuous missing constraint penalty (softer gradient, scaled by terminal weight)
            missing_penalty = 0.05 * float(len(missing)) * self.terminal_reward_weight
            reward -= missing_penalty
            info["missing_penalty"] = missing_penalty

            if self.state_data["mode"] == "adversarial":
                reward *= 1.0  # No terminal boost; resilience handled via intermediate bonuses
            elif self.state_data["mode"] == "noisy":
                reward *= 1.1
            
            # Feature 4: Track available checkpoints for branching
            if self.checkpoints:
                info["available_checkpoints"] = list(self.checkpoints.keys())
                info["trajectory_branches"] = f"Can replay from step {list(self.checkpoints.keys())[0]} with alternative actions"

            # Feature: Trajectory Score (Composite Quality Measure)
            # Combines consistency, efficiency, and recovery into single trajectory quality metric
            # consistency_score, global_efficiency_score, recovery_score are already calculated above
            
            consistency_score = info.get("consistency_score", 1.0)
            global_efficiency_score = info.get("global_efficiency_score", 0.0)
            recovery_score, recovery_metrics = self._evaluate_recovery(self.state_data, similarity)
            
            # Weighted composition:
            # - 0.4 * consistency: prioritize stable reasoning
            # - 0.3 * efficiency: important but secondary
            # - 0.3 * recovery: resilience to adversarial/noisy info
            trajectory_score = (
                0.4 * consistency_score +
                0.3 * global_efficiency_score +
                0.3 * recovery_score
            )
            
            # Feature: Exploration Completeness
            # Ensure agents ask ALL important questions
            # Prevents: Agents skipping important constraints
            # Reward: agents for discovering as many constraints as possible
            TOTAL_CONSTRAINTS = 6  # use_case, latency, accuracy, data_size, update_frequency, budget
            discovered_constraints = len(self.state_data.get("observed_constraints", {}))
            exploration_completeness = min(1.0, discovered_constraints / float(TOTAL_CONSTRAINTS))
            exploration_bonus = 0.2 * exploration_completeness
            trajectory_score += exploration_bonus
            
            info["exploration_completeness"] = float(exploration_completeness)
            info["exploration_bonus"] = float(exploration_bonus)
            info["discovered_constraints"] = discovered_constraints
            
            # CRITICAL: Penalty for "lucky agents"
            # If oracle_score < 0.3, discount trajectory_score by 50%
            # Ensures: Good process alone ≠ success
            # Prevents: Agents from gaming trajectory without actual accuracy
            oracle_score = info.get("oracle_score", 0.0)
            if oracle_score < 0.3:
                trajectory_score *= 0.5  # Harsh penalty: loses 50% trajectory value
            
            info["recovery_score"] = float(recovery_score)
            info["trajectory_score"] = float(trajectory_score)
            
            # === Combine Learning Signals: Oracle + Trajectory + Process ===
            # Weighted combination for final reward signal used in RL training
            oracle_score = info.get("oracle_score", 0.0)
            
            # Process reward: average of per-step efficiency and process signals
            if self._episode_step_rewards:
                avg_process_reward = sum(self._episode_step_rewards) / len(self._episode_step_rewards)
            else:
                avg_process_reward = 0.0
            
            # Normalize process reward to 0-1 range for fair combination
            # Most step rewards range from -0.2 to 2.5, so normalize by ~2.0
            normalized_process = max(0.0, min(1.0, (avg_process_reward + 0.5) / 2.0))
            
            # Compute combined reward
            combined_reward = self._compute_combined_reward(
                oracle_score=oracle_score,
                trajectory_score=trajectory_score,
                process_reward=normalized_process
            )
            
            info["process_reward"] = float(normalized_process)
            info["combined_reward"] = float(combined_reward)
            info["oracle_weight"] = float(self.oracle_weight)
            info["trajectory_weight"] = float(self.trajectory_weight)
            info["process_weight"] = float(self.process_weight)
            
            # Feature: Trajectory-Level Evaluation (Step-wise Reasoning Quality)
            # These metrics evaluate the QUALITY of the path taken, not just the outcome
            
            # 1. Delta Information Gain: measure step-wise uncertainty reduction
            information_gain_score, ig_metrics = self._evaluate_delta_information_gain(self.state_data)
            info["information_gain_score"] = float(information_gain_score)
            info["early_discoveries"] = int(ig_metrics.get("early_discoveries", 0))
            info["late_discoveries"] = int(ig_metrics.get("late_discoveries", 0))
            
            # 2. Constraint Utilization: check if all discovered constraints are used in decision
            utilization_score, util_metrics = self._evaluate_constraint_utilization(
                self.state_data.get("observed_constraints", {}),
                self.state_data.get("final_recommendation", "")
            )
            info["utilization_score"] = float(utilization_score)
            info["constraints_used"] = int(util_metrics.get("constraints_used", 0))
            info["constraints_observed"] = int(util_metrics.get("constraints_observed", 0))
            
            # 3. Redundancy Score: measure trajectory efficiency (non-repeated questions)
            redundancy_score, red_metrics = self._evaluate_redundancy_score(self.state_data)
            info["redundancy_score"] = float(redundancy_score)
            info["total_questions"] = int(red_metrics.get("total_questions", 0))
            info["repeated_questions"] = int(red_metrics.get("repeated_questions", 0))
            
            # Integrate trajectory-level metrics into terminal reward
            # These add small bonuses for high-quality reasoning paths
            trajectory_quality_bonus = (
                0.05 * information_gain_score +  # +0.05 max for good information ordering
                0.05 * utilization_score +        # +0.05 max for using all constraints
                0.05 * redundancy_score           # +0.05 max for no redundancy
            )
            trajectory_quality_reward = trajectory_quality_bonus * self.terminal_reward_weight
            reward += trajectory_quality_reward
            info["trajectory_quality_bonus"] = float(trajectory_quality_bonus)
            info["trajectory_quality_reward"] = float(trajectory_quality_reward)
            
            # Feature: Decision Justification Score
            # Reward explicit reasoning and tradeoff awareness in final decision
            justification_score, justification_metrics = self._evaluate_justification(
                info,
                self.state_data.get("observed_constraints", {}),
                conflicts  # Already computed above
            )
            
            # Add justification bonus to terminal reward
            justification_reward = 0.25 * justification_score * self.terminal_reward_weight
            reward += justification_reward
            
            info["justification_score"] = float(justification_score)
            info["justification_reward"] = float(justification_reward)
            info["constraints_mentioned"] = int(justification_metrics.get("constraints_mentioned", 0))
            info["tradeoffs_mentioned"] = int(justification_metrics.get("tradeoffs_mentioned", 0))
            info["coverage_score"] = float(justification_metrics.get("coverage_score", 0.0))
            info["tradeoff_awareness_score"] = float(justification_metrics.get("tradeoff_awareness_score", 0.0))

        reward = float(max(min(reward, 2.0), -1.0))
        
        # Feature 9 (Refined): Update path frequency tracking for contextual diversity
        # After episode ends, record which path was chosen for future contextual bonus computation
        if done and self._matched_path_idx >= 0:
            path_names = ["primary", "alternative_1", "alternative_2", "alternative_3", "alternative_4"]
            path_name = path_names[self._matched_path_idx] if self._matched_path_idx < len(path_names) else f"alternative_{self._matched_path_idx}"
            if path_name in self._path_frequency:
                self._path_frequency[path_name] += 1
            self._total_episodes += 1
        
        # === Advanced RL Signals: GAE, n-step returns, entropy ===
        if done:
            episode_summary = self._finalize_episode_learning_signals()
            info["episode_summary"] = episode_summary
            
            # Add step-level advanced signals
            info["action_entropy"] = action_entropy  # Current step entropy
            info["entropy_info"] = entropy_info  # Full entropy statistics
        
        observation = self._build_observation()
        return observation, reward, bool(self.state_data["done"]), info

    def state(self) -> dict:
        return deepcopy(self.state_data)

    def _build_observation(self) -> Observation:
        visible_messages = self._prune_messages(self.state_data["messages"])
        last_visible_assistant = str(self.state_data["last_assistant_message"])
        for msg in reversed(visible_messages):
            if msg.get("role") == "assistant":
                last_visible_assistant = str(msg.get("content", last_visible_assistant))
                break

        return Observation(
            last_assistant_message=last_visible_assistant,
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

    def _prune_messages(self, messages):
        total = 0
        pruned = []
        for msg in reversed(messages):
            tokens = len(str(msg.get("content", "")).split())
            if total + tokens > MAX_CONTEXT_TOKENS:
                break
            pruned.insert(0, msg)
            total += tokens
        return pruned

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

    def _compute_similarity(self, agent_structured: Dict[str, str], oracle_path: Dict[str, object]) -> float:
        """
        Compute similarity between agent and ONE oracle path.
        FIXED: Require actual matches, penalize mismatches more heavily.
        
        Scoring:
        - Exact matches: +0.33 each (model, deployment, architecture)
        - Mismatches: -0.1 each (no partial credit)
        - Generic detection: -0.3 penalty
        
        Result: Random agents get 0.0-0.2, heuristic agents get 0.5-0.8
        """
        score = 0.0
        
        # Exact matches only
        if agent_structured.get("model") == oracle_path.get("model"):
            score += 0.33
        else:
            # Penalize mismatch instead of giving partial credit
            score -= 0.1
        
        if agent_structured.get("deployment") == oracle_path.get("deployment"):
            score += 0.33
        else:
            score -= 0.1
        
        if agent_structured.get("architecture") == oracle_path.get("architecture"):
            score += 0.34
        else:
            score -= 0.1
        
        # Penalize generic architectures
        arch = agent_structured.get("architecture", "").lower()
        generic_terms = ["microservice", "api", "database", "standard", "modular", "generic"]
        if any(term in arch for term in generic_terms):
            score -= 0.3
        
        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, score)))
    
    def _compare(self, agent, oracle):
        """
        CRITICAL FIX: Restore gradient in oracle scoring.
        
        Instead of binarizing to 1.0, keep the continuous similarity score.
        This restores the learning signal:
        
        Random agent → ~0.2-0.4 (lucky guesses on generic architectures)
        Heuristic agent → ~0.6-0.8 (consistent constraint-aware choices)
        Optimal agent → ~0.95-1.0 (precise matches with reasoning)
        
        SUCCESS is now defined as oracle_score >= 0.8, not >= 0.6
        This ensures only quality policies succeed.
        
        Diversity bonuses remain orthogonal (applied separately).
        """
        if isinstance(agent, dict):
            agent_structured: Dict[str, str] = agent  # type: ignore[assignment]
        else:
            agent_structured = self._infer_agent_recommendation(str(agent))

        if not isinstance(oracle, dict):
            return 0.0
        
        # Backward compatibility: old oracle format (single path)
        if "valid_paths" not in oracle:
            score = self._compute_similarity(agent_structured, oracle)
            return float(score)
        
        # New format: check against ALL valid paths
        valid_paths = oracle.get("valid_paths", [])
        best_score = 0.0
        best_path_idx = -1
        
        for idx, valid_path in enumerate(valid_paths):
            score = self._compute_similarity(agent_structured, valid_path)
            if score > best_score:
                best_score = score
                best_path_idx = idx
        
        # FIXED: Keep the continuous score instead of binarizing!
        # This restores the learning signal and prevents random policies from appearing smart.
        #
        # Score interpretation:
        # - 0.0-0.3: Generic/random (no real match, just generic patterns)
        # - 0.3-0.6: Partial match (some components right, but loose)
        # - 0.6-0.8: Good match (mostly correct, constraint-aware)
        # - 0.8-1.0: Excellent match (precise, reasoning-backed)
        
        # Track which path for SEPARATE diversity bonus (orthogonal signal)
        self._matched_path_idx = best_path_idx if best_score >= 0.3 else -1
        self._best_similarity = best_score
        
        return float(best_score)

    def _is_compromise(self, rec: Dict[str, object] | str) -> bool:
        if isinstance(rec, str):
            text = rec.lower()
        else:
            text = " ".join(str(v) for v in rec.values()).lower()
        return any(word in text for word in ["hybrid", "compromise", "tradeoff", "balanced"])

    def _exploration_reward(self, new_constraints_count: int, action_type: str) -> float:
        """Phase-specific reward for exploration: continuous based on new constraints found."""
        if not action_type.startswith("ASK_"):
            return 0.0
        
        # Continuous reward: scale with number of constraints discovered
        # 1 constraint = +0.5, 2+ = +1.0, 0 = -0.1
        if new_constraints_count >= 2:
            return 1.0
        elif new_constraints_count == 1:
            return 0.5
        else:
            # Small penalty for unproductive questions (not harsh)
            return -0.1

    def _refinement_reward(self, info_gain: float, duplicate: bool) -> float:
        """Phase-specific reward for refinement: continuous based on information gain."""
        # Continuous reward scaled by information gain
        # Higher info_gain → higher reward (smooth gradient)
        reward = min(info_gain * 5.0, 1.0)  # Scale info_gain, cap at 1.0
        
        # Soften duplicate penalty: -0.15 instead of -0.3
        if duplicate:
            reward -= 0.15
        
        return reward

    def _decision_reward(self, agent_structured: Dict[str, str], oracle_output: Dict[str, object], hard_conflict: bool, compromise_detected: bool) -> float:
        """Phase-specific reward for decision: continuous similarity-based rewards."""
        # Base reward: continuous similarity score (scaled by 1.5)
        similarity = self._compare(agent_structured, oracle_output)
        base_reward = similarity * 1.5  # 0.0-1.0 similarity → 0.0-1.5 reward
        
        # Compromise handling: continuous based on detection
        compromise_bonus = 0.0
        if hard_conflict:
            if compromise_detected:
                # Reward for detecting tradeoff (scaled by similarity)
                compromise_bonus = 0.75 * similarity  # Up to +0.75 bonus
            else:
                # Small penalty for missing tradeoff (not harsh -1.0)
                compromise_bonus = -0.3 * (1.0 - similarity)  # Soften based on how wrong
        
        return base_reward + compromise_bonus
    
    def _detect_tradeoffs(self, constraints: Dict[str, str]) -> Tuple[list, float]:
        """
        Feature: Explicit Tradeoff Reasoning
        
        Detects conflicts between constraints that require intentional reasoning.
        Returns: (list of conflicts, tradeoff_reasoning_score 0.0-1.0)
        
        Conflicts detected:
        - latency "real_time" + accuracy "near-perfect" → expensive, architectural compromise needed
        - data_size "very large" + budget "low" → infeasible, must choose scale/cost boundary  
        - update_frequency "continuous" + budget "low" → infrastructure cost vs capability tradeoff
        - latency "batch" + use_case "fraud detection" → real-time fraud needs near_real_time at minimum
        """
        conflicts = []
        tradeoff_score = 0.0  # 0 if no conflicts, scales with how many agent detected
        
        latency = constraints.get("latency", "").lower()
        accuracy = constraints.get("accuracy", "").lower()
        data_size = constraints.get("data_size", "").lower()
        budget = constraints.get("budget", "").lower()
        update_freq = constraints.get("update_frequency", "").lower()
        use_case = constraints.get("use_case", "").lower()
        
        # Conflict 1: latency vs accuracy (fundamental tradeoff)
        if latency == "real_time" and accuracy in ["high", "near-perfect"]:
            conflicts.append("latency_accuracy_tradeoff")
        if latency == "batch" and accuracy == "near-perfect" and use_case == "fraud detection":
            conflicts.append("fraud_detection_latency_mismatch")
        
        # Conflict 2: data_size vs budget (cost tradeoff)
        if data_size == "very large" and budget == "low":
            conflicts.append("scale_budget_tradeoff")
        if data_size in ["large", "very large"] and budget == "low":
            conflicts.append("data_storage_cost_tradeoff")
        
        # Conflict 3: update_frequency vs budget (infrastructure cost)
        if update_freq == "continuous" and budget == "low":
            conflicts.append("frequency_budget_tradeoff")
        if update_freq in ["streaming", "continuous"] and budget in ["low", "medium"]:
            conflicts.append("realtime_infrastructure_cost")
        
        # Conflict 4: latency + update_frequency (architectural pressure)
        if latency == "real_time" and update_freq == "batch":
            conflicts.append("latency_update_frequency_mismatch")
        
        # Score: each detected conflict → +0.2 points (up to 1.0)
        # Agent gets 0.2 per expected conflict, but this requires asking about tradeoffs
        # If agent collected <=4 constraints, they shouldn't have detected everything
        constraints_count = len(constraints)
        max_detectable_conflicts = min(len(conflicts), (constraints_count - 1) // 2)  # Usually need >=2 constraints per conflict
        tradeoff_score = float(len(conflicts)) / 5.0  # Normalize by ~5 possible conflicts
        
        return conflicts, min(tradeoff_score, 1.0)
    
    def _evaluate_constraint_conflicts(self, constraints: Dict[str, str]) -> float:
        """
        Reward function for constraint conflict awareness.
        
        Agents should:
        1. Collect diverse constraints (not just latency + accuracy)
        2. Explicitly reason about tradeoffs BEFORE finalizing
        3. Justify architecture choices given constraints
        
        Scoring:
        - No conflicts found: 0.0 (not a problem if truly no tradeoffs)
        - Conflicts found but not discussed: -0.15 (penalty for missing reasoning)
        - Conflicts found and reasoning shown: +0.25 (reward for awareness)
        
        This encourages agents to think beyond individual constraints.
        """
        conflicts, _score = self._detect_tradeoffs(constraints)
        
        if not conflicts:
            return 0.0  # No conflicts = nothing to penalize
        
        # Check if agent mentioned tradeoffs in recent messages
        recent_messages = self.state_data.get("messages", [])[-4:]  # Last 2 exchanges
        tradeoff_keywords = ["tradeoff", "trade-off", "compromise", "conflict", "tension", "tension", "balance", "versus", "vs."]
        
        mentioned_tradeoff = any(
            any(keyword in str(msg.get("content", "")).lower() for keyword in tradeoff_keywords)
            for msg in recent_messages
        )
        
        if mentioned_tradeoff:
            return 0.25  # Agent showed explicit reasoning about tradeoffs
        else:
            return -0.15  # Agent missed the opportunity for deeper reasoning
    
    def _compute_calibration_reward(self, belief_state: Dict[str, Dict[str, any]], true_constraints: Dict[str, str]) -> Tuple[float, Dict[str, any]]:
        """
        Feature: Calibrated Belief State Reward
        
        Rewards agents for:
        1. Having HIGH confidence only when correct (precision)
        2. Having LOW confidence when wrong (appropriate uncertainty)
        3. Updating beliefs based on new evidence
        
        Penalty for:
        1. Overconfidence: high confidence but wrong value
        2. Underconfidence: high confidence needed but not shown
        
        Returns: (calibration_reward, metrics_dict)
        """
        reward = 0.0
        metrics = {
            "avg_confidence": 0.0,
            "correct_high_confidence": 0,
            "incorrect_high_confidence": 0,
            "brier_score": 0.0,  # Standard calibration metric
        }
        
        total_brier_loss = 0.0
        for constraint_key, belief_entry in belief_state.items():
            belief_value = belief_entry.get("value")
            confidence = belief_entry.get("confidence", 0.0)
            true_value = true_constraints.get(constraint_key)
            
            if belief_value is None:
                # No observation yet
                metrics["avg_confidence"] += confidence
                continue
            
            is_correct = (belief_value == true_value)
            
            # Brier score: (confidence - correctness)^2
            # Lower is better (0 = perfectly calibrated)
            brier_loss = (confidence - float(is_correct)) ** 2
            total_brier_loss += brier_loss
            
            if is_correct and confidence >= 0.7:
                # Good: high confidence and correct
                metrics["correct_high_confidence"] += 1
                reward += 0.1  # Reward calibrated correctness
            elif is_correct and confidence < 0.7:
                # Suboptimal: correct but underconfident
                reward += 0.05  # Still correct, but missed confidence
            elif not is_correct and confidence >= 0.7:
                # Bad: overconfident and wrong
                metrics["incorrect_high_confidence"] += 1
                reward -= 0.2  # Penalize overconfidence
            elif not is_correct and confidence < 0.3:
                # Good: appropriately uncertain when wrong
                reward += 0.05
            else:
                # Moderate confidence when wrong (expected)
                reward -= 0.05
            
            metrics["avg_confidence"] += confidence
        
        # Normalize metrics
        n_constraints = len(belief_state)
        if n_constraints > 0:
            metrics["avg_confidence"] /= n_constraints
            metrics["brier_score"] = total_brier_loss / n_constraints
        
        # Scale reward by calibration quality
        # Perfect calibration (low Brier score) → reward bonus
        calibration_bonus = -0.5 * metrics["brier_score"]  # Negative loss → positive reward
        reward += calibration_bonus
        
        return reward * self.terminal_reward_weight, metrics
    
    def _compute_gae(self, episode_rewards: List[float], episode_baselines: List[float]) -> List[float]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Formula: A_t^GAE = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        Parameters:
        - gae_lambda: λ in [0, 1]
          - λ=0: Use only 1-step TD (biased but low variance)
          - λ=1: Use full monte-carlo return (unbiased but high variance)
          - λ=0.95: Standard choice (good bias-variance tradeoff)
        
        Returns:
        - advantages: GAE-computed advantages for each timestep
        """
        if not episode_rewards or not episode_baselines:
            return []
        
        advantages = []
        gae = 0.0
        
        # Compute backwards from end of episode
        for t in reversed(range(len(episode_rewards))):
            # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            if t < len(episode_rewards) - 1:
                next_value = episode_baselines[t + 1]
            else:
                next_value = 0.0  # Terminal state has V=0
            
            delta = episode_rewards[t] + self.gamma * next_value - episode_baselines[t]
            
            # GAE update: A_t = delta_t + (gamma * lambda) * A_{t+1}
            gae = delta + (self.gamma * self.gae_lambda) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def _compute_nstep_returns(self, episode_rewards: List[float], episode_baselines: List[float], n: int = 3) -> List[float]:
        """
        Compute n-step returns for bootstrapping.
        
        Formula: G_t^(n) = sum_{i=0}^{n-1} gamma^i * r_{t+i} + gamma^n * V(s_{t+n})
        
        This provides intermediate targets between 1-step TD and monte-carlo.
        - n=1: One-step TD return (low variance, high bias)
        - n=3: Three-step return (balanced)
        - n=∞: Full monte-carlo return (high variance, unbiased)
        
        Returns:
        - nstep_returns: n-step return targets for each timestep
        """
        if not episode_rewards or not episode_baselines:
            return []
        
        nstep_returns = []
        
        for t in range(len(episode_rewards)):
            nstep_return = 0.0
            
            # Sum n-step discounted rewards
            for i in range(n):
                if t + i < len(episode_rewards):
                    nstep_return += (self.gamma ** i) * episode_rewards[t + i]
            
            # Bootstrap from value function at step t+n
            if t + n < len(episode_baselines):
                bootstrap_value = episode_baselines[t + n]
            else:
                bootstrap_value = 0.0  # Terminal state
            
            nstep_return += (self.gamma ** n) * bootstrap_value
            nstep_returns.append(nstep_return)
        
        return nstep_returns
    
    def _compute_action_entropy(self, action_type: str) -> Tuple[float, Dict[str, float]]:
        """
        Track action distribution entropy to monitor exploration diversity.
        
        Entropy measures how varied the agent's actions are:
        - H(π) = -sum_a π(a) * log(π(a))
        - High entropy: Agent tries many different actions (exploring)
        - Low entropy: Agent repeats same actions (exploiting or stuck)
        
        Perfect uniform = log(num_actions) ≈ 1.95 bits
        Perfect deterministic = 0.0 bits
        
        Returns:
        - entropy: Shannon entropy of action distribution
        - entropy_info: Dict with action probabilities and insights
        """
        import math
        
        # Track this action
        self._action_history.append(action_type)
        self._action_counts[action_type] = self._action_counts.get(action_type, 0) + 1
        
        total_actions = len(self._action_history)
        if total_actions == 0:
            return 0.0, {"entropy": 0.0, "action_distribution": {}}
        
        # Compute empirical action distribution
        action_probs = {}
        for action, count in self._action_counts.items():
            action_probs[action] = count / total_actions
        
        # Compute Shannon entropy: H = -sum(p * log(p))
        entropy = 0.0
        for prob in action_probs.values():
            if prob > 0:
                entropy -= prob * math.log(prob)
        
        # Normalize to [0, 1] range (divided by max entropy)
        num_unique_actions = len(self._action_counts)
        max_entropy = math.log(num_unique_actions) if num_unique_actions > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Track entropy history for behavior detection
        self._episode_entropy_history.append(entropy)
        
        # Identify exploration patterns
        most_common = max(self._action_counts.items(), key=lambda x: x[1]) if self._action_counts else ("unknown", 0)
        most_common_pct = 100 * most_common[1] / total_actions
        
        entropy_info = {
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "num_unique_actions": int(num_unique_actions),
            "max_possible_entropy": float(max_entropy),
            "action_distribution": action_probs.copy(),
            "most_common_action": most_common[0],
            "most_common_pct": float(most_common_pct),
            "is_exploring": normalized_entropy > 0.5,  # High entropy = exploring
            "is_deterministic": normalized_entropy < 0.2,  # Low entropy = stuck
        }
        
        return entropy, entropy_info
    
    def _detect_entropy_behavior(self) -> Tuple[str, float]:
        """
        Detect behavior patterns based on entropy trajectory.
        
        Patterns:
        - 'overconfident': Low entropy early → agent commits to single action too quickly
        - 'confused': High entropy late → agent uncertain even near end
        - 'learning': Decreasing entropy → agent learning/converging
        - 'steady': Constant entropy → agent maintains consistent exploration
        - 'adapting': Variable entropy → agent adjusting exploration as needed
        
        Returns:
        - behavior_pattern: String classification
        - entropy_decay_rate: Rate of entropy change per step
        """
        if len(self._episode_entropy_history) < 2:
            return "unknown", 0.0
        
        entropy_vals = self._episode_entropy_history
        n_steps = len(entropy_vals)
        
        # Compute decay rate (linear regression slope)
        if n_steps >= 2:
            # Simple slope: (final - initial) / n_steps
            entropy_decay_rate = (entropy_vals[-1] - entropy_vals[0]) / max(n_steps - 1, 1)
        else:
            entropy_decay_rate = 0.0
        
        early_entropy = entropy_vals[0] if entropy_vals else 0.0
        late_entropy = entropy_vals[-1] if entropy_vals else 0.0
        mid_entropy = entropy_vals[n_steps // 2] if n_steps > 1 else early_entropy
        
        # Pattern detection logic
        if early_entropy < 0.3 and late_entropy < 0.3:
            pattern = "overconfident"  # Low from start to end
        elif late_entropy > 1.0 and early_entropy < late_entropy:
            pattern = "confused"  # High at end, increased from start
        elif entropy_decay_rate < -0.1:  # Decreasing by at least 0.1 per step
            pattern = "learning"  # Entropy declining = converging
        elif entropy_decay_rate > 0.1:  # Increasing
            pattern = "adapting"  # Entropy increasing = exploring more
        else:
            pattern = "steady"  # Roughly constant
        
        return pattern, float(entropy_decay_rate)
    
    def _compute_combined_reward(self,
                                  oracle_score: float,
                                  trajectory_score: float,
                                  process_reward: float) -> float:
        """
        Combine multiple reward signals with configurable weights.
        
        Formula:
        combined = oracle_weight * oracle_score +
                   trajectory_weight * trajectory_score +
                   process_weight * process_reward
        
        Where weights sum to approximately 1.0.
        
        Returns:
        - combined_reward: Weighted combination
        """
        # Normalize weights to sum to 1.0
        total_weight = self.oracle_weight + self.trajectory_weight + self.process_weight
        if total_weight == 0:
            total_weight = 1.0
        
        norm_oracle = self.oracle_weight / total_weight
        norm_trajectory = self.trajectory_weight / total_weight
        norm_process = self.process_weight / total_weight
        
        combined = (
            norm_oracle * oracle_score +
            norm_trajectory * trajectory_score +
            norm_process * process_reward
        )
        
        return float(combined)
    
    def _finalize_episode_learning_signals(self) -> Dict[str, Any]:
        """
        Compute episode-final learning signals: GAE, n-step returns, entropy summary.
        Called at episode end to prepare full trajectory for RL training.
        
        Returns:
        - episode_summary: Dict with all advanced signals for the episode
        """
        summary = {}
        
        # GAE computation on process rewards (per-step signals)
        if self._episode_step_rewards and self._episode_baselines:
            gae_advantages = self._compute_gae(self._episode_step_rewards, self._episode_baselines)
            summary["gae_advantages"] = gae_advantages
            summary["gae_lambda"] = self.gae_lambda
            summary["gae_reward_signal"] = "process"  # Indicates GAE is based on process rewards
        
        # Note on combined reward integration:
        # Combined rewards (oracle + trajectory + process) are computed at episode end.
        # For RL training, you can:
        # 1. Use gae_advantages as-is (process-based policy gradient)
        # 2. Reweight advantages by (combined_reward / avg_process) for signal combination
        # 3. Use combined_reward for value function bootstrap (replacing final baseline)
        
        # n-step returns (compute for n=1, 3, 5)
        if self._episode_step_rewards and self._episode_baselines:
            for n_steps in [1, 3, 5]:
                if n_steps <= len(self._episode_step_rewards):
                    nstep_returns = self._compute_nstep_returns(
                        self._episode_step_rewards, 
                        self._episode_baselines, 
                        n=n_steps
                    )
                    summary[f"nstep_{n_steps}_returns"] = nstep_returns
        
        # Action entropy summary with behavior detection
        if self._action_history:
            total_entropy, entropy_info = self._compute_action_entropy("")  # Empty string to skip incrementing
            summary["action_entropy"] = total_entropy
            summary["entropy_info"] = entropy_info
            summary["total_actions_taken"] = len(self._action_history)
            
            # Detect entropy-based behavior patterns
            behavior_pattern, entropy_decay = self._detect_entropy_behavior()
            summary["entropy_behavior_pattern"] = behavior_pattern
            summary["entropy_decay_rate"] = entropy_decay
            summary["entropy_history"] = self._episode_entropy_history.copy()
            
            # Add behavior interpretation
            behavior_info = {
                "overconfident": "Agent committed to single action too early (low entropy throughout)",
                "confused": "Agent remained uncertain even near episode end (high entropy late)",
                "learning": "Agent converged/learned by behaving more consistently (decreasing entropy)",
                "adapting": "Agent adjusted exploration strategy during episode (increasing entropy)",
                "steady": "Agent maintained consistent exploration level (stable entropy)",
                "unknown": "Insufficient data to classify behavior",
            }
            summary["behavior_interpretation"] = behavior_info.get(behavior_pattern, "")
        
        # Episode statistics
        if self._episode_step_rewards:
            summary["episode_total_reward"] = sum(self._episode_step_rewards)
            summary["episode_avg_reward"] = sum(self._episode_step_rewards) / len(self._episode_step_rewards)
            summary["episode_max_reward"] = max(self._episode_step_rewards)
            summary["episode_min_reward"] = min(self._episode_step_rewards)
            summary["episode_length"] = len(self._episode_step_rewards)
        
        return summary
    
    def _analyze_failure(self, agent_structured: Dict[str, str], oracle_output: Dict[str, object], 
                         similarity: float, coverage: float, observed_constraints: Dict[str, str],
                         hidden_constraints: Dict[str, str], final_reasoning: str, agent_output: str) -> Dict[str, Any]:
        """
        Detailed failure analysis when oracle_score < 1.0.
        Identifies root causes to help diagnose why agent's recommendation failed.
        
        Returns dict with:
        - failure_type: "coverage", "wrong_choice", "reasoning", "tradeoff", or "multiple"
        - component_mismatches: which of {model, deployment, architecture} are wrong
        - missing_constraints: which constraints weren't discovered
        - oracle_recommendation: the correct answer from oracle
        - agent_error: what the agent got wrong
        - analysis: human-readable diagnosis
        """
        analysis = {
            "failure_detected": similarity < 1.0,
            "similarity_score": float(similarity),
            "coverage_score": float(coverage),
        }
        
        # Identify which components are mismatched
        mismatches = []
        if agent_structured.get("model") != oracle_output.get("model"):
            mismatches.append("model")
        if agent_structured.get("deployment") != oracle_output.get("deployment"):
            mismatches.append("deployment")
        if agent_structured.get("architecture") != oracle_output.get("architecture"):
            mismatches.append("architecture")
        
        analysis["component_mismatches"] = mismatches
        analysis["oracle_model"] = str(oracle_output.get("model", "unknown"))
        analysis["oracle_deployment"] = str(oracle_output.get("deployment", "unknown"))
        analysis["oracle_architecture"] = str(oracle_output.get("architecture", "unknown"))
        analysis["agent_model"] = agent_structured.get("model", "unknown")
        analysis["agent_deployment"] = agent_structured.get("deployment", "unknown")
        analysis["agent_architecture"] = agent_structured.get("architecture", "unknown")
        
        # Identify missing constraints
        missing = []
        for constraint_key in hidden_constraints:
            if constraint_key not in observed_constraints:
                missing.append(constraint_key)
        
        analysis["missing_constraints"] = missing
        analysis["num_constraints_discovered"] = len(observed_constraints)
        analysis["num_constraints_total"] = len(hidden_constraints)
        
        # Determine failure type
        failure_types = []
        if coverage < 0.8:
            failure_types.append("coverage")
        if mismatches:
            failure_types.append("wrong_choice")
        if not any(word in final_reasoning.lower() for word in 
                   ["tradeoff", "balance", "compromise", "conflict", "tension"]):
            failure_types.append("missing_reasoning")
        
        analysis["failure_type"] = failure_types if failure_types else ["unknown"]
        
        # Generate human-readable analysis
        diagnosis_parts = []
        
        if coverage < 0.8:
            diagnosis_parts.append(
                f"Agent discovered {len(observed_constraints)}/{len(hidden_constraints)} constraints "
                f"(coverage: {coverage:.1%}). Missing: {', '.join(missing[:3])}"
            )
        
        if mismatches:
            diagnosis_parts.append(
                f"Component mismatch: {', '.join(mismatches)} incorrect. "
                f"Should be: {oracle_output.get('architecture', 'unknown')}"
            )
        
        if not any(word in final_reasoning.lower() for word in 
                   ["tradeoff", "balance", "compromise", "conflict", "tension"]):
            diagnosis_parts.append(
                "Agent did not reason about tradeoffs or conflicts between constraints"
            )
        
        if len(agent_output) < 50:
            diagnosis_parts.append("Agent reasoning was too brief to demonstrate understanding")
        
        analysis["diagnosis"] = " | ".join(diagnosis_parts) if diagnosis_parts else "Unknown failure type"
        
        return analysis
    
    def _compute_dense_rewards(self, 
                               information_gain: float,
                               exploration_reward: float,
                               refinement_reward: float,
                               belief_calibration_reward: float,
                               contradiction_handling_reward: float,
                               efficiency_reward: float,
                               consistency_reward: float,
                               counterfactual_reward: float,
                               step_penalties: float,
                               phase: str,
                               action_type: str) -> Tuple[Dict[str, float], float]:
        """
        Compute structured reward components for RL training.
        
        Returns:
        - components: Dict with each reward component tracked separately
        - total_step_reward: Sum of all components
        
        This enables:
        - Component-level analysis of which signals drive behavior
        - Dense reward shaping for credit assignment
        - Learning signal inspection during training
        """
        components = {
            "information_gain": float(information_gain),
            "exploration_reward": float(exploration_reward),
            "refinement_reward": float(refinement_reward),
            "belief_calibration_reward": float(belief_calibration_reward),
            "contradiction_handling_reward": float(contradiction_handling_reward),
            "efficiency_reward": float(efficiency_reward),
            "consistency_reward": float(consistency_reward),
            "counterfactual_reward": float(counterfactual_reward),
            "step_penalties": float(step_penalties),
        }
        
        # Total step-level reward (before terminal scaling)
        total_step_reward = (
            information_gain +
            exploration_reward +
            refinement_reward +
            belief_calibration_reward +
            contradiction_handling_reward +
            efficiency_reward +
            consistency_reward +
            counterfactual_reward +
            step_penalties
        )
        
        components["phase"] = phase
        components["action_type"] = action_type
        components["total_step_reward"] = float(total_step_reward)
        
        return components, float(total_step_reward)
    
    def _compute_advantage_signal(self, step_reward: float) -> Tuple[float, Dict[str, float]]:
        """
        Compute advantage signal A(s,a) = Q(s,a) - V(s) for RL training.
        
        Uses exponential moving average (EMA) baseline:
        - baseline = rolling average of all past step rewards
        - advantage = current_reward - baseline
        
        Returns:
        - advantage: A(s,a) value
        - signal_info: Dict with baseline, advantage, and normalization info
        
        Advantage signal helps RL algorithms:
        - Reduce variance in policy gradients
        - Identify which steps improved performance
        - Learn value function baseline
        """
        # Update rolling baseline with EMA
        # baseline_t = (1 - alpha) * baseline_{t-1} + alpha * reward_t
        self._rolling_baseline = (
            (1.0 - self._baseline_alpha) * self._rolling_baseline +
            self._baseline_alpha * step_reward
        )
        
        # Compute advantage
        advantage = step_reward - self._rolling_baseline
        
        # Normalize advantage to [-1, 1] approximately (for stability)
        # Use standard deviation of recent rewards if available
        recent_rewards = self._episode_step_rewards[-20:] if self._episode_step_rewards else [step_reward]
        if len(recent_rewards) > 1:
            reward_std = (sum((r - self._rolling_baseline) ** 2 for r in recent_rewards) / len(recent_rewards)) ** 0.5
            normalized_advantage = advantage / max(reward_std, 0.01)  # Avoid division by near-zero
        else:
            normalized_advantage = advantage / max(abs(self._rolling_baseline), 0.01)
        
        signal_info = {
            "baseline": float(self._rolling_baseline),
            "advantage_raw": float(advantage),
            "advantage_normalized": float(normalized_advantage),
            "step_reward": float(step_reward),
            "global_step": self._global_step_count,
        }
        
        return float(advantage), signal_info
    
    def _detect_and_reward_contradiction_handling(self, action_type: str, state_data: Dict[str, Any]) -> float:
        """
        Feature: Adversarial Contradiction Handling Reward
        
        In adversarial mode, reward agents for:
        1. Re-asking same constraint (skepticism, asking for clarification)
        2. Revising their recommendations when contradictions emerge
        3. Not blindly trusting conflicting info
        
        Reward structure:
        - +0.15 for re-asking a constraint (shows skepticism of adversarial responses)
        - +0.20 for reconsidering/revising recommendation after learning contradictory info  
        - -0.10 for NOT asking follow-ups when contradictions detected
        
        This makes adversarial mode a true test of decision robustness.
        """
        mode = str(state_data.get("mode", "clean")).lower()
        if mode != "adversarial":
            return 0.0  # Only applies in adversarial mode
        
        messages = state_data.get("messages", [])
        observed_constraints = state_data.get("observed_constraints", {})
        
        # Check for re-asking (skepticism signal)
        re_ask_bonus = 0.0
        if len(messages) >= 4:
            # Look for repeated questions about same topic
            recent_actions = [m.get("role") for m in messages[-4:] if isinstance(m, dict)]
            if action_type in ["ASK_LATENCY", "ASK_ACCURACY", "ASK_DATA_SIZE", "ASK_UPDATE_FREQUENCY", "ASK_BUDGET"]:
                # Count how many times AGENTasked about this constraint
                constraint_asks = [m for m in messages if isinstance(m, dict) and "ASK_" in str(m.get("content", ""))]
                if len(constraint_asks) > 1:  # Re-asking detected
                    re_ask_bonus = 0.15
        
        # Check for contradictory observations (conflicting values for same key)
        contradiction_bonus = 0.0
        contradictions_found = {}
        for key in observed_constraints:
            # Check belief state to see if confidence dropped (contradiction signal)
            belief_state = state_data.get("belief_state", {})
            if key in belief_state:
                belief_info = belief_state.get(key, {})
                confidence = belief_info.get("confidence", 0.0)
                # If confidence is in medium range (0.4-0.7), likely seen contradictory info
                if 0.3 < confidence < 0.7:
                    contradictions_found[key] = True
        
        # Reward for handling contradictions appropriately
        if contradictions_found:
            # Check if agent asked follow-up questions about these keys
            follow_up_count = 0
            for key in contradictions_found:
                key_to_action = {
                    "latency": "ASK_LATENCY",
                    "accuracy": "ASK_ACCURACY",
                    "data_size": "ASK_DATA_SIZE",
                    "update_frequency": "ASK_UPDATE_FREQUENCY",
                    "budget": "ASK_BUDGET",
                }
                action_to_check = key_to_action.get(key)
                if action_to_check:
                    # Count asks about this constraint in recent messages
                    asks_count = sum(1 for m in messages[-6:] if isinstance(m, dict) and action_to_check in str(m.get("content", "")))
                    if asks_count >= 2:  # Multi-ask shows investigation
                        follow_up_count += 1
            
            if follow_up_count > 0:
                contradiction_bonus = 0.20 * (follow_up_count / len(contradictions_found))
            else:
                # Penalty for not investigating contradictions
                contradiction_bonus = -0.10
        
        return re_ask_bonus + contradiction_bonus
    
    def _evaluate_belief_calibration(self, belief_state: Dict[str, Dict[str, any]]) -> float:
        """
        Feature: Belief Calibration Reward
        
        Reward agents for maintaining well-calibrated confidence in their beliefs.
        
        Scoring:
        - High confidence (0.7-1.0): +0.05 per constraint (justified when observations agree)
        - Medium confidence (0.4-0.7): +0.02 per constraint (cautious, good for noisy environments)
        - Low confidence (<0.4): -0.05 if no conflicts (agent too uncertain)
        - Overconfidence detected: -0.10 per conflicting observation
        
        Returns:
        - Expected value: 0.0-0.3 range
        - Calibration reward = (correct_confidence_level)
        """
        if not belief_state:
            return 0.0
        
        reward = 0.0
        constraint_count = 0
        
        for key, state in belief_state.items():
            if state.get("value") is None:
                continue  # No observation yet
            
            constraint_count += 1
            confidence = state.get("confidence", 0.0)
            
            # Reward appropriate confidence levels
            if confidence >= 0.7:
                # High confidence: good if observations are consistent
                reward += 0.05 * confidence  # Higher confidence = more reward (up to 0.05)
            elif confidence >= 0.4:
                # Medium confidence: appropriate for uncertain scenarios
                reward += 0.02 * confidence  # Modest reward
            
            # Penalize underconfidence (below 0.3 when you have observations)
            if confidence < 0.3 and state.get("value") is not None:
                reward -= 0.03  # Agent too uncertain despite observations
        
        # Reward is per-constraint, normalize
        if constraint_count > 0:
            calibration_reward = reward / max(constraint_count, 1)
        else:
            calibration_reward = 0.0
        
        return min(max(calibration_reward, -0.2), 0.15)  # Clamp to reasonable range
    
    def _is_action_gated_in_phase(self, action_type: str, phase: str) -> bool:
        """
        Feature 5: Phase transitions control dynamics.
        Certain actions are only allowed in specific phases.
        """
        if phase == "exploration":
            # Can ask anything in exploration
            return False
        elif phase == "refinement":
            # In refinement, can't ask basic USE_CASE questions
            # Force deeper exploration of implications
            if action_type == "ASK_USE_CASE":
                return True  # Gated (not allowed)
            return False
        elif phase == "decision":
            # In decision phase, can only finalize, not ask more
            if action_type.startswith("ASK_"):
                return True  # Gated (not allowed)
            return False
        return False
    
    def _evaluate_question_efficiency(self, action_type: str, messages: List[Dict[str, str]],
                                     belief_state: Dict[str, Dict[str, Any]],
                                     hidden_constraints: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Feature: Process Quality Rewards
        
        Reward agents for efficient, logically-sequenced question paths.
        Penalize wasted effort (repeats, irrelevance) and reward clarity achievement.
        
        Metrics:
        1. Repeated Questions: -0.10 per duplicate ask
        2. Uncertainty Resolution: +0.20 when confidence transitions low→high
        3. Question Relevance: -0.05 for questions unrelated to hidden_constraints
        4. Logical Sequence: +0.15 for building-block pattern (constraint A informs B)
        
        Returns: (efficiency_reward, metrics_dict)
        """
        metrics = {
            "repeated_questions": 0,
            "uncertainty_resolved": 0,
            "irrelevant_questions": 0,
            "logical_sequences": 0,
            "efficiency_reward": 0.0,
        }
        
        if not isinstance(action_type, str) or not action_type.startswith("ASK_"):
            # FINALIZE or invalid action
            return 0.0, metrics
        
        # Extract constraint key from action (e.g., "ASK_LATENCY" -> "latency")
        constraint_key = action_type.replace("ASK_", "").lower()
        
        # ===== 1. DETECT REPEATED QUESTIONS =====
        # Count how many times this constraint has been asked
        question_history = self.state_data.get("question_history", {})
        ask_count = question_history.get(action_type, 0)
        
        if ask_count > 0:
            # This is a repeat ask
            metrics["repeated_questions"] = ask_count
            repeat_penalty = -0.10 * min(ask_count, 2)  # Cap penalty at 2 repeats
        else:
            repeat_penalty = 0.0
        
        # Update question history for next step
        question_history[action_type] = ask_count + 1
        
        # ===== 2. DETECT UNCERTAINTY RESOLUTION =====
        previous_uncertainty = self.state_data.get("previous_uncertainty", {})
        prev_confidence = previous_uncertainty.get(constraint_key, 0.0)
        curr_confidence = belief_state.get(constraint_key, {}).get("confidence", 0.0)
        
        uncertainty_bonus = 0.0
        if prev_confidence < 0.5 and curr_confidence >= 0.7:
            # Resolved low→high confidence (agent reduced uncertainty significantly)
            uncertainty_bonus = 0.20
            metrics["uncertainty_resolved"] += 1
        elif prev_confidence < 0.3 and curr_confidence >= 0.5:
            # Partial resolution (low→medium)
            uncertainty_bonus = 0.10
            metrics["uncertainty_resolved"] += 1
        
        # ===== 3. CHECK QUESTION RELEVANCE =====
        relevance_penalty = 0.0
        if constraint_key not in hidden_constraints:
            # Asking about constraint NOT in hidden_constraints
            relevance_penalty = -0.05
            metrics["irrelevant_questions"] += 1
        
        # ===== 4. DETECT LOGICAL SEQUENCES =====
        # Track constraint discovery order
        constraint_discovery_order = self.state_data.get("constraint_discovery_order", [])
        
        logical_bonus = 0.0
        if constraint_key in hidden_constraints:
            # Build dependency logic: some constraints inform others
            dependencies = {
                "data_size": ["accuracy", "latency"],  # Data size affects accuracy and latency
                "latency": ["update_frequency", "accuracy"],  # Latency constrains update frequency
                "accuracy": ["budget"],  # Accuracy requirements drive budget
                "budget": ["latency", "data_size"],  # Budget constrains resources
            }
            
            # Check if this constraint builds on previously asked constraints
            if constraint_key in dependencies:
                prior_constraints = constraint_discovery_order
                required_prior = dependencies[constraint_key]
                
                # Count how many dependencies were already asked
                dependencies_asked = sum(1 for dep in required_prior if dep in prior_constraints)
                
                if dependencies_asked >= 1:
                    # Logical progression: asking about B after asking about its dependency A
                    logical_bonus = 0.15 * (dependencies_asked / max(len(required_prior), 1))
                    metrics["logical_sequences"] += 1
        
        # Add to discovery order
        if constraint_key not in constraint_discovery_order and constraint_key in hidden_constraints:
            constraint_discovery_order.append(constraint_key)
        
        # ===== COMPUTE FINAL EFFICIENCY REWARD =====
        efficiency_reward = repeat_penalty + uncertainty_bonus + relevance_penalty + logical_bonus
        
        # Clamp to reasonable range [-0.25, +0.35]
        efficiency_reward = min(max(efficiency_reward, -0.25), 0.35)
        
        metrics["efficiency_reward"] = efficiency_reward
        
        return efficiency_reward, metrics
    
    def _evaluate_consistency(self, observed_constraints: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Feature: Consistency Score (Stable Reasoning)
        
        Measures whether agent maintains stable beliefs about constraints across the episode.
        Rewards agents for consistent reasoning; penalizes flip-flopping.
        
        Logic:
        1. Track all values each constraint has taken (constraint_value_history)
        2. When a constraint value changes, increment flip_flop_count
        3. Return consistency_score = max(0, 1 - flip_flop_count * 0.2)
        
        Returns: (consistency_reward, metrics_dict)
        
        Reward Range: [-0.35, +0.25]
        - Max stability: No changes, reward +0.25 per constraint observed
        - Perfect consistency: All constraints stable, +0.25 total
        - Flip-flop: Each change -0.10, can go negative
        """
        metrics = {
            "consistency_violations": 0,
            "stable_constraints": 0,
            "total_observations": len(observed_constraints),
            "consistency_score": 1.0,
        }
        
        # Initialize constraint_value_history if not exists
        constraint_value_history = self.state_data.get("constraint_value_history", {})
        
        violations = 0
        
        # Check each observed constraint for value changes
        for constraint_key, constraint_value in observed_constraints.items():
            if constraint_key not in constraint_value_history:
                # First time seeing this constraint - add to history
                constraint_value_history[constraint_key] = [constraint_value]
            else:
                # Constraint seen before - check if value changed
                previous_value = constraint_value_history[constraint_key][-1]
                
                if previous_value != constraint_value:
                    # Value changed - penalty for flip-flop
                    violations += 1
                    # Record the new value
                    constraint_value_history[constraint_key].append(constraint_value)
                else:
                    # Value remained the same - reward consistency
                    metrics["stable_constraints"] += 1
        
        # Calculate consistency score: 1.0 - (violations * 0.2)
        # Each flip-flop costs 0.2 points, clamped to [0.0, 1.0]
        consistency_score = max(0.0, 1.0 - (violations * 0.2))
        metrics["consistency_score"] = consistency_score
        metrics["consistency_violations"] = violations
        
        # Update flip_flop_count in state_data
        self.state_data["constraint_value_history"] = constraint_value_history
        self.state_data["flip_flop_count"] = int(self.state_data.get("flip_flop_count", 0)) + violations
        
        # Reward/penalty calculation
        # Stable constraints: +0.05 per stable constraint (up to +0.30 for 6 constraints)
        stability_bonus = 0.05 * metrics["stable_constraints"]
        
        # Flip-flop penalty: -0.15 per violation (scales with how many times values change)
        flipflop_penalty = -0.15 * violations
        
        # Combine rewards: bonus for stability, penalty for changes
        consistency_reward = stability_bonus + flipflop_penalty
        
        # Clamp to reasonable range [-0.35, +0.25]
        consistency_reward = min(max(consistency_reward, -0.35), 0.25)
        
        metrics["consistency_reward"] = consistency_reward
        
        return consistency_reward, metrics
    
    def _evaluate_global_efficiency(self, steps_taken: int, task_id: str) -> Tuple[float, Dict[str, Any]]:
        """
        Feature: Global Efficiency Score (Terminal-level)
        
        Evaluates overall episode efficiency based on steps taken.
        Prevents "slow but safe" agents from gaming the system.
        
        Logic:
        - optimal_steps (min_steps): task-specific minimum efficient steps
        - If steps_taken <= optimal_steps: score = 1.0 (perfect efficiency)
        - Otherwise: score = max(0.0, 1 - (steps_taken - optimal_steps) * 0.1)
        
        Returns: (efficiency_score, metrics_dict)
        
        Example:
        - Task: easy (optimal=6 steps)
        - Agent takes 6 steps → score = 1.0, reward = +0.35
        - Agent takes 8 steps → score = 0.8, reward = +0.28
        - Agent takes 15 steps → score = 0.1, reward = +0.035
        - Agent takes 20+ steps → score approaches 0.0, reward → 0.0
        """
        # Get optimal steps for this task
        optimal_steps = self.optimal_steps.get(task_id, 9)
        
        metrics = {
            "steps_taken": steps_taken,
            "optimal_steps": optimal_steps,
            "efficiency_score": 1.0,
        }
        
        # Calculate efficiency score
        if steps_taken <= optimal_steps:
            # Perfect or near-perfect efficiency
            efficiency_score = 1.0
        else:
            # Penalty: 0.1 per step beyond optimal
            excess_steps = steps_taken - optimal_steps
            efficiency_score = max(0.0, 1.0 - (excess_steps * 0.1))
        
        metrics["efficiency_score"] = efficiency_score
        
        # Convert efficiency score to reward bonus
        # Maximum bonus: +0.35 (when efficiency_score = 1.0)
        # Scales linearly with efficiency
        global_efficiency_reward = 0.35 * efficiency_score
        
        metrics["global_efficiency_reward"] = global_efficiency_reward
        
        return global_efficiency_reward, metrics
    
    def _evaluate_recovery(self, state_data: Dict[str, object], oracle_similarity: float) -> Tuple[float, Dict[str, Any]]:
        """
        Feature: Smooth Recovery Score (Terminal-level)
        
        Measures agent's ability to recover from adversarial/noisy information.
        Uses continuous recovery metric: recovery_score = 1 - (mistakes - recovered) / max(1, mistakes)
        
        This prevents harsh penalties and noisy scoring by measuring partial recovery.
        
        Logic:
        - mistakes = flip_flop_count (times constraint values changed)
        - recovered = mistakes * oracle_similarity (estimated recovery via final accuracy)
        - recovery_score = fraction of mistakes successfully recovered from
        
        Returns: (recovery_score, metrics_dict)
        
        Recovery Score Range: [0.0, 1.0]
        - No mistakes (flip_flops=0): 1.0 (perfect)
        - All mistakes recovered (oracle_similarity=1.0): 1.0
        - Half mistakes recovered: 0.5 (partial)
        - No recovery (oracle_similarity=0.0): 0.0
        """
        mode = str(state_data.get("mode", "clean"))
        flip_flop_count = int(state_data.get("flip_flop_count", 0))
        oracle_similarity = min(max(oracle_similarity, 0.0), 1.0)  # Clamp to [0.0, 1.0]
        
        metrics = {
            "mode": mode,
            "flip_flop_count": flip_flop_count,
            "oracle_similarity": oracle_similarity,
        }
        
        # Clean mode: no recovery challenge needed
        if mode == "clean":
            recovery_score = 1.0
            metrics["recovery_score"] = recovery_score
            return recovery_score, metrics
        
        # === SMOOTH RECOVERY FORMULA ===
        # recovery_score = 1 - (mistakes - recovered) / max(1, mistakes)
        # Where: mistakes = flip_flop_count
        #        recovered = flip_flop_count * oracle_similarity
        # This gives a continuous metric from 0.0 to 1.0 measuring recovery %
        
        mistakes = flip_flop_count
        
        if mistakes == 0:
            # No mistakes encountered = perfect recovery
            recovery_score = 1.0
        else:
            # Agent made mistakes. Recovery depends on oracle_similarity
            # If oracle_similarity = 1.0: fully recovered from all mistakes (recovery_score = 1.0)
            # If oracle_similarity = 0.5: recovered from 50% (recovery_score = 0.5)
            # If oracle_similarity = 0.0: recovered from none (recovery_score = 0.0)
            recovered = mistakes * oracle_similarity
            unrecovered = mistakes - recovered
            recovery_score = 1.0 - (unrecovered / max(1, mistakes))
            # This simplifies to: recovery_score = oracle_similarity (when mistakes > 0)
            recovery_score = min(1.0, max(0.0, recovery_score))
        
        metrics["mistakes"] = mistakes
        metrics["recovered"] = float(mistakes * oracle_similarity if mistakes > 0 else 0)
        metrics["recovery_score"] = recovery_score
        
        return recovery_score, metrics
    
    def _evaluate_justification(self, info: Dict[str, Any], observed_constraints: Dict[str, str], 
                               detected_tradeoffs: List[str]) -> Tuple[float, Dict[str, Any]]:
        """
        Feature: Decision Justification Score (Terminal-level)
        
        Rewards agents for providing explicit reasoning about their final decision.
        Encourages transparency and tradeoff awareness.
        
        Logic:
        1. Check if agent provided final reasoning (implicit via recommendation quality)
        2. Score coverage: how many constraints mentioned in reasoning
        3. Score tradeoff awareness: how many detected conflicts addressed in reasoning
        4. Composite: min(1.0, 0.5 * coverage + 0.5 * conflicts)
        
        Returns: (justification_score, metrics_dict)
        
        Justification Score: [0.0-1.0]
        - 1.0: Full coverage + all tradeoffs addressed
        - 0.5: Partial coverage + some tradeoff awareness
        - 0.0: No reasoning or poor justification
        """
        metrics = {
            "constraints_mentioned": 0,
            "tradeoffs_mentioned": 0,
            "coverage_score": 0.0,
            "tradeoff_awareness_score": 0.0,
            "justification_score": 0.0,
        }
        
        # Get final reasoning/recommendation from state or info
        final_reasoning = str(self.state_data.get("final_reasoning", ""))
        final_recommendation = str(self.state_data.get("final_recommendation", ""))
        
        # Combine both for analysis
        full_text = (final_reasoning + " " + final_recommendation).lower()
        
        # If no reasoning text captured, score is 0
        if not full_text or len(full_text.strip()) < 5:
            return 0.0, metrics
        
        # ===== COVERAGE SCORE =====
        # How many observed constraints are mentioned in the reasoning?
        constraints_mentioned = 0
        constraint_names = {
            "use_case": ["use case", "usecase", "use-case"],
            "latency": ["latency", "latency requirement", "latency constraint"],
            "accuracy": ["accuracy", "accuracy requirement", "precision"],
            "data_size": ["data size", "data_size", "data volume", "scale"],
            "update_frequency": ["update frequency", "update_frequency", "frequency"],
            "budget": ["budget", "cost", "constraint on cost"],
        }
        
        for constraint_key, aliases in constraint_names.items():
            if constraint_key in observed_constraints:
                # Check if any alias appears in reasoning
                if any(alias in full_text for alias in aliases):
                    constraints_mentioned += 1
        
        metrics["constraints_mentioned"] = constraints_mentioned
        
        # Coverage score: percentage of observed constraints mentioned
        if len(observed_constraints) > 0:
            coverage_score = constraints_mentioned / len(observed_constraints)
        else:
            coverage_score = 0.0
        
        metrics["coverage_score"] = coverage_score
        
        # ===== TRADEOFF AWARENESS SCORE =====
        # How many detected tradeoffs are mentioned/addressed in reasoning?
        tradeoffs_mentioned = 0
        tradeoff_keywords = {
            "latency_accuracy": ["tradeoff", "trade-off", "latency vs accuracy", "accuracy latency"],
            "scale_budget": ["scale budget", "budget constraint", "resource constraint"],
            "frequency_budget": ["frequency budget", "update cost", "frequency constraint"],
            "latency_usecase": ["latency usecase", "requirements latency"],
            "compromise": ["compromise", "balance", "optimal", "mitigation"],
        }
        
        for tradeoff_key, keywords in tradeoff_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                tradeoffs_mentioned += 1
        
        metrics["tradeoffs_mentioned"] = tradeoffs_mentioned
        
        # Tradeoff awareness score: based on how many tradeoffs were detected and mentioned
        if len(detected_tradeoffs) > 0:
            # If tradeoffs were detected, reward agent for addressing them
            tradeoff_awareness_score = min(1.0, tradeoffs_mentioned / max(len(detected_tradeoffs), 1))
        else:
            # No tradeoffs detected - still reward if agent mentions balanced approach
            if "compromise" in full_text or "balance" in full_text or "mitigation" in full_text:
                tradeoff_awareness_score = 0.5
            else:
                tradeoff_awareness_score = 0.0
        
        metrics["tradeoff_awareness_score"] = tradeoff_awareness_score
        
        # ===== COMPOSITE JUSTIFICATION SCORE =====
        # Weighted combination: 0.5 * coverage + 0.5 * tradeoff_awareness
        justification_score = min(1.0, 0.5 * coverage_score + 0.5 * tradeoff_awareness_score)
        
        metrics["justification_score"] = justification_score
        
        return justification_score, metrics
    
    def _evaluate_delta_information_gain(self, state_data: Dict[str, object]) -> Tuple[float, Dict[str, Any]]:
        """
        Feature: Delta Information Gain (Step-wise Reasoning Quality)
        
        Measures cumulative uncertainty reduction across the trajectory.
        - High when: early questions reduce uncertainty, later questions are confirmations
        - Low when: late questions add new uncertainty, random ordering
        
        Returns: (information_gain_score, metrics_dict)
        
        Score Range: [0.0, 1.0]
        - 1.0: Perfect information ordering (reduce uncertainty monotonically)
        - 0.5: Moderate ordering (some backtracking)
        - 0.0: Poor ordering (increasing uncertainty or noise)
        """
        constraint_discovery_order = state_data.get("constraint_discovery_order", [])
        total_constraints = 6  # use_case, latency, accuracy, data_size, update_frequency, budget
        
        metrics = {
            "constraint_discovery_order": len(constraint_discovery_order),
            "information_gain_score": 1.0,
        }
        
        # If no constraints discovered, no information gain
        if len(constraint_discovery_order) == 0:
            metrics["information_gain_score"] = 1.0  # No gain, but also no penalty
            return 1.0, metrics
        
        # Information gain: measure early vs late discovery
        # Better: discover constraints early, confirm later
        # Worse: discover constraints late, then have to backtrack
        
        early_discoveries = min(3, len(constraint_discovery_order))  # First 3 constraints
        late_discoveries = max(0, len(constraint_discovery_order) - 3)
        
        # Scoring:
        # - Early discoveries contribute more (reduce uncertainty earlier)
        # - Late discoveries contribute less (less time to use the info)
        if len(constraint_discovery_order) <= 3:
            # All discovered early = good ordering
            information_gain_score = 1.0
        else:
            # Mixed: some early, some late
            # Penalize late discoveries: each late discovery reduces score
            late_penalty = (late_discoveries / float(len(constraint_discovery_order))) * 0.4
            information_gain_score = max(0.0, 1.0 - late_penalty)
        
        metrics["information_gain_score"] = float(information_gain_score)
        metrics["early_discoveries"] = early_discoveries
        metrics["late_discoveries"] = late_discoveries
        
        return information_gain_score, metrics
    
    def _evaluate_constraint_utilization(self, observed_constraints: Dict[str, str], 
                                         final_recommendation: str) -> Tuple[float, Dict[str, Any]]:
        """
        Feature: Constraint Utilization Score (Decision Quality)
        
        Measures if agent uses ALL discovered constraints in final decision.
        - High when: final recommendation mentions/uses all observed constraints
        - Low when: agent ignores some constraints that were discovered
        
        Returns: (utilization_score, metrics_dict)
        
        Score Range: [0.0, 1.0]
        - 1.0: Used all constraints in decision
        - 0.5: Used 50% of discovered constraints
        - 0.0: Ignored all constraints (degenerate case)
        """
        if not final_recommendation or len(final_recommendation.strip()) < 5:
            return 0.0, {"utilization_score": 0.0, "constraints_used": 0}
        
        # Constraint aliases for matching in recommendation text
        constraint_aliases = {
            "use_case": ["use case", "usecase", "use-case", "application", "scenario"],
            "latency": ["latency", "low-latency", "real-time", "response time"],
            "accuracy": ["accuracy", "precision", "correct", "error rate"],
            "data_size": ["data size", "data_size", "scale", "volume", "large dataset"],
            "update_frequency": ["update frequency", "update_frequency", "frequently", "real-time"],
            "budget": ["budget", "cost", "expensive", "affordable"],
        }
        
        full_text = final_recommendation.lower()
        constraints_used = 0
        
        for constraint_key, aliases in constraint_aliases.items():
            if constraint_key in observed_constraints:
                # Check if any alias appears in recommendation
                if any(alias in full_text for alias in aliases):
                    constraints_used += 1
        
        total_observed = len(observed_constraints)
        
        if total_observed == 0:
            utilization_score = 1.0  # Nothing to use
        else:
            utilization_score = min(1.0, constraints_used / float(total_observed))
        
        metrics = {
            "utilization_score": float(utilization_score),
            "constraints_used": constraints_used,
            "constraints_observed": total_observed,
        }
        
        return utilization_score, metrics
    
    def _evaluate_redundancy_score(self, state_data: Dict[str, object]) -> Tuple[float, Dict[str, Any]]:
        """
        Feature: Redundancy Score (Trajectory Efficiency)
        
        Measures fraction of questions that are repetitions.
        - High when: few repeated questions = efficient trajectory
        - Low when: many redundant questions = inefficient trajectory
        
        Formula: redundancy_ratio = repeated_questions / total_questions
        Score: 1.0 - redundancy_ratio (inverted: higher is better)
        
        Returns: (redundancy_score, metrics_dict)
        
        Score Range: [0.0, 1.0]
        - 1.0: No repeated questions (perfect efficiency)
        - 0.5: 50% of questions were repeats
        - 0.0: All questions were repeats (degenerate)
        """
        question_history = state_data.get("question_history", {})
        
        metrics = {
            "redundancy_score": 1.0,
        }
        
        # Count total questions and repeated questions
        total_questions = sum(question_history.values())
        
        if total_questions == 0:
            metrics["redundancy_score"] = 1.0
            return 1.0, metrics
        
        # Repeated questions: count questions asked more than once
        repeated_questions = 0
        for action_type, count in question_history.items():
            if count > 1:
                # All repetitions beyond the first are wasted
                repeated_questions += (count - 1)
        
        redundancy_ratio = repeated_questions / float(total_questions)
        redundancy_score = 1.0 - redundancy_ratio
        redundancy_score = max(0.0, min(1.0, redundancy_score))
        
        metrics["redundancy_score"] = float(redundancy_score)
        metrics["total_questions"] = total_questions
        metrics["repeated_questions"] = repeated_questions
        
        return redundancy_score, metrics
    
    def _apply_observation_noise(self, observed: Dict[str, str]) -> Tuple[Dict[str, str], list]:
        """
        Feature 6: Stochastic observation noise.
        Noise probability depends on mode:
        - clean: 0% (deterministic)
        - noisy: 15% per constraint
        - adversarial: 25% per constraint
        
        Creates variance, realistic difficulty, smoother curves.
        
        Returns: (noisy_observed, list of corrupted constraint keys)
        """
        noisy_observed = dict(observed)
        noisy_constraints = []
        
        # Mode-dependent noise probability
        mode = str(self.state_data.get("mode", "clean"))
        if mode == "clean":
            noise_probability = 0.0  # Deterministic for testing
        elif mode == "noisy":
            noise_probability = 0.15  # Mild corruption
        elif mode == "adversarial":
            noise_probability = 0.25  # More aggressive
        else:
            noise_probability = 0.2  # Default
        
        # Skip noise application if probability is 0
        if noise_probability == 0:
            return noisy_observed, noisy_constraints
        
        # Possible realistic incorrect values per constraint type
        wrong_values = {
            "use_case": ["recommendation ranking", "fraud detection", "multimodal assistant"],
            "latency": ["batch", "near_real_time", "real_time"],
            "accuracy": ["low", "medium", "high", "near-perfect"],
            "data_size": ["small", "moderate", "large", "very large"],
            "update_frequency": ["daily", "hourly", "streaming", "continuous"],
            "budget": ["low", "medium", "high"],
        }
        
        # Apply noise with mode-dependent probability per observation
        for key, value in observed.items():
            if random.random() < noise_probability:
                # Corrupt this constraint with a wrong value
                if key in wrong_values:
                    # Pick a different wrong value
                    alternatives = [v for v in wrong_values[key] if v != value]
                    if alternatives:
                        noisy_observed[key] = random.choice(alternatives)
                        noisy_constraints.append(key)
        
        return noisy_observed, noisy_constraints
    
        """
        Feature 6: Stochastic observation noise.
        Noise probability depends on mode:
        - clean: 0% (deterministic)
        - noisy: 15% per constraint
        - adversarial: 25% per constraint
        
        Creates variance, realistic difficulty, smoother curves.
        
        Returns: (noisy_observed, list of corrupted constraint keys)
        """
        noisy_observed = dict(observed)
        noisy_constraints = []
        
        # Mode-dependent noise probability
        mode = str(self.state_data.get("mode", "clean"))
        if mode == "clean":
            noise_probability = 0.0  # Deterministic for testing
        elif mode == "noisy":
            noise_probability = 0.15  # Mild corruption
        elif mode == "adversarial":
            noise_probability = 0.25  # More aggressive
        else:
            noise_probability = 0.2  # Default
        
        # Skip noise application if probability is 0
        if noise_probability == 0:
            return noisy_observed, noisy_constraints
        
        # Possible realistic incorrect values per constraint type
        wrong_values = {
            "use_case": ["recommendation ranking", "fraud detection", "multimodal assistant"],
            "latency": ["batch", "near_real_time", "real_time"],
            "accuracy": ["low", "medium", "high", "near-perfect"],
            "data_size": ["small", "moderate", "large", "very large"],
            "update_frequency": ["daily", "hourly", "streaming", "continuous"],
            "budget": ["low", "medium", "high"],
        }
        
        # Apply noise with mode-dependent probability per observation
        for key, value in observed.items():
            if random.random() < noise_probability:
                # Corrupt this constraint with a wrong value
                if key in wrong_values:
                    # Pick a different wrong value
                    alternatives = [v for v in wrong_values[key] if v != value]
                    if alternatives:
                        noisy_observed[key] = random.choice(alternatives)
                        noisy_constraints.append(key)
        
        return noisy_observed, noisy_constraints
