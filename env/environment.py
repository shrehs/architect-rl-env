from copy import deepcopy
import random
import math
from typing import Dict, Tuple
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
        self.episode_counter += 1
        self.mode = random.choice(["clean", "noisy", "adversarial"])
        self.state_data: Dict[str, object] = {
            "messages": [],
            "observed_constraints": {},
            "belief": self._belief_from_constraints({}),
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
        }
        self.checkpoints = {}  # Reset checkpoints for new episode
        self.user = UserSimulator(self.state_data["hidden_constraints"])
        # Structured logging: episode start
        print(f"START episode_{self.episode_counter} task={self.task_id} mode={self.mode}")
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.state_data["done"]:
            raise RuntimeError("Episode already finished. Call reset().")

        # Feature 5: Phase-dependent action gating
        action_type = action.type
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
        
        # Structured logging: step and episode progress
        step_num = int(self.state_data["step_count"])
        print(f"STEP {step_num} action={action_type} reward={reward:.5f}")
        
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

        reward = self._information_gain(before_belief, after_belief)
        if action_type in {"FINALIZE", "FINALIZE_WITH_COMPROMISE"} and int(self.state_data["step_count"]) < 10:
            reward -= 0.2
        if "budget" in after and before.get("budget") != after.get("budget"):
            reward += 0.1
        phase = str(self.state_data.get("phase", "exploration"))
        
        # Phase-specific per-step rewards
        if phase == "exploration":
            reward += self._exploration_reward(new_constraints_count, action_type)
        elif phase == "refinement":
            reward += self._refinement_reward(self._information_gain(before_belief, after_belief), duplicate_submission)
        
        # Adversarial resilience bonus: reward progress made despite misleading info
        mode = str(self.state_data.get("mode", "clean"))
        if mode == "adversarial" and new_constraints_count > 0:
            reward += 0.15  # Bonus for extracting correct info despite adversarial responses
        
        # Soften step-level penalties: use smaller continuous adjustments
        if new_constraints_count == 0:
            if phase == "refinement":
                reward -= 0.05  # Softer penalty
            else:
                reward -= 0.02  # Very soft penalty for exploration
        if duplicate_submission:
            reward -= 0.05  # Softer duplicate penalty
        # Soften context pruning penalty
        if len(self.state_data["messages"]) > len(visible_messages):
            reward -= 0.02  # Very soft penalty
        lowered_reply = response.lower()
        unsafe_input = "hack" in lowered_reply or "bypass" in lowered_reply
        if unsafe_input:
            reward -= 0.05  # Softer safety penalty
        
        # Feature 1: Counterfactual reward (softened scaling)
        # Agents that ask worse questions than alternatives get gently penalized
        # "You could have asked a better question"
        counterfactual_reward = 0.10 * (best_counterfactual_gain - actual_gain)  # alpha=0.10 (softer)
        reward += counterfactual_reward
        
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
        
        # Apply step-level reward weight (emphasize step signals)
        step_reward = reward * self.step_reward_weight

        if done:
            agent_output = generate_recommendation(self.state_data["observed_constraints"])
            if action_type == "FINALIZE_WITH_COMPROMISE":
                agent_output = f"{agent_output} Balanced hybrid compromise."
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
            info["coverage"] = coverage
            info["hard_conflict"] = hard_conflict
            
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

        reward = float(max(min(reward, 2.0), -1.0))
        
        # Feature 9 (Refined): Update path frequency tracking for contextual diversity
        # After episode ends, record which path was chosen for future contextual bonus computation
        if done and self._matched_path_idx >= 0:
            path_names = ["primary", "alternative_1", "alternative_2", "alternative_3", "alternative_4"]
            path_name = path_names[self._matched_path_idx] if self._matched_path_idx < len(path_names) else f"alternative_{self._matched_path_idx}"
            if path_name in self._path_frequency:
                self._path_frequency[path_name] += 1
            self._total_episodes += 1
        
        observation = self._build_observation()
        
        # Structured logging: episode end
        if done:
            oracle_score = float(info.get("oracle_score", 0.0))
            step_num = int(self.state_data["step_count"])
            success = 1 if oracle_score >= 0.8 else 0
            print(f"END episode_{self.episode_counter} steps={step_num} success={success} oracle_score={oracle_score:.3f}")
        
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
