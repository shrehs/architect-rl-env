def shaped_step_reward(new_constraints_count: int, repeated_turn: bool) -> float:
    if new_constraints_count > 0:
        reward = min(0.3, 0.1 * float(new_constraints_count))
    elif repeated_turn:
        reward = -0.2
    else:
        reward = -0.05
    return float(max(-0.2, min(0.3, reward)))
