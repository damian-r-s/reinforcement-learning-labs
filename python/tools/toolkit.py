from typing import Optional

import numpy as np

def update_estimate(reward, n, q_old):
    if n == 0:
        raise ValueError("n must be >= 1")

    q_new = q_old + (reward - q_old) / n
    return q_new

def true_action_values(k, mean=0.0, std=1.0, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()
    result = rng.normal(mean, std, size=k)
    return result

def get_reward(action, true_values, std=1.0, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()
    true_value = true_values[action]
    sample = rng.normal(true_value, std)
    return sample

def epsilon_greedy_action(q_estimates: np.ndarray, epsilon, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()

    if rng.random() < epsilon:
        random_idx = rng.integers(0, len(q_estimates))
        return random_idx
    else:
        exploit_idx = np.argmax(q_estimates)
        return exploit_idx