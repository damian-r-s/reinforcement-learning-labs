import pytest
import numpy as np
from tools.toolkit import update_estimate, true_action_values, get_reward, epsilon_greedy_action

def test_update_estimate_first_reward():
    new_estimate = update_estimate(reward=1, n=1, q_old=0)

    assert new_estimate == 1

def test_update_estimate_raises_on_zero_n():
    with pytest.raises(ValueError):
        update_estimate(reward=1, n = 0, q_old=0)

def test_averaging_reward():
    new_estimate  = update_estimate(reward=1, n=1, q_old=0)
    new_estimate2 = update_estimate(reward=1, n=2, q_old=new_estimate)

    assert new_estimate2 == 1

def test_true_action_values():
    rng = np.random.default_rng(0)

    array = true_action_values(k=10, rng=rng)

    assert len(array) == 10

def test_true_action_values_is_deterministic():
    values_a = true_action_values(k = 10, rng=np.random.default_rng(42))
    values_b = true_action_values(k = 10, rng=np.random.default_rng(42))

    assert np.array_equal(values_a, values_b)

def test_get_reward_is_deterministic():
    true_values = np.array([1.0, 2.0, 3.0])
    reward_a = get_reward(action=0, true_values=true_values, rng=np.random.default_rng(1))
    reward_b = get_reward(action=0, true_values=true_values, rng=np.random.default_rng(1))
    assert reward_a == reward_b

def test_get_reward_avarages_close_to_true_value():
    true_values = np.array([5.0])
    rng = np.random.default_rng(7)

    samples = [get_reward(action=0, true_values=true_values, rng=rng) for _ in range(10_000)]

    mean = np.mean(samples)
    
    assert abs(mean - 5.0) < 0.1


def test_epsilon_greedy_action_always_exploit():
    q_estimates = np.array([4.0, 5.0])

    rng = np.random.default_rng(7)
    idx = epsilon_greedy_action(q_estimates=q_estimates, epsilon=0, rng=rng)

    assert idx == 1

def test_epsilon_greedy_action_always_exploring():
    q_estimates = np.array([4.0, 5.0])
    
    rng = np.random.default_rng(7)
    idx = epsilon_greedy_action(q_estimates=q_estimates, epsilon=1, rng=rng)
    
    assert 0 <= idx < len(q_estimates)