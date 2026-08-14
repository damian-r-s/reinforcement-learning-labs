import pytest
import numpy as np
from tools.toolkit import update_estimate

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