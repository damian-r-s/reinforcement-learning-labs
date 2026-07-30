def update_estimate(reward, n, q_old):
    if n == 0:
        raise ValueError("n must be >= 1")

    q_new = q_old + (reward - q_old) / n
    return q_new