import numpy as np

def compute_stats(all_rewards):
    all_rewards = np.array(all_rewards)
    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)
    return mean, std