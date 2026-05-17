# Utility functions for environment creation and management in reinforcement learning experiments.

import gymnasium as gym


#-----------------------
# Function to create and seed the environment for reproducibility
#-----------------------
def make_env(env_name, seed):
    """
    Create a Gym environment and set the seed for reproducibility.
    Args:
    - env_name: The name of the environment to create.
    - seed: The random seed to set.
    Returns:
    - The created and seeded environment.
    """
    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env

def make_env_render(env_name, seed):
    """
    Create a Gym environment with video rendering and set the seed for reproducibility.
    Args:
    - env_name: The name of the environment to create.
    - seed: The random seed to set.
    Returns:
    - The created and seeded environment.
    """
    env = gym.make(env_name, render_mode="rgb_array")
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env