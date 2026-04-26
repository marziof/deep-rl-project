# Implement experiment on cartpole environment

from src.utils.seed import set_seed
import gymnasium as gym

# -----------------------
# Run one episode
# -----------------------
def run_episode(env, agent):
    """
    Run a single episode using the given agent and environment, returning the total reward.
    Args:
    - env: The environment to interact with (e.g., CartPole-v1)
    - agent: An instance of an agent that can choose actions based on states
    Returns:
    - total_reward: The cumulative reward obtained during the episode
    """
    # 1. Reset the environment to start a new episode
    state, _ = env.reset()
    done = False
    total_reward = 0

    # 2. Loop until the episode is over (terminated or truncated)
    while not done:
        # Agent chooses an action based on the current state
        action = agent.act(state)
        # Execute the action and observe the next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Check if the episode has ended
        done = terminated or truncated
        # Update the current state and accumulate the reward
        state = next_state
        total_reward += reward

    return total_reward

# -----------------------
# Main experiment loop
# -----------------------
def run_experiment(agent, n_episodes=100, seed=0):
    """
    Experiment loop to run multiple episodes with a random agent and collect rewards.
    Args:
    - agent: An instance of an agent that can choose actions based on states
    - n_episodes: The number of episodes to run for the experiment
    - seed: The random seed for reproducibility
    Returns:
    - rewards: A list of total rewards obtained from each episode
    """
    set_seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)  # Set environment seed for reproducibility
    env.action_space.seed(seed)

    rewards = []

    for ep in range(n_episodes):
        r = run_episode(env, agent)
        rewards.append(r)
        print(ep, r)

    env.close()
    return rewards


#--------------
# Run experiments across seeds
#--------------
def run_experiments(agent_fn, seeds, n_episodes=100):
    all_rewards = []

    for seed in seeds:
        agent = agent_fn()
        rewards = run_experiment(agent, n_episodes=n_episodes, seed=seed)
        all_rewards.append(rewards)

    return all_rewards