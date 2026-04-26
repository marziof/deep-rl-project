# Implement experiment on cartpole environment

from src.utils.seed import set_seed
from src.utils.env import make_env
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
        # For DQN-style agent
        if hasattr(agent, "store"):
            agent.store(state, action, reward, next_state, done)
        if hasattr(agent, "update"):
            agent.update()
        if done and hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()
        #
        # Update the current state and accumulate the reward
        state = next_state
        total_reward += reward

    return total_reward

# -----------------------
# Main experiment loop
# -----------------------
def run_experiment(env, agent, n_episodes=100):
    rewards = []

    for ep in range(n_episodes):
        r = run_episode(env, agent)
        rewards.append(r)

    return rewards


#--------------
# Run experiments across seeds
#--------------
def run_experiments(agent_fn, seeds, n_episodes=100):
    all_rewards = []

    for seed in seeds:
        env = make_env("CartPole-v1", seed=seed)

        state_dim = env.observation_space.shape[0]
        action_space = env.action_space

        agent = agent_fn(action_space, state_dim)

        rewards = run_experiment(env, agent, n_episodes)

        all_rewards.append(rewards)

    return all_rewards