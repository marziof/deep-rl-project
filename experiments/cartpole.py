# Implement experiment on cartpole environment

from src.utils.seed import set_seed
from src.utils.env import make_env
from src.utils.logger import Logger
from src.evaluation import evaluate
import gymnasium as gym
import numpy as np
# -----------------------
# Run one episode
# -----------------------
def run_episode(env: gym.Env, agent, logger: Logger):
    """
    Run a single episode using the given agent and environment, returning the total reward.
    Args:
    - env: The environment to interact with (e.g., CartPole-v1)
    - agent: An instance of an agent that can choose actions based on states
    - logger: An instance of a Logger to log episode rewards and other metrics
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
        # if done and hasattr(agent, "decay_epsilon"):
        #     agent.decay_epsilon()
        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()
        #
        # Update the current state and accumulate the reward
        state = next_state
        total_reward += reward
    
    # log the total reward for this episode + decay epsilon if applicable
    logger.log_episode_reward(total_reward)
    if hasattr(agent, "decay_epsilon"):
        #agent.decay_epsilon()
        logger.log_epsilon(agent.eps)

    return total_reward

# -----------------------
# Main experiment loop
# -----------------------
def run_experiment(env, agent, logger, n_episodes=100, eval_interval=10, seed=0):
    #rewards = []
    for ep in range(n_episodes):
        r = run_episode(env, agent, logger)
        #rewards.append(r)
        # if hasattr(agent, "decay_epsilon"):
        #     agent.decay_epsilon()
        if ep % eval_interval == 0:
            scores = []
            for s in range(3):
                eval_env = make_env("CartPole-v1", seed=seed + 1000 + s)
                scores.append(evaluate(eval_env, agent, n_episodes=5))
                eval_env.close()
            logger.log_eval_reward(np.mean(scores))
    return logger


#--------------
# Run experiments across seeds
#--------------
def run_experiments(agent_fn, seeds, n_episodes=100, eval_interval=10):
    all_logs = []

    for seed in seeds:
        env = make_env("CartPole-v1", seed=seed)

        state_dim = env.observation_space.shape[0]
        action_space = env.action_space

        agent = agent_fn(action_space, state_dim)

        logger = Logger()

        try:
            logger = run_experiment(env, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed)
        finally:
            env.close()

        all_logs.append(logger)

    return all_logs

# def run_experiments(agent_fn, seeds, n_episodes=100):
#     all_rewards = []

#     for seed in seeds:
#         env = make_env("CartPole-v1", seed=seed)

#         state_dim = env.observation_space.shape[0]
#         action_space = env.action_space

#         agent = agent_fn(action_space, state_dim)

#         if hasattr(agent, "q_net"):
#             import torch
#             agent.q_net.train()
#             params_before = list(agent.q_net.parameters())[0].clone()

#             # force a few updates
#             for _ in range(10):
#                 agent.update()

#             params_after = list(agent.q_net.parameters())[0]
#             print("Weights changed:", not torch.equal(params_before, params_after))
#             print("Buffer size:", len(agent.buffer))
#             print("Epsilon:", agent.eps)
#         try:
#             rewards = run_experiment(env, agent, n_episodes)
#         finally:
#             env.close()

#         all_rewards.append(rewards)

#     return all_rewards
