# Implement experiment on pendulum environment

from src.utils.seed import set_seed
from src.utils.env import make_env
from src.utils.logger import Logger
from src.evaluation import evaluate
import gymnasium as gym
import numpy as np
# -----------------------
# Run one episode
# -----------------------
def run_episode(env: gym.Env, agent, logger: Logger, algo_name="sac"):
    """
    Run a single episode using the given agent and environment, returning the total reward.
    Args:
    - env: The environment to interact with (e.g., CartPole-v1)
    - agent: An instance of an agent that can choose actions based on states
    - logger: An instance of a Logger to log episode rewards and other metrics
    Returns:
    - total_reward: The cumulative reward obtained during the episode
    """
    state, _ = env.reset()
    done = False
    total_reward = 0
    episode_losses = []
    steps_in_episode = 0

    if algo_name=="sac":
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps_in_episode += 1
            if hasattr(agent, "store"):
                agent.store(state, action, reward, next_state, done)
            if hasattr(agent, "update"):
                n_updates = getattr(agent, "gradient_steps", 1) 
                
                for _ in range(n_updates):
                    loss = agent.update()
                    if loss is not None: 
                        episode_losses.append(loss["q_loss"])
                
            state = next_state
            total_reward += reward
    
    elif algo_name=="td3":
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps_in_episode += 1
            if hasattr(agent, "store"):
                agent.store(state, action, reward, next_state, done)
            if hasattr(agent, "update"):
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
    
    logger.log_episode_reward(total_reward)
    logger.log_steps_in_episode(steps_in_episode)

    if len(episode_losses) > 0:
        logger.log_loss(np.mean(episode_losses))
    else:
        logger.log_loss(0.0)

    return total_reward

# # -----------------------
# # Main experiment loop
# # -----------------------
# def run_experiment(env, agent, logger, n_episodes=100, eval_interval=10, seed=0):
#     #rewards = []
#     for ep in range(n_episodes):
#         r = run_episode(env, agent, logger)
#         #rewards.append(r)
#         # if hasattr(agent, "decay_epsilon"):
#         #     agent.decay_epsilon()
#         if ep % eval_interval == 0:
#             scores = []
#             for s in range(3):
#                 eval_env = make_env("Pendulum-v1", seed=seed + 1000 + s)
#                 scores.append(evaluate(eval_env, agent, n_episodes=5, seed=seed + 1000 + s))
#                 eval_env.close()
#             logger.log_eval_reward(np.mean(scores))
#     return logger


# #--------------
# # Run experiments across seeds
# #--------------
# def run_experiments(agent_fn, seeds, n_episodes=100, eval_interval=10):
#     all_logs = []

#     for seed in seeds:
#         env = make_env("Pendulum-v1", seed=seed)

#         state_dim = env.observation_space.shape[0]
#         action_space = env.action_space

#         agent = agent_fn(action_space, state_dim)

#         logger = Logger()

#         try:
#             logger = run_experiment(env, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed)
#         finally:
#             env.close()

#         all_logs.append(logger)

#     return all_logs