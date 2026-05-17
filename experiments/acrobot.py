# Implement experiment on cartpole environment

import torch

from src.utils.seed import set_seed
from src.utils.env import make_env, make_env_render
from src.utils.logger import Logger
from src.evaluation import evaluate, evaluate_PPO
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
    episode_losses = []

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
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)
            
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

    if len(episode_losses) > 0:
        logger.log_loss(np.mean(episode_losses))
    else:
        logger.log_loss(0.0)

    return total_reward

# -----------------------
# Main experiment loop
# -----------------------
def run_experiment(env, agent, logger, n_episodes=100, eval_interval=10, seed=0, create_videos=False):
    #rewards = []
    for ep in range(n_episodes):
        r = run_episode(env, agent, logger)
        #rewards.append(r)
        # if hasattr(agent, "decay_epsilon"):
        #     agent.decay_epsilon()
        if ep % eval_interval == 0:
            scores = []
            for s in range(3):
                if create_videos:
                    eval_env = make_env_render("Acrobot-v1", seed=seed + 1000 + s)
                    scores.append(evaluate(eval_env, agent, n_episodes=5, visualize=True, video_title=f"acrobot_seed_{seed}_{s}_episode_{ep}"))
                    eval_env.close()
                else:
                    eval_env = make_env("Acrobot-v1", seed=seed + 1000 + s)
                    scores.append(evaluate(eval_env, agent, n_episodes=5, visualize=False))
                    eval_env.close()
            logger.log_eval_reward(np.mean(scores))
    return logger


#--------------
# Run experiments across seeds
#--------------
def run_experiments(agent_fn, seeds, n_episodes=100, eval_interval=10, create_videos=False):
    all_logs = []

    for seed in seeds:
        env = make_env("Acrobot-v1", seed=seed)

        state_dim = env.observation_space.shape[0]
        action_space = env.action_space

        agent = agent_fn(action_space, state_dim)

        logger = Logger()

        try:
            logger = run_experiment(env, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed, create_videos=create_videos)
        finally:
            env.close()

        all_logs.append(logger)

    return all_logs

def run_PPO_iteration(env_vector, agent, logger): #Equivalent of "episode" for PPO, for the plots.
    '''Runs one iteration of the PPO algorithm, which consists of collecting trajectories from multiple actors, estimating advantages and value targets, and updating the actor and critic networks.'''  
    # Collect trajectories from multiple actors (data collection phase)
    total_avg_reward = 0
    states, _ = env_vector.reset()
    for _ in range(agent.time_per_actor):
        states = states
        actions, log_probs = agent.act(states) # returns actions for all actors
        next_states, rewards, terms, truncs, _ = env_vector.step(actions)
        #  Compute TDs
        dones = terms | truncs
        for i in range(agent.n_actors):
            agent.store(states[i], actions[i], rewards[i], next_states[i], log_probs[i], dones[i])
        states = next_states
        total_avg_reward += np.mean(rewards)
    
    agent.calculate_advantages() #PHASE 1: Estimate advantages and value targets for the collected trajectories using GAE
    loss = agent.update() #PHASE 2: Update the actor and critic networks using the collected trajectories and advantage estimations
    logger.log_episode_reward(total_avg_reward)
    logger.log_loss(loss)
    return total_avg_reward

def run_experiment_acrobot_PPO(env_vector, agent, logger, n_iterations=100, eval_interval=10, seed=0, create_videos=False, video_interval=500):
    #rewards = []
    for it in range(n_iterations):
        print(f"Seed: {seed}, iteration: {it}")
        r = run_PPO_iteration(env_vector, agent, logger)
        
        
        if it % eval_interval == 0:
            scores = []
            for s in range(3):
                if create_videos and s == 0 and (it%video_interval == 0): 
                    eval_env = make_env_render("Acrobot-v1", seed=seed + 1000 + s)
                    scores.append(evaluate_PPO(eval_env, agent, n_episodes=5, visualize=True, video_title=f"acrobot_ppo_seed_{seed}_episode_{it}"))
                    eval_env.close()
                else:
                    eval_env = make_env("Acrobot-v1", seed=seed + 1000 + s)
                    scores.append(evaluate_PPO(eval_env, agent, n_episodes=5, visualize=False))
                    eval_env.close()
            logger.log_eval_reward(np.mean(scores))
    return logger

def run_experiments_acrobot_PPO(agent_fn, seeds, n_episodes=100, eval_interval=10, create_videos=False, video_interval=500):
    all_logs = []

    for seed in seeds:

        
        env_test = gym.make("Acrobot-v1") # we will create vectorized envs later, here we just need it to get the state and action space dimensions for the agent initialization
        state_dim = env_test.observation_space.shape[0]
        action_space = env_test.action_space

        agent = agent_fn(action_space, state_dim)
        envs  = gym.vector.SyncVectorEnv([
            lambda: gym.make("Acrobot-v1") for _ in range(agent.n_actors)
        ])
        logger = Logger()

        try:
            logger = run_experiment_acrobot_PPO(envs, agent, logger, n_iterations=n_episodes, eval_interval=eval_interval, seed=seed, create_videos=create_videos, video_interval=video_interval)
        finally:
            envs.close()
            env_test.close()

        all_logs.append(logger)

    return all_logs