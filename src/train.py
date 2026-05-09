import numpy as np
import gymnasium as gym
from src.utils.seed import set_seed
from src.utils.env import make_env
from src.utils.logger import Logger
from src.evaluation import evaluate
from src.utils.stats import compute_stats
from src.utils.plotting import plot_learning_curve
from experiments.pendulum_sac import run_episode as pendulum_run_episode
from experiments.cartpole import run_episode as cartpole_run_episode


def run_experiments(env_name, algo_name, agent_fn, seeds, n_episodes=100, eval_interval=10):
    all_logs = []

    for seed in seeds:
        env = make_env(env_name, seed=seed)

        state_dim = env.observation_space.shape[0]
        action_space = env.action_space

        agent = agent_fn(action_space, state_dim)

        logger = Logger()

        try:
            if env_name=="Pendulum-v1":
                logger = run_experiment(env_name,env, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed)
            elif env_name=="CartPole-v1":
                logger = run_experiment(env_name,env, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed)
        finally:
            env.close()

        all_logs.append(logger)

    return all_logs

def run_experiment(env_name,env, agent, logger, n_episodes=100, eval_interval=10, seed=0):
    #rewards = []
    for ep in range(n_episodes):
        if env_name=="Pendulum-v1":
            r = pendulum_run_episode(env, agent, logger)
        elif env_name=="Cartpole-v1":
            r = cartpole_run_episode(env, agent, logger)
        #rewards.append(r)
        # if hasattr(agent, "decay_epsilon"):
        #     agent.decay_epsilon()
        if ep % eval_interval == 0:
            scores = []
            for s in range(3):
                eval_env = make_env(env_name, seed=seed + 1000 + s)
                scores.append(evaluate(eval_env, agent, n_episodes=5))
                eval_env.close()
            logger.log_eval_reward(np.mean(scores))
    return logger
