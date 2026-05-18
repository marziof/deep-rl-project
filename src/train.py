import numpy as np
import gymnasium as gym
from src.utils.seed import set_seed
from src.utils.env import make_env
from src.utils.logger import Logger
from src.evaluation import *
from src.utils.env import *
from src.utils.stats import compute_stats
from src.utils.plotting import plot_learning_curve
from experiments.pendulum_sac import run_episode as pendulum_run_episode
from experiments.cartpole import run_episode as cartpole_run_episode
from experiments.cartpole import *


def run_experiments(env_name, algo_name, agent_fn, seeds, n_episodes=100, n_iterations=100,eval_interval=10, create_videos=False, video_interval=500, save_dir=None):
    all_logs = []
    if algo_name != "ppo":
        for seed in seeds:
            env = make_env(env_name, seed=seed)

            state_dim = env.observation_space.shape[0]
            action_space = env.action_space

            agent = agent_fn(action_space, state_dim)

            logger = Logger(algo_name = algo_name, seed=seed, env_name=env_name)
            try:
                if env_name=="Pendulum-v1":
                    logger = run_experiment(env_name,algo_name,env, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed, save_dir=save_dir)
                elif env_name=="CartPole-v1":
                    logger = run_experiment(env_name,algo_name,env, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed, save_dir=save_dir)
            finally:
                env.close()

            all_logs.append(logger)

        return all_logs
    else:
        for seed in seeds:        
            env_test = gym.make(env_name) # we will create vectorized envs later, here we just need it to get the state and action space dimensions for the agent initialization
            state_dim = env_test.observation_space.shape[0]
            action_space = env_test.action_space

            agent = agent_fn(action_space, state_dim)
            envs  = gym.vector.SyncVectorEnv([
                lambda: gym.make(env_name) for _ in range(agent.n_actors)
            ])
            logger = Logger(algo_name = algo_name, seed=seed, env_name=env_name)

            try:
                #logger = run_experiment_PPO(envs, env_name, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed, create_videos=create_videos, video_interval=video_interval, save_dir=save_dir)
                logger = run_experiment(env_name, algo_name, None, agent, logger, n_episodes, eval_interval=eval_interval, seed=seed, create_videos=create_videos, video_interval=video_interval, save_dir=save_dir, env_vector=envs, n_iterations=n_iterations)
            finally:
                envs.close()
                env_test.close()

            all_logs.append(logger)

        return all_logs

def run_experiment(env_name, algo_name, env, agent, logger, n_episodes=100, n_iterations=100,eval_interval=10, seed=0, create_videos=False, video_interval=500, env_vector=None, save_dir=None):
    #rewards = []
    if algo_name != "ppo":
        for ep in range(n_episodes):
            if env_name=="Pendulum-v1":
                r, n_steps = pendulum_run_episode(env, agent, logger,algo_name=algo_name)
            elif env_name=="CartPole-v1":
                r, n_steps = cartpole_run_episode(env,agent, logger,algo_name=algo_name)
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
    else:
        for it in range(n_iterations):
            print(f"Seed: {seed}, iteration: {it}")
            r = run_PPO_iteration(env_vector, agent, logger)
            
            
            if it % eval_interval == 0:
                scores = []
                for s in range(3):
                    if create_videos and s == 0 and (it%video_interval == 0):
                        eval_env = make_env_render(env_name, seed=seed + 1000 + s)
                        scores.append(evaluate_PPO(eval_env, agent, n_episodes=5, visualize=True, video_title=f"{env_name}_ppo_seed_{seed}_{s}_episode_{it}"), save_dir=save_dir)
                        eval_env.close()
                    else:
                        eval_env = make_env(env_name, seed=seed + 1000 + s)
                        scores.append(evaluate_PPO(eval_env, agent, n_episodes=5, visualize=False))
                        eval_env.close()
                logger.log_eval_reward(np.mean(scores))
        return logger
    
def run_PPO_iteration(env_vector, agent, logger): #Equivalent of "episode" for PPO, for the plots.
    '''Runs one iteration of the PPO algorithm, which consists of collecting trajectories from multiple actors, estimating advantages and value targets, and updating the actor and critic networks.'''  
    # Collect trajectories from multiple actors (data collection phase)
    total_avg_reward = 0
    states, _ = env_vector.reset()
    steps_in_iteration = agent.time_per_actor * len(env_vector.envs)
    for t in range(agent.time_per_actor):
        actions, log_probs = agent.act(states) # returns actions for all actors
        next_states, rewards, terms, truncs, _ = env_vector.step(actions)
        #  Compute TDs
        agent.store(t, states, actions, rewards, next_states, log_probs, terms | truncs)
        states = next_states
        total_avg_reward += np.mean(rewards)
    agent.calculate_advantages() #PHASE 1: Estimate advantages and value targets for the collected trajectories using GAE
    loss = agent.update() #PHASE 2: Update the actor and critic networks using the collected trajectories and advantage estimations
    logger.log_episode_reward(total_avg_reward, n_steps)
    logger.log_loss(loss)
    return total_avg_reward

