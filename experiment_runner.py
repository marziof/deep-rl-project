import yaml
import sys
import os
import importlib
import numpy as np
import pandas as pd
import gymnasium as gym
import argparse
import shutil
from src.utils.seed import set_seed
from src.utils.env import make_env
from src.utils.logger import Logger
from src.evaluation import evaluate
from src.utils.stats import compute_stats
from src.utils.plotting import plot_learning_curve
from src.train import*
from datetime import datetime

# Mapping algorithm requirements to action space types
ALGO_COMPATIBILITY = {
    "dqn": "discrete",
    "ddqn": "discrete",
    "sac": "continuous",
    "td3": "continuous",
    "ppo": "both"
}

def get_environment_filename(env_type):
    if env_type=="Pendulum-v1":
        return "pendulum_sac"
    elif env_type=="CartPole-v1":
        return "cartpole"

def check_compatibility(env_name, algo_name):
    # Temporary env to check space type
    temp_env = gym.make(env_name)
    is_discrete = isinstance(temp_env.action_space, gym.spaces.Discrete)
    temp_env.close()

    required = ALGO_COMPATIBILITY.get(algo_name.lower())
    
    if required == "discrete" and not is_discrete:
        return False, f"{algo_name.upper()} only works with Discrete envs (like CartPole)."
    if required == "continuous" and is_discrete:
        return False, f"{algo_name.upper()} only works with Continuous envs (like Pendulum)."
    return True, ""

def main():
    set_seed(42)
    # Load configuratin
    parser = argparse.ArgumentParser(description="Run RL Experiments")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/conf.yaml", 
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exp_name = config['experiment'].get('exp_name')
    if not exp_name:
        exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    save_dir = os.path.join("results", "plots", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    config_save_path = os.path.join(save_dir, "config_used.yaml")
    shutil.copy(args.config, config_save_path)
    print(f"Config saved to: {config_save_path}")

    # Experiment settings
    env_type = config['experiment']['env_name']
    algo_name = config['experiment']['algo']
    seeds = config['experiment']['seeds']
    n_episodes = config['experiment']['n_episodes']
    n_iterations = config['experiment']['n_iterations']
    eval_interval = config['experiment']['eval_interval']
    exp_name = config['experiment']['exp_name']
    create_videos = config['experiment']['create_videos']
    video_interval = config['experiment']['video_interval']

    compatible, message = check_compatibility(env_type, algo_name)
    if not compatible:
        print(f"ERROR: {message}")
        sys.exit(1) # Exit before doing heavy imports

    # Import of the experiements
    exp_module_name = f"experiments.{get_environment_filename(env_type)}"
    #exp_module = importlib.import_module(exp_module_name)

    algo_module = importlib.import_module(f"src.algorithms.{algo_name}")
    # Handling special case of PPO algo that can accept continuous and discrete environments
    temp_env = gym.make(env_type)
    is_discrete = isinstance(temp_env.action_space, gym.spaces.Discrete)
    temp_env.close()
    if algo_name == "ppo" and not is_discrete:
        agent_class = getattr(algo_module, "PPOAgentContinuous")
    else:
        agent_class = getattr(algo_module, f"{algo_name.upper()}Agent")
        
    # Launch experiment 
    print(f"--- Launching {algo_name.upper()} on {env_type.upper()} ---")
    all_logs = run_experiments(
        env_name=env_type,
        algo_name=algo_name,
        agent_fn=lambda action_space, 
        state_dim: agent_class(
            action_space=action_space,
            state_dim=state_dim,
            **config['algos'][algo_name],
        ),
        seeds=seeds,
        n_episodes=n_episodes,
        n_iterations=n_iterations,
        eval_interval=eval_interval,
        create_videos=create_videos,
        video_interval=video_interval,
        save_dir=save_dir,
    )

    # Plotting
    train_rewards = [logger.episode_rewards for logger in all_logs] 
    mean_r, std_r = compute_stats(train_rewards)
    plot_learning_curve(np.arange(len(mean_r)), mean_r, std_r, 
    title=f"{env_type.capitalize()} Training ({algo_name.upper()})", exp_name=exp_name)

    # Evaluation rewards
    eval_rewards = [logger.eval_rewards for logger in all_logs]
    mean_eval, std_eval = compute_stats(eval_rewards)
    plot_learning_curve(np.arange(len(mean_eval)) * config['experiment']['eval_interval'], 
    mean_eval, std_eval, title=f"{env_type.capitalize()} Evaluation ({algo_name.upper()})", exp_name=exp_name)

    # save logger to df
    SAVE_DIR = os.path.join("results", "data", exp_name)
    SAVE_PATH = f"{SAVE_DIR}/{algo_name}_{env_type}_logs_nep{n_episodes}_eps09999.csv"
    os.makedirs(SAVE_DIR, exist_ok=True)
    df_list = []
    for logger in all_logs:
        df = logger.to_dataframe()
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"Logs saved to {SAVE_PATH} ")

if __name__ == "__main__":
    main()
