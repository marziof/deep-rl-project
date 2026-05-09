import yaml
import sys
import os
import importlib
import numpy as np
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
    eval_interval = config['experiment']['eval_interval']
    exp_name = config['experiment']['exp_name']

    compatible, message = check_compatibility(env_type, algo_name)
    if not compatible:
        print(f"ERROR: {message}")
        sys.exit(1) # Exit before doing heavy imports

    # Import of the experiements
    exp_module_name = f"experiments.{get_environment_filename(env_type)}"
    exp_module = importlib.import_module(exp_module_name)

    algo_module = importlib.import_module(f"src.algorithms.{algo_name}")
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
        eval_interval=eval_interval,
    )

    # Plotting
    train_rewards = [logger.episode_rewards for logger in all_logs] 
    mean_r, std_r = compute_stats(train_rewards)
    plot_learning_curve(np.arange(len(mean_r)), mean_r, std_r, 
    title=f"{env_type.capitalize()} Training ({algo_name.upper()})", exp_name=exp_name)

    # Evaluation Rewards
    eval_rewards = [logger.eval_rewards for logger in all_logs]
    mean_eval, std_eval = compute_stats(eval_rewards)
    plot_learning_curve(np.arange(len(mean_eval)) * config['experiment']['eval_interval'], 
    mean_eval, std_eval, title=f"{env_type.capitalize()} Evaluation ({algo_name.upper()})", exp_name=exp_name)

if __name__ == "__main__":
    main()
