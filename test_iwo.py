import numpy as np

from src.utils.seed import set_seed
from src.utils.plotting import plot_learning_curve
from src.utils.env import make_env
from src.utils.stats import compute_stats
from experiments.cartpole import run_experiments, run_experiments_PPO
from experiments.acrobot import run_experiments_acrobot_PPO, run_experiments_acrobot_PPO
from experiments.mountaincar import run_experiments_mountaincar_PPO, run_experiment_mountaincar_PPO
from experiments.lunarlander import run_experiments_lunarlander_PPO, run_experiment_lunarlander_PPO
from src.algorithms.dqn import DQNAgent
from src.algorithms.ppo import PPOAgent, PPOAgentContinuous

#===PPO testing===
set_seed(0)
eval_interval = 1
env_name = "CartPole-v1"
#hidden_dims = [8, 16, 32, 64, 128]
hidden_dim = 32
times_per_actor = [128, 256, 512, 1024, 2048]
for time_per_actor in times_per_actor:
    print(f"Running {env_name} PPO with time_per_actor={time_per_actor}...")
    ppo_logs = run_experiments_PPO(
        env_name=env_name,
        agent_fn=lambda action_space, state_dim: PPOAgent(
            action_space=action_space,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            gamma=0.99,
            lr=1e-3,
            n_actors = 2,
            time_per_actor = time_per_actor,
            n_epochs=5,
            batch_size=64,
            epsilon_clip=0.2
        ),
        seeds=[0],
        n_episodes=50,
        eval_interval=eval_interval,
        create_videos=False,
        video_interval=10 #MUST BE DIVISIBLE BY eval_interval, otherwise we will never save videos
    )

    eval_rewards_ppo = [logger.eval_rewards for logger in ppo_logs]
    mean_eval_rewards_ppo, std_eval_rewards_ppo = compute_stats(eval_rewards_ppo)
    eval_episodes = np.arange(len(mean_eval_rewards_ppo)) * eval_interval

    plot_learning_curve(eval_episodes, mean_eval_rewards_ppo, std_eval_rewards_ppo, save_figure=True, title=f"{env_name} PPO (time_per_actor={time_per_actor})", filename=f"{env_name}_ppo_time{time_per_actor}_scan")
