import numpy as np

from src.utils.seed import set_seed
from src.utils.plotting import plot_learning_curve
from src.utils.env import make_env
from src.utils.stats import compute_stats

from src.algorithms.sac import SACAgent

sac_logs = run_experiments(
    agent_fn=lambda action_space, state_dim: SACAgent(
        action_space=action_space,
        state_dim=state_dim,
        gamma=0.99,
        batch_size=256,
        buffer_capacity=int(1e6),
        target_update_freq=1000,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        gradient_steps=1
    ),
    seeds=[0, 1, 2],
    n_episodes=300,
    eval_interval=10
)

rewards_sac = [logger.episode_rewards for logger in sac_logs] 
mean_rewards_sac, std_rewards_sac = compute_stats(rewards_sac)
episodes = np.arange(len(mean_rewards_sac))

plot_learning_curve(episodes, mean_rewards_sac, std_rewards_sac, title="Pendulum Learning Curve1 (3 seeds)")

# plot evaluation rewards
eval_rewards_sac = [logger.eval_rewards for logger in sac_logs]
mean_eval_rewards_sac, std_eval_rewards_sac = compute_stats(eval_rewards_sac)
eval_episodes = np.arange(len(mean_eval_rewards_sac)) * 10
plot_learning_curve(eval_episodes, mean_eval_rewards_sac, std_eval_rewards_sac, title="Pendulum Learning Curve2 (3 seeds)")

# plot episode reward (for sac) with moving average

# mva_ep_rewards_sac = [logger.moving_average(window=20) for logger in sac_logs]
# mean_ep_rewards_sac, std_ep_rewards_sac = compute_stats(mva_ep_rewards_sac)
# episodes = np.arange(len(mean_ep_rewards_sac))
# plot_learning_curve(episodes, mean_ep_rewards_sac, std_ep_rewards_sac)


