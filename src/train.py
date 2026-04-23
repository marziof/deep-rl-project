"""Unified training entry-point.

Usage examples
--------------
Train DQN on CartPole:
    python src/train.py --algo dqn --config configs/dqn.yaml

Override env and seed at the command line:
    python src/train.py --algo sac --config configs/sac.yaml --env Pendulum-v1 --seed 0
"""

import argparse
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import yaml

# Make sure the src/ tree is importable when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import gymnasium as gym
import torch

from utils.logger import Logger
from utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def _evaluate(agent, env_name: str, n_episodes: int, seed: int, algo: str) -> float:
    """Run *n_episodes* greedy episodes and return the mean return."""
    eval_env = gym.make(env_name)
    returns = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            if algo in ("sac",):
                action = agent.select_action(obs, deterministic=True)
            else:
                # DQN, TD3, PPO all accept training=False / no kwarg
                action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total += reward
        returns.append(total)
    eval_env.close()
    return float(np.mean(returns))


def _get_dims(env: gym.Env):
    obs_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = int(env.action_space.n)
        action_scale = 1.0
        is_continuous = False
    else:
        action_dim = int(np.prod(env.action_space.shape))
        action_scale = float(env.action_space.high[0])
        is_continuous = True
    return obs_dim, action_dim, action_scale, is_continuous


# ---------------------------------------------------------------------------
# Per-algorithm training loops
# ---------------------------------------------------------------------------

def _train_offpolicy(agent, env, cfg: Dict[str, Any], logger: Logger, algo: str) -> None:
    """Off-policy loop shared by DQN, SAC, TD3."""
    total_timesteps = cfg["total_timesteps"]
    learning_starts = cfg.get("learning_starts", 1000)
    train_freq = cfg.get("train_freq", 1)
    gradient_steps = cfg.get("gradient_steps", 1)
    log_interval = cfg.get("log_interval", 1000)
    eval_interval = cfg.get("eval_interval", 5000)
    n_eval_episodes = cfg.get("n_eval_episodes", 10)
    seed = cfg.get("seed", 42)
    env_name = cfg["env"]

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_len = 0
    ep_num = 0

    for t in range(1, total_timesteps + 1):
        action = agent.select_action(obs, training=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward
        ep_len += 1

        if done:
            ep_num += 1
            logger.log(
                {"episode": ep_num, "ep_reward": round(ep_reward, 2), "ep_len": ep_len},
                step=t,
            )
            ep_reward = 0.0
            ep_len = 0
            obs, _ = env.reset()

        if t >= learning_starts and t % train_freq == 0:
            for _ in range(gradient_steps):
                agent.update()

        if t % log_interval == 0:
            logger.log({"timestep": t}, step=t)

        if t % eval_interval == 0:
            mean_ret = _evaluate(agent, env_name, n_eval_episodes, seed, algo)
            logger.log({"eval_mean_reward": round(mean_ret, 2)}, step=t)


def _train_ppo(agent, env, cfg: Dict[str, Any], logger: Logger) -> None:
    """On-policy loop for PPO."""
    total_timesteps = cfg["total_timesteps"]
    n_steps = cfg.get("n_steps", 2048)
    log_interval = cfg.get("log_interval", 1)
    eval_interval = cfg.get("eval_interval", 10000)
    n_eval_episodes = cfg.get("n_eval_episodes", 10)
    seed = cfg.get("seed", 42)
    env_name = cfg["env"]

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_num = 0
    t = 0

    while t < total_timesteps:
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, log_prob, value, done)
            obs = next_obs
            ep_reward += reward
            t += 1

            if done:
                ep_num += 1
                if ep_num % log_interval == 0:
                    logger.log(
                        {"episode": ep_num, "ep_reward": round(ep_reward, 2)}, step=t
                    )
                ep_reward = 0.0
                obs, _ = env.reset()

        # Get bootstrap value
        _, _, last_value = agent.select_action(obs)
        losses = agent.update(last_value=last_value)
        logger.log(losses, step=t)

        if t % eval_interval < n_steps:
            mean_ret = _evaluate(agent, env_name, n_eval_episodes, seed, "ppo")
            logger.log({"eval_mean_reward": round(mean_ret, 2)}, step=t)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(algo: str, obs_dim: int, action_dim: int, action_scale: float,
                 cfg: Dict[str, Any], device: torch.device):
    hidden_sizes = cfg.get("hidden_sizes", [256, 256])

    if algo == "dqn":
        from algorithms.dqn import DQN
        return DQN(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            learning_rate=cfg.get("learning_rate", 1e-3),
            gamma=cfg.get("gamma", 0.99),
            epsilon_start=cfg.get("epsilon_start", 1.0),
            epsilon_end=cfg.get("epsilon_end", 0.05),
            epsilon_decay=cfg.get("epsilon_decay", 0.995),
            buffer_size=cfg.get("buffer_size", 50_000),
            batch_size=cfg.get("batch_size", 64),
            target_update_freq=cfg.get("target_update_freq", 1000),
            device=device,
        )
    elif algo == "ppo":
        from algorithms.ppo import PPO
        return PPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            learning_rate=cfg.get("learning_rate", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            clip_range=cfg.get("clip_range", 0.2),
            vf_coef=cfg.get("vf_coef", 0.5),
            ent_coef=cfg.get("ent_coef", 0.01),
            n_epochs=cfg.get("n_epochs", 10),
            batch_size=cfg.get("batch_size", 64),
            max_grad_norm=cfg.get("max_grad_norm", 0.5),
            device=device,
        )
    elif algo == "sac":
        from algorithms.sac import SAC
        return SAC(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_scale=action_scale,
            hidden_sizes=hidden_sizes,
            learning_rate=cfg.get("learning_rate", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            tau=cfg.get("tau", 5e-3),
            ent_coef=cfg.get("ent_coef", "auto"),
            target_entropy=cfg.get("target_entropy", "auto"),
            buffer_size=cfg.get("buffer_size", 100_000),
            batch_size=cfg.get("batch_size", 256),
            device=device,
        )
    elif algo == "td3":
        from algorithms.td3 import TD3
        return TD3(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_scale=action_scale,
            hidden_sizes=hidden_sizes,
            learning_rate=cfg.get("learning_rate", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            tau=cfg.get("tau", 5e-3),
            policy_delay=cfg.get("policy_delay", 2),
            target_noise=cfg.get("target_noise", 0.2),
            target_noise_clip=cfg.get("target_noise_clip", 0.5),
            exploration_noise=cfg.get("exploration_noise", 0.1),
            buffer_size=cfg.get("buffer_size", 100_000),
            batch_size=cfg.get("batch_size", 256),
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deep RL agent")
    parser.add_argument("--algo", required=True, choices=["dqn", "ppo", "sac", "td3"])
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--env", default=None, help="Override environment name")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--log_dir", default="results/logs", help="Directory for logs"
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.env is not None:
        cfg["env"] = args.env
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = _make_env(cfg["env"], seed)
    obs_dim, action_dim, action_scale, is_continuous = _get_dims(env)
    print(
        f"Env: {cfg['env']} | obs_dim={obs_dim} | action_dim={action_dim} "
        f"| continuous={is_continuous}"
    )

    run_name = f"{args.algo}_{cfg['env']}_{seed}_{int(time.time())}"
    log_dir = os.path.join(args.log_dir, run_name)
    logger = Logger(log_dir)

    agent = _build_agent(args.algo, obs_dim, action_dim, action_scale, cfg, device)

    if args.algo == "ppo":
        _train_ppo(agent, env, cfg, logger)
    else:
        _train_offpolicy(agent, env, cfg, logger, algo=args.algo)

    env.close()
    logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
