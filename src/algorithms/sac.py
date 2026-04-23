"""Soft Actor-Critic (SAC) for continuous action spaces.

Reference: Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy Maximum
Entropy Deep Reinforcement Learning with a Stochastic Actor"
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.mlp import ActorContinuous, Critic
from buffers.replay_buffer import ReplayBuffer


class SAC:
    """SAC agent with automatic entropy tuning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        hidden_sizes: List[int] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        ent_coef: Union[float, str] = "auto",
        target_entropy: Union[float, str] = "auto",
        buffer_size: int = 100_000,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        # Actor
        self.actor = ActorContinuous(
            obs_dim, action_dim, list(hidden_sizes), action_scale
        ).to(self.device)

        # Twin critics and targets
        self.critic1 = Critic(obs_dim, action_dim, list(hidden_sizes)).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, list(hidden_sizes)).to(self.device)
        self.target_critic1 = Critic(obs_dim, action_dim, list(hidden_sizes)).to(
            self.device
        )
        self.target_critic2 = Critic(obs_dim, action_dim, list(hidden_sizes)).to(
            self.device
        )
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate,
        )

        # Entropy coefficient
        if ent_coef == "auto":
            self.target_entropy = (
                -float(action_dim) if target_entropy == "auto" else float(target_entropy)
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.log_alpha = torch.log(
                torch.tensor(float(ent_coef), device=self.device)
            )
            self.alpha_optimizer = None
            self.target_entropy = None

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim, self.device)

    # ------------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, det_action = self.actor.sample(obs_t)
        result = det_action if deterministic else action
        return result.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    def update(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_obs)
            q1_next = self.target_critic1(next_obs, next_actions)
            q2_next = self.target_critic2(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards + self.gamma * (1 - dones) * min_q_next

        # Critic update
        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        pi, log_pi, _ = self.actor.sample(obs)
        q1_pi = self.critic1(obs, pi)
        q2_pi = self.critic2(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = None
        if self.alpha_optimizer is not None:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update targets
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()) if alpha_loss is not None else 0.0,
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "log_alpha": self.log_alpha,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic1.load_state_dict(ckpt["critic1"])
        self.target_critic2.load_state_dict(ckpt["critic2"])
        self.log_alpha = ckpt["log_alpha"]
