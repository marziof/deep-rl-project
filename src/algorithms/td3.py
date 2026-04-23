"""Twin Delayed Deep Deterministic Policy Gradient (TD3).

Reference: Fujimoto et al. (2018) "Addressing Function Approximation Error
in Actor-Critic Methods"
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.mlp import ActorDeterministic, Critic
from buffers.replay_buffer import ReplayBuffer


class TD3:
    """TD3 agent with target policy smoothing and delayed policy updates."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        hidden_sizes: List[int] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.exploration_noise = exploration_noise
        self.action_scale = action_scale
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self._update_counter = 0

        # Actor
        self.actor = ActorDeterministic(
            obs_dim, action_dim, list(hidden_sizes), action_scale
        ).to(self.device)
        self.target_actor = ActorDeterministic(
            obs_dim, action_dim, list(hidden_sizes), action_scale
        ).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Twin critics
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

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim, self.device)

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, training: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        if training:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -self.action_scale, self.action_scale)
        return action

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

        self._update_counter += 1
        batch = self.buffer.sample(self.batch_size)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.target_noise
            ).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.target_actor(next_obs) + noise).clamp(
                -self.action_scale, self.action_scale
            )
            q1_next = self.target_critic1(next_obs, next_actions)
            q2_next = self.target_critic2(next_obs, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(q1_next, q2_next)

        # Critic update
        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = None
        if self._update_counter % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update all targets
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

            actor_loss_val = float(actor_loss.item())

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_val if actor_loss_val is not None else 0.0,
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_actor.load_state_dict(ckpt["actor"])
        self.target_critic1.load_state_dict(ckpt["critic1"])
        self.target_critic2.load_state_dict(ckpt["critic2"])
