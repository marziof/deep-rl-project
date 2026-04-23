"""Proximal Policy Optimisation (PPO) for discrete and continuous spaces.

Reference: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.mlp import ActorDiscrete, MLP


class RolloutBuffer:
    """On-policy rollout storage for PPO."""

    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, action, reward, log_prob, value, done) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def __len__(self) -> int:
        return len(self.rewards)


class PPO:
    """PPO agent with clipped surrogate objective (discrete action spaces)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = (64, 64),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        n_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")

        self.actor = ActorDiscrete(obs_dim, action_dim, list(hidden_sizes)).to(
            self.device
        )
        self.critic = MLP(obs_dim, 1, list(hidden_sizes)).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )

        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if training:
                action, log_prob, entropy = self.actor.get_action(obs_t)
            else:
                logits = self.actor(obs_t)
                action = logits.argmax(dim=-1)
                log_prob = torch.zeros(1, device=self.device)
                entropy = torch.zeros(1, device=self.device)
            value = self.critic(obs_t)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    # ------------------------------------------------------------------
    def store_transition(self, obs, action, reward, log_prob, value, done) -> None:
        self.buffer.add(obs, action, reward, log_prob, value, done)

    # ------------------------------------------------------------------
    def _compute_gae(self, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.buffer.rewards
        values = self.buffer.values + [last_value]
        dones = self.buffer.dones

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_t = advantages_t + torch.tensor(
            self.buffer.values, dtype=torch.float32
        ).to(self.device)
        return advantages_t, returns_t

    # ------------------------------------------------------------------
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        advantages, returns = self._compute_gae(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.tensor(
            np.array(self.buffer.obs), dtype=torch.float32
        ).to(self.device)
        actions_t = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device)
        old_log_probs_t = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32
        ).to(self.device)

        n = len(self.buffer)
        indices = np.arange(n)
        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                idx = indices[start : start + self.batch_size]
                mb_obs = obs_t[idx]
                mb_actions = actions_t[idx]
                mb_old_lp = old_log_probs_t[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                new_log_probs, entropy = self.actor.evaluate(mb_obs, mb_actions)
                values_pred = self.critic(mb_obs).squeeze(1)

                ratio = (new_log_probs - mb_old_lp).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(
                    1 - self.clip_range, 1 + self.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                vf_loss = ((values_pred - mb_ret) ** 2).mean()
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        self.buffer.clear()
        return {
            "pg_loss": float(np.mean(pg_losses)),
            "vf_loss": float(np.mean(vf_losses)),
            "ent_loss": float(np.mean(ent_losses)),
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
