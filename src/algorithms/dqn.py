"""Deep Q-Network (DQN) for discrete action spaces.

Reference: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.mlp import MLP
from buffers.replay_buffer import ReplayBuffer


class DQN:
    """DQN agent with experience replay and a target network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        device: Optional[torch.device] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or torch.device("cpu")

        # Networks
        self.q_net = MLP(obs_dim, action_dim, list(hidden_sizes)).to(self.device)
        self.target_net = MLP(obs_dim, action_dim, list(hidden_sizes)).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim, self.device)
        self._update_counter = 0

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(obs, np.array([action]), reward, next_obs, done)

    # ------------------------------------------------------------------
    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs = batch["obs"]
        actions = batch["actions"].long().squeeze(1)
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Current Q values
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1))

        # Target Q values (no gradient)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Periodically update the target network
        self._update_counter += 1
        if self._update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"q_net": self.q_net.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
