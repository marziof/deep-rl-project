"""Experience replay buffer shared by DQN, SAC and TD3."""

from collections import deque
from typing import Dict, Optional, Tuple
import random

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular experience replay buffer.

    Stores transitions ``(obs, action, reward, next_obs, done)`` and
    returns randomly sampled mini-batches as PyTorch tensors.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.max_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self._storage: deque = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._storage.append((obs, action, reward, next_obs, done))

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch = random.sample(self._storage, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        return {
            "obs": torch.as_tensor(np.array(obs), dtype=torch.float32).to(self.device),
            # Actions are stored as float32 for API uniformity across continuous and
            # discrete algorithms.  DQN converts them back to long inside its update().
            "actions": torch.as_tensor(np.array(actions), dtype=torch.float32).to(
                self.device
            ),
            "rewards": torch.as_tensor(
                np.array(rewards, dtype=np.float32)
            ).unsqueeze(1).to(self.device),
            "next_obs": torch.as_tensor(np.array(next_obs), dtype=torch.float32).to(
                self.device
            ),
            "dones": torch.as_tensor(
                np.array(dones, dtype=np.float32)
            ).unsqueeze(1).to(self.device),
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._storage)
