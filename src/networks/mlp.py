"""Shared MLP building blocks used by all algorithms."""

from typing import List, Optional, Type

import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[Type[nn.Module]] = None,
) -> nn.Sequential:
    """Return a fully-connected network as a ``nn.Sequential``."""
    layers: List[nn.Module] = []
    in_size = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_size, h))
        layers.append(activation())
        in_size = h
    layers.append(nn.Linear(in_size, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """General-purpose multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            input_dim, output_dim, hidden_sizes, activation, output_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorContinuous(nn.Module):
    """Gaussian actor for continuous action spaces (used by SAC / TD3)."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.trunk = build_mlp(obs_dim, hidden_sizes[-1], hidden_sizes[:-1])
        self.mu_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.action_scale = action_scale

    def forward(self, obs: torch.Tensor):
        x = self.trunk(obs)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        """Return a squashed action, its log-prob and the deterministic action."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = dist.log_prob(x_t)
        # Enforce action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic_action = torch.tanh(mu) * self.action_scale
        return action, log_prob, deterministic_action


class ActorDeterministic(nn.Module):
    """Deterministic actor for continuous action spaces (TD3)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, action_dim, hidden_sizes, output_activation=nn.Tanh)
        self.action_scale = action_scale

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.action_scale


class Critic(nn.Module):
    """Q-value network for actor-critic algorithms."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class ActorDiscrete(nn.Module):
    """Categorical actor for discrete action spaces (PPO)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, action_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_action(self, obs: torch.Tensor):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
