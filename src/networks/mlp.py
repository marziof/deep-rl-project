# Implementation of MLP model used in deep RL algos

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=256 ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden_units=64)
        self.max_action = max_action

    def forward(self, state):
        return torch.tanh(self.net(state)) * self.max_action