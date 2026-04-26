# Implementation of DQN algo

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.networks.mlp import MLP
from src.buffers.replay_buffer import ReplayBuffer


#  algo - from https://zhuanlan.zhihu.com/p/468385820

# Q-learning with replay buffer and target network:
# 1. save target network params phi' <- phi
#   2. collect dataset {(s,a,s',r)} using some policy (e.g. epsilon-greedy), add to B
# N x   3. sample random batch from replay buffer B (s, a, r, s')
#   (K x)  4. phi <- phi - alpha * grad_phi(Q_phi)(Q_phi - [r + gamma*max_a' Q_phi'(s', a')])

# "classic" deep Q-learning algo:
# 1. take action a, observe (s,a,s',r), add to B
# 2. sample mini-batch from B, uniformly
# 3. compute yj = rj + gamma * max_a' Q_phi'(s', a') using target network Q_phi'
# 4. update phi <- phi - alpha * grad_phi(Q_phi)(Q_phi(s,a) - yj)
# 5. update phi', copy phi every N steps


class DQNAgent:

    def __init__(self, action_space, state_dim, gamma, batch_size, eps, eps_min, eps_decay, target_update_freq, buffer_capacity, lr):
        super().__init__()
        self.action_space = action_space
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        self.update_step = 0
        self.buffer = ReplayBuffer(buffer_capacity)
        self.q_net = MLP(input_dim=state_dim, output_dim=action_space.n) # input_dim = dim of state space, output_dim = dim of action space
        self.target_net = MLP(input_dim=state_dim, output_dim=action_space.n)
        self.target_net.load_state_dict(self.q_net.state_dict()) # initialize target net with same weights as q_net
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_net.to(self.device)


    def act(self, state):
        """Epsilon-greedy action selection, returns an action index based on the current state"""
        if random.random() < self.eps:
            return self.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    def update(self):
        self.update_step += 1
        # 1. Check if we have enough samples in the buffer to sample a batch
        if len(self.buffer) < self.batch_size:
            return
        if len(self.buffer) < 1000:
            return
        # 2. Sample an action
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones_tensor = (dones_tensor > 0.5).float()
        # 3. Compute target Q-values using the target network
        with torch.no_grad():
            # yj = rj + gamma * max_a' Q_phi'(s', a')
            max_next_q = self.target_net(next_states_tensor).max(1, keepdim=True)[0]
            target = rewards_tensor + self.gamma * (1 - dones_tensor) * max_next_q
        # 4. Compute the loss (MSE)
        q_values = self.q_net(states_tensor).gather(1, actions_tensor)
        loss = torch.nn.functional.mse_loss(q_values, target)
        # 5. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 6. update target network every N-steps
        if self.update_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)