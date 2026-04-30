# Implementation of TD3 algo

import random
from turtle import done, mode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.networks.mlp import MLP
from src.networks.actor_net import Actor
from src.buffers.replay_buffer import ReplayBuffer

# article: https://arxiv.org/pdf/1802.09477

# Algorithm TD3
# Initialize critic networks Qθ1, Qθ2, and actor network πφ with random parameters θ1, θ2, φ
# Initialize target networks to same weights as critic networks
# Initialize replay buffer B
# for t = 1 to T do
    # Select action with exploration noise a ∼ πφ(s) + eps, eps∼ N (0, σ) and observe reward r and new state s'
    # Store transition tuple (s, a, r, s') in B
    # Sample mini-batch of N transitions (s, a, r, s') from B
    # update ã, y
    # update critics to min MSE loss bt y and Qθi(s, a)
    # Delayed policy updates:
    # if t mod d then
        # Update actor by the deterministic policy gradient: grad phi J_phi
        # Update target networks: theta' and phi'
    # end if
# end for



class TD3Agent:

    def __init__(self, action_space, state_dim, gamma, batch_size, policy_delay, buffer_capacity, lr, sigma, sigma_tilde, c, tau):
        #super().__init__()
        self.action_space = action_space
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_delay = policy_delay # d
        self.sigma = sigma
        self.sigma_tilde = sigma_tilde
        self.c = c
        self.tau = tau
        self.update_step = 0 # time t
        self.buffer = ReplayBuffer(buffer_capacity)
        action_dim = action_space.shape[0]
        max_action = action_space.high[0]
        self.crit_net1 = MLP(input_dim=state_dim + action_dim, output_dim=1) # input_dim = dim of state space, output_dim = dim of action space
        self.crit_net2 = MLP(input_dim=state_dim + action_dim, output_dim=1)
        self.actor_net = Actor(state_dim, action_dim, max_action)
        self.target_net1 = MLP(input_dim=state_dim + action_dim, output_dim=1)
        self.target_net2 = MLP(input_dim=state_dim + action_dim, output_dim=1)
        self.target_actor_net = Actor(state_dim, action_dim, max_action)
        self.target_net1.load_state_dict(self.crit_net1.state_dict()) # initialize target net with same weights as q_net
        self.target_net2.load_state_dict(self.crit_net2.state_dict()) # initialize target net with same weights as q_net
        self.target_actor_net.load_state_dict(self.actor_net.state_dict()) # initialize target actor net with same weights as actor net
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=lr)
        self.critics_optimizer = torch.optim.Adam(list(self.crit_net1.parameters()) + list(self.crit_net2.parameters()), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crit_net1.to(self.device)
        self.crit_net2.to(self.device)
        self.actor_net.to(self.device)
        self.target_net1.to(self.device)
        self.target_net2.to(self.device)
        self.target_actor_net.to(self.device)
        self.action_low = torch.tensor(action_space.low, dtype=torch.float32).to(self.device)
        self.action_high = torch.tensor(action_space.high, dtype=torch.float32).to(self.device)
        self.eval_mode = False

    
    def act(self, state):
        """Action selection: a ~ πφ(s) + eps, eps∼ N (0, σ)"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor_net(state_tensor).squeeze(0).cpu().numpy()
        # evaluation: fully deterministic, no exploration
        if self.eval_mode:
            return action
        # training: add exploration noise
        noise = np.random.normal(0, self.sigma, action.shape)
        action = action + noise
        return np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)
    
    def update(self):
        self.update_step += 1 # time 
        # 1. Check if we have enough samples in the buffer to sample a batch
        if len(self.buffer) < self.batch_size:
            return
        # 2. Sample a minibatch of transitions
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        # 3. Compute target Q-values using the target network
        with torch.no_grad():
            # draw action a~ and y = r + gamma * min_theta1,2 Q_theta'(s', a~)
            noise = torch.normal(0, self.sigma_tilde, size=actions_tensor.shape).to(self.device)
            eps = noise.clamp(-self.c, self.c)
            a_tilde = self.target_actor_net(next_states_tensor) + eps
            a_tilde = torch.clamp(a_tilde, self.action_low, self.action_high)
            q1_target = self.target_net1(torch.cat([next_states_tensor, a_tilde], dim=1))
            q2_target = self.target_net2(torch.cat([next_states_tensor, a_tilde], dim=1))
            min_q = torch.min(q1_target, q2_target)
            y = rewards_tensor + self.gamma* (1 - dones_tensor) * min_q
        # 4. Compute the loss (MSE)
        q1_values = self.crit_net1(torch.cat([states_tensor, actions_tensor], dim=1))
        q2_values = self.crit_net2(torch.cat([states_tensor, actions_tensor], dim=1))
        loss_1 = torch.nn.functional.mse_loss(q1_values, y)
        loss_2 = torch.nn.functional.mse_loss(q2_values, y)
        # 5. Optimize to get good theta values
        self.critics_optimizer.zero_grad()
        loss = loss_1 + loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.crit_net1.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.crit_net2.parameters(), 10)
        self.critics_optimizer.step()
        loss_value = loss.item()
        # 6. update target networks every N-steps
        if self.update_step % self.policy_delay == 0:
            actor_actions = self.actor_net(states_tensor)
            actor_loss = -self.crit_net1(torch.cat([states_tensor, actor_actions], dim=1)).mean()
            # optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # soft updates
            self.soft_update(self.target_net1, self.crit_net1)
            self.soft_update(self.target_net2, self.crit_net2)
            self.soft_update(self.target_actor_net, self.actor_net)

        return loss_value
    
    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def set_eval_mode(self, mode: bool):
        self.eval_mode = mode