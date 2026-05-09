# Implementation of SAC algo

import random
from turtle import done, mode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.networks.mlp import MLP
from src.buffers.replay_buffer import ReplayBuffer

class SACAgent:
    def __init__(self, action_space, state_dim, gamma, batch_size, buffer_capacity, target_update_freq, tau=0.005, lr=3e-4, gradient_steps=1, alpha=0.2):
        # Algorithm parameters
        self.action_space = action_space
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha
        #self.lambda_v = lambda_v
        #self.lambda_q = lambda_q
        #self.lambda_pi = lambda_pi
        self.lr = lr
        self.gamma = gamma
        self.update_step = 0
        self.target_update_freq = target_update_freq
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
            self.is_discrete = True
        else:
            self.action_dim = action_space.shape[0]
            self.is_discrete = False
        
        # Defining 5 nets : 4 for architecture and 1 for target
        self.actor_net = MLP(input_dim=state_dim, output_dim=self.action_dim * 2)
        self.q1_net = MLP(input_dim=state_dim + self.action_dim, output_dim=1)
        self.q2_net = MLP(input_dim=state_dim + self.action_dim, output_dim=1)
        self.v_net = MLP(input_dim=state_dim, output_dim=1)
        self.v_target_net = MLP(input_dim=state_dim, output_dim=1)

        self.v_target_net.load_state_dict(self.v_net.state_dict())

        # Optimizers
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=3e-4)
        self.q_optimizer = optim.Adam(list(self.q1_net.parameters()) + 
                                      list(self.q2_net.parameters()), lr=3e-4)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=3e-4)

        # Others
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_mode = False
        self.buffer = ReplayBuffer(buffer_capacity)

    def update(self):
        self.update_step += 1
        # 1. Check that we have enough samples in the buffer to sample a batch
        if len(self.buffer) < self.batch_size:
            return
        
        # 2. Sample an action 
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(states).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        action_batch = torch.FloatTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Value network update 
        value_pred = self.v_net(state_batch)

        with torch.no_grad():
            new_actions, log_probs = self.sample_action_and_log_prob(state_batch)
            q1_val = self.q1_net(torch.cat([state_batch, new_actions], dim=-1))
            q2_val = self.q2_net(torch.cat([state_batch, new_actions], dim=-1))
            min_q = torch.min(q1_val, q2_val)
            target_v = min_q - (self.alpha * log_probs)
        v_loss = 0.5 * nn.MSELoss()(value_pred, target_v)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Q-values update
        with torch.no_grad():
            target_q = reward_batch + (1 - done_batch) * self.gamma * self.v_target_net(next_state_batch)
        current_q1 = self.q1_net(torch.cat([state_batch, action_batch], dim=-1))
        current_q2 = self.q2_net(torch.cat([state_batch, action_batch], dim=-1))
        
        q_loss = 0.5 * (nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q))

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Actor update 
        new_actions, log_probs = self.sample_action_and_log_prob(state_batch)
        q1_new = self.q1_net(torch.cat([state_batch, new_actions], dim=-1))
        q2_new = self.q2_net(torch.cat([state_batch, new_actions], dim=-1))
        min_q_new = torch.min(q1_new, q2_new)

        # Policy loss: alpha * log_pi - Q
        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update_target()

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item()
    }

    def soft_update_target(self):
        # This is the EMA update for the Target Value Network (Psi-bar)
        # We do NOT use an optimizer here.
        with torch.no_grad():
            # Loop through both networks' parameters simultaneously
            for target_param, param in zip(self.v_target_net.parameters(), self.v_net.parameters()):
                # Apply the formula: target = tau * current + (1 - tau) * target
                new_weight = self.tau * param.data + (1.0 - self.tau) * target_param.data
                target_param.data.copy_(new_weight)    

        

    def store(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def set_eval_mode(self, mode: bool):
        self.eval_mode = mode
    
    def act(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.actor_net(state)
            mu, log_std = torch.chunk(output, 2, dim=-1)
            std = log_std.clamp(-20,2).exp()
            if self.eval_mode:
                action = torch.tanh(mu)
            else:
                pi = torch.distributions.Normal(mu, std)
                u = pi.sample()
                action = torch.tanh(u)
        action = action.cpu().numpy().reshape(-1)
        if not self.is_discrete:
            return action * self.action_space.high[0]
        return action
    
    def sample_action_and_log_prob(self, state):
        # Variant of act to deliver useful variables
        output = self.actor_net(state)
        mu, log_std = torch.chunk(output, 2, dim=-1)
        
        # (Prevents std from being 0 or too large, which crashes math)
        std = log_std.clamp(-20, 2).exp() 
        

        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample() 
        
        action = torch.tanh(u)
        
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        
        return action, log_prob.sum(dim=-1, keepdim=True)
    

    




    


    