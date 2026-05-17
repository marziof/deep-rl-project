# Implementation of PPO algo (optimized version)
from turtle import done, mode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from src.networks.mlp import MLP


class PPOAgent:
    def __init__(self, action_space, state_dim, hidden_dim, gamma, lr, n_actors, time_per_actor, n_epochs, batch_size, epsilon_clip = 0.2):
        self.action_space = action_space
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actors = n_actors
        self.time_per_actor = time_per_actor
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eps = epsilon_clip
        self.gamma = gamma
        self.lr = lr
        self.lambda_param = 0.95
        self.actor_net = MLP(input_dim=state_dim, output_dim=action_space.n, hidden_dim=hidden_dim)
        self.critic_net = MLP(input_dim=state_dim, output_dim=1, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(list(self.actor_net.parameters()) + list(self.critic_net.parameters()), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.entropy_coef = 0.01

        self.states       = np.zeros((n_actors, time_per_actor, state_dim))
        self.next_states  = np.zeros((n_actors, time_per_actor, state_dim))
        self.actions      = np.zeros((n_actors, time_per_actor))
        self.rewards      = np.zeros((n_actors, time_per_actor))
        self.log_probs    = np.zeros((n_actors, time_per_actor))
        self.dones        = np.zeros((n_actors, time_per_actor))
        self.advantages   = np.zeros((n_actors, time_per_actor))
        self.value_targets = np.zeros((n_actors, time_per_actor))
        self.t = 0

    def act(self, states):
        "Action selection using the actor network, returns an action index based on the current state"
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.actor_net(states_tensor) # probability dist of actions from the actor network
            action_distribution = Categorical(logits=logits) # create a categorical distribution from the logits
            chosen_actions =  action_distribution.sample() 
            log_probs = action_distribution.log_prob(chosen_actions) # compute the log probability of the chosen actions
        return chosen_actions.cpu().numpy(), log_probs

    def store(self, t, states, actions, rewards, next_states, log_probs, dones):
        self.states[:, t]       = states
        self.next_states[:, t]  = next_states
        self.actions[:, t]      = actions
        self.rewards[:, t]      = rewards
        self.log_probs[:, t]    = log_probs
        self.dones[:, t]        = dones

    def calculate_advantages(self):
        '''Function that estimates the advantages and the value targets for the optimization phase of PPO'''

        #Calculate TD residuals
        TD_residuals = np.zeros((self.n_actors, self.time_per_actor))

        states_flat      = self.states.reshape(-1, self.state_dim)
        next_states_flat = self.next_states.reshape(-1, self.state_dim)

        states_tensor = torch.tensor(states_flat, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(next_states_flat, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            values = self.critic_net(states_tensor).squeeze(1).cpu().numpy()
            next_values = self.critic_net(next_states_tensor).squeeze(1).cpu().numpy()
        
        values      = values.reshape(self.n_actors, self.time_per_actor)
        next_values = next_values.reshape(self.n_actors, self.time_per_actor)
        next_values[self.dones.astype(bool)] = 0.0

        TD_residuals = (self.rewards + self.gamma * next_values - values)
        
        #Calculate Advantages
        advantage = np.zeros(self.n_actors)
        for j in reversed(range(self.time_per_actor)):
            advantage = TD_residuals[:, j] + self.gamma * self.lambda_param * advantage * (1 - self.dones[:, j])
            self.advantages[:, j] = advantage
        
        #Calculate Value targets
        self.value_targets = self.advantages + values
        self.advantages    = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def update(self):
        total_steps = self.n_actors * self.time_per_actor

        states_tensor        = torch.tensor(self.states.reshape(total_steps, self.state_dim), dtype=torch.float32).to(self.device)
        actions_tensor       = torch.tensor(self.actions.reshape(total_steps), dtype=torch.int64).unsqueeze(1).to(self.device)
        old_log_probs_tensor = torch.tensor(self.log_probs.reshape(total_steps), dtype=torch.float32).unsqueeze(1).to(self.device)
        advantages_tensor    = torch.tensor(self.advantages.reshape(total_steps), dtype=torch.float32).unsqueeze(1).to(self.device)
        value_targets_tensor = torch.tensor(self.value_targets.reshape(total_steps), dtype=torch.float32).unsqueeze(1).to(self.device)

        total_loss_values = []

        for _ in range(self.n_epochs):
            indices = torch.randperm(total_steps) 
            # Update actor network
            for start in range(0, len(indices), self.batch_size):
                #print(f"Updating PPO, epoch {_+1}/{self.n_epochs}, batch starting at index {start}...")
                end = start + self.batch_size
                batch_indices        = indices[start:end]
                batch_states         = states_tensor[batch_indices]
                batch_actions        = actions_tensor[batch_indices]
                batch_old_log_probs  = old_log_probs_tensor[batch_indices]
                batch_advantages     = advantages_tensor[batch_indices]
                batch_value_targets  = value_targets_tensor[batch_indices]

                logits = self.actor_net(batch_states)
                action_distribution = Categorical(logits=logits)
                new_log_probs = action_distribution.log_prob(batch_actions.squeeze()) #Get pi_theta(a|s) for the actions taken in the trajectories, using the current policy (actor network)
                entropy = action_distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs.squeeze()) #Get the ratio pi_theta(a|s) / pi_theta_old(a|s) using the log probabilities
                surrogate_loss_1 = ratio * batch_advantages.squeeze()
                surrogate_loss_2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantages.squeeze()
                actor_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()

                # Update critic network
                values = self.critic_net(batch_states)
                critic_loss = nn.MSELoss()(values, batch_value_targets)

                # Backpropagation
                total_loss = actor_loss + critic_loss - self.entropy_coef * entropy # Add an entropy bonus to encourage exploration
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                total_loss_value = total_loss.item()
                total_loss_values.append(total_loss_value)

        return np.mean(total_loss_values)

class PPOAgentContinuous:
    def __init__(self, action_space, state_dim, hidden_dim, gamma, lr, n_actors, time_per_actor, n_epochs, batch_size, epsilon_clip = 0.2):
        self.action_space = action_space
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actors = n_actors
        self.time_per_actor = time_per_actor
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eps = epsilon_clip
        self.gamma = gamma
        self.lr = lr
        self.lambda_param = 0.95
        self.entropy_coef = 0.01
        
        self.action_dim = action_space.shape[0]
        self.actor_net = MLP(input_dim=state_dim, output_dim=self.action_dim, hidden_dim=hidden_dim)
        self.critic_net = MLP(input_dim=state_dim, output_dim=1, hidden_dim=hidden_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.optimizer = optim.Adam(list(self.actor_net.parameters()) + list(self.critic_net.parameters()) + [self.log_std], lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.states       = np.zeros((n_actors, time_per_actor, state_dim))
        self.next_states  = np.zeros((n_actors, time_per_actor, state_dim))
        self.actions      = np.zeros((n_actors, time_per_actor, self.action_dim))
        self.rewards      = np.zeros((n_actors, time_per_actor))
        self.log_probs    = np.zeros((n_actors, time_per_actor))
        self.dones        = np.zeros((n_actors, time_per_actor))
        self.advantages   = np.zeros((n_actors, time_per_actor))
        self.value_targets = np.zeros((n_actors, time_per_actor))
        self.t = 0

    def act(self, states):
        "Action selection using the actor network, returns an action index based on the current state"
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mean = self.actor_net(states_tensor) # probability dist of actions from the actor network
            std = torch.exp(self.log_std).expand_as(mean)
            action_distribution = Normal(mean, std) # create a normal distribution from the mean and std
            chosen_actions =  action_distribution.sample() 
            log_probs = action_distribution.log_prob(chosen_actions).sum(dim=-1) # compute the log probability of the chosen actions

            low  = self.action_space.low
            high = self.action_space.high
            actions_np = np.clip(chosen_actions.cpu().numpy(), low, high)
        return actions_np, log_probs

    def store(self, t, states, actions, rewards, next_states, log_probs, dones):
        self.states[:, t]       = states
        self.next_states[:, t]  = next_states
        self.actions[:, t]      = actions
        self.rewards[:, t]      = rewards
        self.log_probs[:, t]    = log_probs
        self.dones[:, t]        = dones

    def calculate_advantages(self):
        '''Function that estimates the advantages and the value targets for the optimization phase of PPO'''

        #Calculate TD residuals
        TD_residuals = np.zeros((self.n_actors, self.time_per_actor))

        states_flat      = self.states.reshape(-1, self.state_dim)
        next_states_flat = self.next_states.reshape(-1, self.state_dim)

        states_tensor = torch.tensor(states_flat, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(next_states_flat, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            values = self.critic_net(states_tensor).squeeze(1).cpu().numpy()
            next_values = self.critic_net(next_states_tensor).squeeze(1).cpu().numpy()
        
        values      = values.reshape(self.n_actors, self.time_per_actor)
        next_values = next_values.reshape(self.n_actors, self.time_per_actor)
        next_values[self.dones.astype(bool)] = 0.0

        TD_residuals = (self.rewards + self.gamma * next_values - values)
        
        #Calculate Advantages
        advantage = np.zeros(self.n_actors)
        for j in reversed(range(self.time_per_actor)):
            advantage = TD_residuals[:, j] + self.gamma * self.lambda_param * advantage * (1 - self.dones[:, j])
            self.advantages[:, j] = advantage
        
        #Calculate Value targets
        self.value_targets = self.advantages + values
        self.advantages    = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def update(self):
        total_steps = self.n_actors * self.time_per_actor

        states_tensor        = torch.tensor(self.states.reshape(total_steps, self.state_dim), dtype=torch.float32).to(self.device)
        actions_tensor       = torch.tensor(self.actions.reshape(total_steps, self.action_dim), dtype=torch.float32).to(self.device)
        old_log_probs_tensor = torch.tensor(self.log_probs.reshape(total_steps), dtype=torch.float32).unsqueeze(1).to(self.device)
        advantages_tensor    = torch.tensor(self.advantages.reshape(total_steps), dtype=torch.float32).unsqueeze(1).to(self.device)
        value_targets_tensor = torch.tensor(self.value_targets.reshape(total_steps), dtype=torch.float32).unsqueeze(1).to(self.device)

        total_loss_values = []

        for _ in range(self.n_epochs):
            indices = torch.randperm(total_steps) 
            # Update actor network
            for start in range(0, len(indices), self.batch_size):
                #print(f"Updating PPO, epoch {_+1}/{self.n_epochs}, batch starting at index {start}...")
                end = start + self.batch_size
                batch_indices        = indices[start:end]
                batch_states         = states_tensor[batch_indices]
                batch_actions        = actions_tensor[batch_indices]
                batch_old_log_probs  = old_log_probs_tensor[batch_indices]
                batch_advantages     = advantages_tensor[batch_indices]
                batch_value_targets  = value_targets_tensor[batch_indices]

                mean = self.actor_net(batch_states)
                std = torch.exp(self.log_std).expand_as(mean)
                action_distribution = Normal(mean, std)
                new_log_probs = action_distribution.log_prob(batch_actions).sum(dim=-1) #Get pi_theta(a|s) for the actions taken in the trajectories, using the current policy (actor network)
                entropy = action_distribution.entropy().sum(dim=-1).mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs.squeeze()) #Get the ratio pi_theta(a|s) / pi_theta_old(a|s) using the log probabilities
                surrogate_loss_1 = ratio * batch_advantages.squeeze()
                surrogate_loss_2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantages.squeeze()
                actor_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()

                # Update critic network
                
                values = self.critic_net(batch_states)
                critic_loss = nn.MSELoss()(values, batch_value_targets)

                # Backpropagation
                total_loss = actor_loss + critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                total_loss_value = total_loss.item()
                total_loss_values.append(total_loss_value)

        return np.mean(total_loss_values)
        

