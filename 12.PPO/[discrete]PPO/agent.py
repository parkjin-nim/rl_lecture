# agent.py
import torch
import random

import numpy as np

from torch import nn
from collections import deque


def calc_return_seq_tensor(gamma, reward_seq_tensor):
    seq_length, n_batch = reward_seq_tensor.shape
    gamma_seq = gamma * torch.ones(reward_seq_tensor.shape)
    return_seq = torch.zeros(seq_length, n_batch)  # Initialize return sequence
    
    for t in range(seq_length):
        gamma_seq_from_t = gamma_seq[t:, :]  # (n_seq, n_batch)
        powers = torch.arange(seq_length - t).unsqueeze(-1).repeat(1, n_batch)  # (n_seq, n_batch)
        gamma_power_seq_from_t = torch.pow(gamma_seq_from_t, powers)  # (n_seq, n_batch)
        reward_seq_from_t = reward_seq_tensor[t:, :]  # (n_seq, n_batch)
        g_t = torch.sum(reward_seq_from_t * gamma_power_seq_from_t, dim=0)  # (n_batch)
        return_seq[t, :] = g_t

    return return_seq


class Agent(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.config = config

        d_state = env.observation_space.shape[0]
        n_action = env.action_space.n

        self.encoder = nn.Sequential(
            nn.Linear(d_state, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU()
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, n_action),
            nn.Softmax(dim=-1)
        )

        self.batch = deque([], maxlen=config.batch_size)  # list of episodes

    def crate_trajectory(self):
        trajectory = {
            'state': list(),
            'pi_old': list(),
            'action': list(),
            'reward': list(),
            'state_next': list(),
            'done': list(),
        }
        return trajectory

    def add_to_batch(self, s, pi_old, a, r, s_next, done):
        if (
                len(self.batch) == 0 
                or len(self.batch[-1]['state']) == self.config.seq_length
            ):
            trajectory = self.crate_trajectory()
            self.batch.append(trajectory)

        if not done:
            length_to_append = 1
        else:
            # When the trajectory is done before it is full, append the last data until the end
            length_to_append = self.config.seq_length - len(self.batch[-1]['state'])

        for _ in range(length_to_append):
            self.batch[-1]['state'].append(s)
            self.batch[-1]['pi_old'].append(pi_old)
            self.batch[-1]['action'].append(a)
            self.batch[-1]['reward'].append(r)
            self.batch[-1]['state_next'].append(s_next)
            self.batch[-1]['done'].append(done)

    def set_optimizer(self):
        self.optim = torch.optim.Adam(
            self.parameters(), 
            lr=self.config.lr
        )

    def forward(self, x):
        h_enc = self.encoder(x)
        value = self.value_head(h_enc)
        pi = self.policy_head(h_enc)
        return pi, value

    def action(self, x):
        # used when sampling an action for a state
        with torch.no_grad():
            x = torch.from_numpy(x).float().reshape(1, -1)
            pi, value = self.forward(x)
            a = torch.distributions.Categorical(pi).sample().item()
            pi = pi.numpy().squeeze(0)  # (n_action)
        return pi, a

    def train(self):
        for k in range(self.config.k_epoch):
            minibatch = random.sample(self.batch, self.config.minibatch_size)
            state_seq_array = np.array([trajectory['state'] for trajectory in minibatch])  # (n_batch, n_seq, *dim_state)
            pi_old_seq_array = np.array([trajectory['pi_old'] for trajectory in minibatch])  # (n_batch, n_seq, n_action)
            action_seq_array = np.array([trajectory['action'] for trajectory in minibatch], dtype=np.int64)  # (n_batch, n_seq)
            reward_seq_array = np.array([trajectory['reward'] for trajectory in minibatch])  # (n_batch, n_seq)
            state_next_seq_array = np.array([trajectory['state_next'] for trajectory in minibatch])  # (n_batch, n_seq, *dim_state)
            done_seq_array = np.array([trajectory['done'] for trajectory in minibatch])  # (n_batch, n_seq)

            state_seq_tensor = torch.from_numpy(
                state_seq_array
            ).float().transpose(0, 1)  # (n_seq, n_batch, *dim_states)
            pi_old_seq_tensor = torch.from_numpy(pi_old_seq_array).transpose(0, 1)  # (n_seq, n_batch, n_action)
            action_seq_tensor = torch.from_numpy(action_seq_array).transpose(0, 1)  # (n_seq, n_batch)
            reward_seq_tensor = torch.from_numpy(reward_seq_array).float().transpose(0, 1)  # (n_seq, n_batch)
            state_next_seq_tensor = torch.from_numpy(
                state_next_seq_array
            ).float().transpose(0, 1)  # (n_seq, n_batch, *dim_states)
            done_seq_tensor = torch.from_numpy(done_seq_array).float().transpose(0,1)  # (n_seq, n_batch)
            
            # mask for updating policy, until the transition that its done is True
            update_mask = done_seq_tensor.roll(1, dims=0)  # (n_seq, n_batch)
            update_mask[0, :] = 0   # (n_seq, n_batch)
            update_mask = 1 - update_mask  # (n_seq, n_batch)

            pi, value = self.forward(state_seq_tensor)  # (n_seq, n_batch, n_action), (n_seq, n_batch, 1)
            _, value_next = self.forward(state_next_seq_tensor)  # (n_seq, n_batch, 1)
            value = value.squeeze(-1)  # (n_seq, n_batch)
            value_next = value_next.squeeze(-1)  # (n_seq, n_batch)

            delta = reward_seq_tensor + self.config.gamma * (1 - done_seq_tensor) * value_next - value
            gae = calc_return_seq_tensor(self.config.lam * self.config.gamma, update_mask * delta.detach())

            pi_chosen = pi.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))  # (n_seq, n_batch, 1)
            pi_chosen = pi_chosen.squeeze(-1)  # (n_seq, n_batch)

            pi_old_chosen = pi_old_seq_tensor.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))  # (n_seq, n_batch, 1)
            pi_old_chosen = pi_old_chosen.squeeze(-1)  # (n_seq, n_batch)

            value_target = (
                reward_seq_tensor 
                + self.config.gamma * (1 - done_seq_tensor) * value_next.detach()
            )  # (n_seq, n_batch)

            loss_critic = torch.mean(update_mask * (value_target - value) ** 2)

            r = pi_chosen / pi_old_chosen
            loss_actor = -torch.mean(
                update_mask * torch.min(
                    gae * r, 
                    gae * torch.clip(r, 1 - self.config.eps_clip, 1 + self.config.eps_clip)
                )
            )
            
            loss_exp = -torch.mean(
                update_mask 
                * torch.sum(-pi * torch.log(pi + 1e-15), dim=-1)  # (n_seq, n_batch, n_action) -> (n_seq, n_batch)
            )
            loss = self.config.c1 * loss_critic + self.config.c2 * loss_actor + self.config.c3 * loss_exp

            loss_critic_avg = loss_critic * self.config.seq_length * self.config.minibatch_size / update_mask.sum()
            entropy_avg = -loss_exp * self.config.seq_length * self.config.minibatch_size / update_mask.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.batch.clear()

        return loss_critic_avg.detach().item(), entropy_avg.detach().item()






















