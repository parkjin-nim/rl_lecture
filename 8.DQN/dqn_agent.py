# dqn_agent.py
import gym
import torch
import random
import datetime

import numpy as np

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils import create_env
from configuration import config
from collections import deque


class ReplayMemory:
    def __init__(self, config):
        self.config = config
        self.buffer = deque([], maxlen=self.config.replay_capacity)

    def getsize(self):
        return len(self.buffer)

    def append(self, transition):
        buffer_size = len(self.buffer)
        self.buffer.append(transition)

    def sample(self, size):
        buffer_size = len(self.buffer)
        if buffer_size >= size:
            samples = random.sample(self.buffer, size)
        else:
            assert False, f"Buffer size ({buffer_size}) is smaller than the sample size ({size})"

        return samples


class DQNAgent(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.config = config
        self.replay_memory = ReplayMemory(self.config)

        d_state = env.observation_space.shape[0]
        n_action = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(d_state, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, n_action)
        )

        self.target_network = nn.Sequential(
            nn.Linear(d_state, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, n_action)
        )

        for param in self.target_network.parameters():
            param.requires_grad = False

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            params=self.network.parameters(),
            lr=self.config.lr, 
            weight_decay=1e-3
        )

    def forward(self, x):
        Qs = self.network(x)
        return Qs

    def forward_target_network(self, x):
        Qs = self.target_network(x)
        return Qs

    def get_argmax_action(self, x):
        s = torch.from_numpy(x).reshape(1, -1).float()
        Qs = self.forward(s)
        argmax_action = Qs.argmax(dim=-1).item()
        return argmax_action

    def train(self):
        transitions = self.replay_memory.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_array = np.stack(states, axis=0)  # (n_batch, d_state)
        actions_array = np.stack(actions, axis=0, dtype=np.int64)  # (n_batch)
        rewards_array = np.stack(rewards, axis=0)  # (n_batch)
        next_states_array = np.stack(next_states, axis=0)  # (n_batch, d_state)
        dones_array = np.stack(dones, axis=0)  # (n_batch)

        states_tensor = torch.from_numpy(states_array).float()  # (n_batch, d_state)
        actions_tensor = torch.from_numpy(actions_array)  # (n_batch)
        rewards_tensor = torch.from_numpy(rewards_array).float()  # (n_batch)
        next_states_tensor = torch.from_numpy(next_states_array).float()  # (n_batch, d_state)
        dones_tensor = torch.from_numpy(dones_array).float()  # (n_batch)

        Qs = self.forward(states_tensor)  # (n_batch, n_action)
        next_Qs = self.forward_target_network(next_states_tensor)  # (n_batch, n_action)

        # index dimension should be the same as the source tensor
        chosen_Q = Qs.gather(dim=-1, index=actions_tensor.reshape(-1, 1)).reshape(-1)
        target_Q = rewards_tensor + (1 - dones_tensor) * config.gamma * next_Qs.max(dim=-1).values
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(chosen_Q, target_Q)

        # Update by gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def get_eps(config, step):
    eps_init = config.eps_init
    eps_final = config.eps_final
    if step >= config.eps_decrease_step:
        eps = eps_final
    else:
        m = (eps_final - eps_init) / config.eps_decrease_step
        eps = eps_init + m * step
    return eps


def eval_agent(config, env, agent):
    score_sum = 0
    step_count_sum = 0
    for _ in range(config.num_eval_episode):
        s = env.reset()
        step_count = 0
        done = False
        score = 0
        while not done:
            with torch.no_grad():
                a = agent.get_argmax_action(s)

            s_next, r, done, info = env.step(a)
            step_count += 1
            
            score += r
            s = s_next
        
        score_sum += score
        step_count_sum += step_count

    score_avg = score_sum / config.num_eval_episode
    step_count_avg = step_count_sum / config.num_eval_episode
    return score_avg, step_count_avg


if __name__ == "__main__":
    env = create_env(config)
    env_eval = create_env(config)
    agent = DQNAgent(env, config)
    agent.set_optimizer()
    
    dt_now = datetime.datetime.now()
    logdir = f"logdir/{dt_now.strftime('%y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(logdir)

    # Reset Replay Buffer
    init_replay_buffer_size = int(config.replay_init_ratio * config.replay_capacity)
    s = env.reset()
    step_count = 0
    for _ in range(init_replay_buffer_size):
        a = np.random.choice(env.action_space.n)  # uniform random action
        s_next, r, done, info = env.step(a)
        step_count += 1

        transition = (s, a, r, s_next, done)
        agent.replay_memory.append(transition)

        s = s_next
        if done:
            s = env.reset()
            step_count = 0

    # Train agent
    s = env.reset()
    step_count = 0
    for step_train in range(config.train_env_steps):
        eps = get_eps(config, step_train)
        is_random_action = np.random.choice(2, p=[1 - eps, eps])
        if is_random_action:
            a = np.random.choice(env.action_space.n)  # uniform random action
        else:
            a = agent.get_argmax_action(s)
        
        s_next, r, done, info = env.step(a)
        step_count += 1

        transition = (s, a, r, s_next, done)
        agent.replay_memory.append(transition)

        s = s_next
        if done:
            s = env.reset()
            step_count = 0

        if step_train % config.target_update_period == 0:
            agent.update_target_network()

        if step_train % 4 == 0:
            loss = agent.train()

        if step_train % config.eval_period == 0:
            score_avg, step_count_avg = eval_agent(config, env_eval, agent)
            print(
                f"[{step_train}] eps: {eps:.3f} loss: {loss:.3f} "
                + f"score_avg: {score_avg:.3f} step_count_avg: {step_count_avg:.3f}"
            )
            writer.add_scalar("Train/loss", loss, step_train)
            writer.add_scalar("Eval/score_avg", score_avg, step_train)
            writer.add_scalar("Eval/step_count_avg", step_count_avg, step_train)

    torch.save(agent.state_dict(), f"{logdir}/state_dict.pth")






















