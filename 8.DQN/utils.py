# utils.py
import gym

import numpy as np

from collections import deque


def create_env(config, render_mode='rgb_array'):
    env = gym.make("Pong-ram-v0", render_mode=render_mode)
    env = ObsNormalizationWrapper(env)
    env = RepeatedActionWrapper(env, config.action_repeat)
    return env


class RepeatedActionWrapper(gym.Wrapper):
    def __init__(self, env, n_repeat):
        self.env = env
        self.n_repeat = n_repeat
        self.recent_states = deque([], maxlen=4)
        self.observation_space = gym.spaces.Box(
            self.env.observation_space.low.repeat(self.n_repeat, axis=-1), 
            self.env.observation_space.high.repeat(self.n_repeat, axis=-1), 
            (self.env.observation_space.shape[0] * self.n_repeat,), 
            np.uint8
        )

    def step(self, action):
        r_sum = 0
        done = False
        for _ in range(self.n_repeat):
            s_next, r, done, info = self.env.step(action)
            self.recent_states.append(s_next)
            r_sum += r
            
            if r != 0:
                done = True
            
            if done:
                break
        
        s_next = np.concatenate(self.recent_states, axis=0)
        return s_next, r_sum, done, info

    def reset(self):
        s = self.env.reset()
        for _ in range(20):
            s, _, _, _ = self.env.step(0)  # No action

        for _ in range(self.n_repeat):
            self.recent_states.append(s)

        s = np.concatenate(self.recent_states, axis=0)
        return s


class ObsNormalizationWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.obs_subtration = 128
        self.obs_division = 128
        self.observation_space = gym.spaces.Box(
            (self.env.observation_space.low - self.obs_subtration) / self.obs_division, 
            (self.env.observation_space.high - self.obs_subtration) / self.obs_division, 
            self.env.observation_space.shape, 
            np.float32
        )

    def reset(self):
        s = self.env.reset()
        s = (s - self.obs_subtration) / self.obs_division
        return s

    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        s_next = (s_next - self.obs_subtration) / self.obs_division
        return s_next, r, done, info


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    @classmethod
    def from_nested_dics(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key:cls.from_nested_dics(data[key]) for key in data})



