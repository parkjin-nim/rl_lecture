# utils.py
import gym
import numpy as np


def create_env(config):
    env = gym.make("CartPole-v1")
    env = RewardShapingWrapper(env)
    env = TimeStepAppendWrapper(env)
    
    return env


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        r = 0.1 * r
        return s_next, r, done, info


class TimeStepAppendWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.observation_space = gym.spaces.Box(
            np.concatenate([self.env.observation_space.low, [0]], axis=-1),
            np.concatenate([self.env.observation_space.high, [5]], axis=-1),
            (self.env.observation_space.shape[0] + 1, ),
            np.float32
        )

    def reset(self):
        self.step_count = 0
        s = self.env.reset()
        s = np.concatenate([s, [0.01 * self.step_count]], axis=-1)
        return s

    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        self.step_count += 1
        s_next = np.concatenate([s_next, [0.01 * self.step_count]], axis=-1)
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

