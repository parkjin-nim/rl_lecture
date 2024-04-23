import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
from itertools import count
import timeit
from datetime import timedelta
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import pybullet as p
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()  
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        linear_input_size = convw * convh * 64
        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)


# preprocess = T.Compose([T.ToPILImage(),
#                     T.Grayscale(num_output_channels=1),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])
preprocess = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(40, interpolation=Image.BICUBIC), #Image.CUBIC는 deprecated됨
                    T.ToTensor()])

def get_screen(env):
    global stacked_screens
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env._get_observation().transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return preprocess(screen).unsqueeze(0).to(device)

if __name__ == '__main__':
    PATH = 'policy_dqn.pt'
    STACK_SIZE = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    env = KukaDiverseObjectEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20, isTest=True)
    #env = KukaDiverseObjectEnv(renders=True, isDiscrete=False)
    env.cid = p.connect(p.DIRECT)
    env.reset()

    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)

    episode = 10
    scores_window = collections.deque(maxlen=100)  # last 100 scores


    # load the model
    checkpoint = torch.load(PATH)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    #evaluate the model
    for i_episode in range(episode):
        env.reset()
        state = get_screen(env)
        stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)
        for t in count():
            stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
            # Select and perform an action
            # action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
            action = policy_net(stacked_states_t)
            #print(action)
            action = action.max(1)[1]
            action = action.view(1, 1)
            #print(t, action.item())
    
            _, reward, done, _ = env.step(action.item())

            # Observe new state
            next_state = get_screen(env)
            stacked_states.append(next_state)
            if done:
                break
        print("Episode: {0:d}, reward: {1}".format(i_episode+1, reward), end="\n")

    env.close()