# train.py
import torch
import datetime

import numpy as np

from utils import create_env
from agent import Agent
from configuration import config
from collections import deque
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    env_list = [create_env(config) for _ in range(config.num_env)]
    agent = Agent(env_list[0], config)
    agent.set_optimizer()
    assert config.batch_size % config.num_env == 0

    dt_now = datetime.datetime.now()
    logdir = f"logdir/{dt_now.strftime('%y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(logdir)

    score_que = deque([], maxlen=config.num_eval_episode)
    count_step_que = deque([], maxlen=config.num_eval_episode)

    score = 0
    count_step = 0
    s_list = [env.reset() for env in env_list]
    score_list = [0 for env in env_list]
    count_step_list = [0 for env in env_list]

    num_iteration = int(config.train_env_steps / config.num_env / config.seq_length)
    for step_iteration in range(num_iteration):
        for i_env in range(config.num_env):
            env = env_list[i_env]
            s = s_list[i_env]
            for _ in range(config.seq_length):
                pi, a = agent.action(s)
                s_next, r, done, info = env.step(a)
                agent.add_to_batch(s, pi, a, r, s_next, done)
                s = s_next
                score_list[i_env] += r
                count_step_list[i_env] += 1

                s_list[i_env] = s
                if done:
                    s = env.reset()
                    s_list[i_env] = s

                    score_que.append(score_list[i_env])
                    count_step_que.append(count_step_list[i_env])

                    score_list[i_env] = 0
                    count_step_list[i_env] = 0

                    break
         
        if len(agent.batch) == config.batch_size:
            loss_critic_avg, entropy_avg = agent.train()
            writer.add_scalar('Train/loss_critic', loss_critic_avg, step_iteration)
            writer.add_scalar('Train/entropy', entropy_avg, step_iteration)

        if len(score_que) == config.num_eval_episode:
            score_avg = np.mean(score_que)
            count_step_avg = np.mean(count_step_que)
            writer.add_scalar('Env/score_avg', score_avg, step_iteration)
            writer.add_scalar('Env/count_step_avg', count_step_avg, step_iteration)
            
            print(
                f"[{step_iteration}] score_avg: {score_avg:.3f} "
                f"count_step_avg: {count_step_avg:.3f} "
                f"loss_critic_avg: {loss_critic_avg:.3f} "
                f"entropy_avg: {entropy_avg:.3f} "
            )
            score_que.clear()
            count_step_que.clear()

    torch.save(agent.state_dict(), f"{logdir}/state_dict.pth")
    

