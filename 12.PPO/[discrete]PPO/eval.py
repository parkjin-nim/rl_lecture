# eval.py
import gym
import time
import torch
import argparse

from utils import create_env
from agent import Agent
from configuration import config


def eval_agent_with_rendering(config, env, agent):
    score_sum = 0
    step_count_sum = 0
    for _ in range(config.num_eval_episode):
        s = env.reset()
        env.render()
        step_count = 0
        done = False
        score = 0
        while not done:
            _, a = agent.action(s)

            s_next, r, done, info = env.step(a)
            env.render()
            step_count += 1
            
            score += r
            s = s_next
        
        score_sum += score
        step_count_sum += step_count

    score_avg = score_sum / config.num_eval_episode
    step_count_avg = step_count_sum / config.num_eval_episode

    print(f"score_avg: {score_avg} step_count_avg: {step_count_avg}")
    return score_avg, step_count_avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval', type=int, default=1,
                        help='Number of episode to be evaluated')
    parser.add_argument('--model_path', type=str,
                        help='Specify the path to your model to be evaluated')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config.num_eval_episode = args.num_eval
    env = create_env(config)
    agent = Agent(env, config)
    
    if args.model_path:
        state_dict = torch.load(args.model_path)
        agent.load_state_dict(state_dict)

    eval_agent_with_rendering(config, env, agent)



