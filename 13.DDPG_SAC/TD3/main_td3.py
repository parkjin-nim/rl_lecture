import gym
import numpy as np
from td3_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    #env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.001, beta=0.001, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=100, layer1_size=400, layer2_size=300,
                n_actions=env.action_space.shape[0])
    n_games = 1500
    filename = 'Walker2d_' + str(n_games) + '_2.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    #agent.load_models()
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
        n_games = 10

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            else:
                env.render(mode='human')
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score)
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    env.close()