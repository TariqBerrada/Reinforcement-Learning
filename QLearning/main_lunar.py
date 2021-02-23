import gym
from dqn import Agent
from utils import plot_learning_curve, save_frames_as_gif
import numpy as np
import joblib

def animate():
    env = gym.make('LunarLander-v2').env
    # rec = VideoRecorder(env, 'recording.mp4')
    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = 4, eps_end = 0.01, input_dims = [8], lr = 0.003)
    agent.Q_eval.load_checkpoint()
    done = False
    count = 0
    score = 0
    observation  = env.reset()
    env.render()
    frames = []
    
    while not done:
        count +=1
        frame = env.render(mode = 'rgb_array')
        frames.append(frame)
        action = agent.choose_action(observation)
        observation, reward, done, info = env.step(action)
    joblib.dump(frames, 'frames.pkl')
    save_frames_as_gif(frames)
    env.close()

    print('game lasted %d moves.'%count)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = 4, eps_end = 0.01, input_dims = [8], lr = 0.003)
    
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        agent.Q_eval.save_checkpoint()
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('episode : %d | score : %.2f | avg_score : %.2f | epsilon : %.2f'%(i, score, avg_score, agent.epsilon))
    x = [i+1 for i in range(n_games)]
    
    filename = './lunar_lander_dqn.png'
    plot_learning_curve(x, scores, eps_history, filename)
    print('... done learning, generating animation from test episode ...')
    animate()