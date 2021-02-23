import gym, joblib, tqdm
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import animation

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
env = gym.make('CartPole-v0').env
# env = gym.wrappers.Monitor(env_to_wrap, './render', force = True)
print('There are %d actions possible.'%env.action_space.n)

class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = torch.nn.Linear(4, 128) # 4 input parameters
        self.fc2 = torch.nn.Linear(128, 128)
        self.actor = torch.nn.Linear(128, 2)
        self.critic = torch.nn.Linear(128, 1)
        self.saved_actions = []
        self.rewards = []
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        action_prob = F.softmax(self.actor(x), dim = -1)
        state_values = self.critic(x)
        return action_prob, state_values
         
def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + 0.99*R
        returns.insert(0, R)
    returns = np.array(returns)
    returns = (returns - returns.mean())/returns.std()

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob*advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

model = ActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr = 4e-3)
eps = np.finfo(np.float32).eps.item()

def train():
    running_reward = 0
    ep_rewards = []
    rn_rewards = []
    for i_episode in tqdm.tqdm(range(500)):
        state = env.reset()
        ep_reward = 0
        
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, info = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        running_reward = 0.05*ep_reward + (1-0.05)*running_reward
        ep_rewards.append(ep_reward)
        rn_rewards.append(running_reward)
        finish_episode()
        if i_episode%10 == 0:
            print('Episode % d | Episode Reward %.4f | Running Reward %.4f'%(i_episode, ep_reward, running_reward))
        if i_episode%50 == 0:
            joblib.dump(model.state_dict(), 'weigths.pth.tar')
        
        if running_reward > 200:
            print('done', env.spec.reward_threshold)
            print('Episode % d | Episode Reward %.4f | Running Reward %.4f'%(i_episode, ep_reward, running_reward))
            break
    print('done, saving weights.')
    joblib.dump(model.state_dict(), 'weigths.pth.tar')
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(ep_rewards, label = 'episode rewards')
    ax2 = fig.add_subplot(212)
    ax2.plot(rn_rewards, label = 'running rewards')
    plt.savefig('losses.png')

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 32.0, frames[0].shape[0] / 32.0), dpi=32)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=.01)
    anim.save(path + filename, writer='pillow', fps=360)

def animate():
    env = gym.make('CartPole-v0').env
    # rec = VideoRecorder(env, 'recording.mp4')
    done = False
    count = 0
    observation  = env.reset()
    env.render()
    frames = []
    while not done:
        count += 1
        frame = env.render(mode = 'rgb_array')
        frames.append(frame)
        # rec.capture_frame()
        action = select_action(observation)
        observation, reward, done, info = env.step(action)
    joblib.dump(frames, 'frames.pkl')
    save_frames_as_gif(frames)
    env.close()
    
    print('game lasted %d moves.'%count)

if __name__ == '__main__':
    train()
    animate()