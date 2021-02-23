import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

def plot_learning_curve(x, scores, epsilons, filename, lines = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label = '1')
    ax2 = fig.add_subplot(111, label = '2', frame_on = False)

    ax.plot(x, epsilons, color = 'C0')
    ax.set_xlabel('training steps', color = 'C0')
    ax.set_ylabel('epsilon', color = 'C0')
    ax.tick_params(axis = 'x', colors = 'C0')
    ax.tick_params(axis = 'y', colors = 'C0')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
    ax2.scatter(x, running_avg, color = 'C1')
    # ax2.axes.get_axis().set_visible(False)
    ax2.yaxis.tick_right()
    plt.savefig(filename)

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 32.0, frames[0].shape[0] / 32.0), dpi=32)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 1)
    anim.save(path + filename, writer='pillow', fps=None)
