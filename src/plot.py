import numpy as np
import matplotlib.pyplot as plt


def setup_plot(shape, scale=0.75):
    figure = plt.figure(figsize=(scale*shape[1], scale*shape[0]))
    ax = figure.add_subplot()
    ax.set_autoscaley_on(True)
    ax.set_autoscalex_on(True)
    ax.set_xlim(-1, shape[0] + 1)
    ax.set_ylim(-1, shape[1] + 1)
    return ax


def plot_values_and_policy(learner):
    value = np.sum(learner.quality * learner.policy, axis=-1)
    vmin = np.min(value) - 1 # Leave space 1 for obstacles
    vmax = np.max(value)
    plt.cla()
    # # ax.axis('off')

    scale = 0.5
    for state in learner.env.get_all_states():
        i, j = state
        if not learner.env.is_the_new_state_allowed(state):
            value[i, j] = vmin  # Set obstacles
        else:
            # Generate arrows for policy
            for action in range(learner.env.action_space.n):
                vx, vy = scale * learner.env.action_dict[action] * learner.policy[i, j][action]
                plt.arrow(j, i, vy, vx, head_width=0.1, color='black', alpha=0.5)

    plt.matshow(value, fignum=0, vmax=vmax, vmin=vmin)
    plt.draw()
    plt.show()
    plt.pause(0.1)
    
def plot_simulation(learner):
    learner.env.reset(learner.env.start_state)
    terminated = False
    while not terminated:
        action_id = learner.choose_action(learner.env.state)
        old_state = learner.env.state
        new_state, reward, terminated, info = learner.env.step(action_id)
        if np.array_equal(old_state, new_state):
            plt.scatter(old_state[1], old_state[0], c='red', s=120)
        else:
            plt.scatter(old_state[1], old_state[0], c='gray', s=120)
            plt.scatter(new_state[1], new_state[0], c='orange', s=120)
        plt.pause(0.05)