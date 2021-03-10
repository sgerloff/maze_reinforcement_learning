import numpy as np
import matplotlib.pyplot as plt
import copy, os

from src.utility import get_project_dir

# Set custom color map, indicating obstacles and goal with "under" and "over" color, respectively.
value_colormap = copy.copy(plt.cm.get_cmap("viridis"))
value_colormap.set_under("#162838")  # Fancy dark grey
value_colormap.set_over("#EB4D4D")  # Nice Red


def setup_plot(shape, scale=0.75):
    figure = plt.figure(figsize=(scale * shape[1], scale * shape[0]))
    ax = figure.add_subplot()
    ax.set_autoscaley_on(True)
    ax.set_autoscalex_on(True)
    ax.set_xlim(-1, shape[0] + 1)
    ax.set_ylim(-1, shape[1] + 1)
    return ax


def plot_values_and_policy(learner, episode_id, record=False):
    plt.cla()
    plt.axis("off")
    value, vmin, vmax = setup_values_and_arrows(learner, plt)

    plt.matshow(value, fignum=0, vmax=vmax, vmin=vmin, cmap=value_colormap)

    plt.draw()
    plt.show()

    if record:
        save_figure(learner, plt, episode_id)
    else:
        plt.pause(0.1)


def setup_values_and_arrows(learner, plt, scale=0.5):
    value = np.sum(learner.quality * learner.policy, axis=-1)
    vmin = np.min(value)
    vmax = np.max(value)

    for state in learner.env.get_all_states():
        i, j = state
        if not learner.env.is_the_new_state_allowed(state):
            value[i, j] = vmin - 1  # Set obstacles
        else:
            # Generate arrows for policy
            for action in range(learner.env.action_space.n):
                vx, vy = scale * learner.env.action_dict[action] * learner.policy[i, j][action]
                plt.arrow(j, i, vy, vx, head_width=0.1, color='black', alpha=0.5)

    value = set_goal_value(learner, value, vmax)
    return value, vmin, vmax


def set_goal_value(learner, value, vmax):
    i, j = learner.env.goal_state
    value[i, j] = vmax + 1
    return value


def save_figure(learner, plt, episode_id):
    value_dir = get_records_folder(learner)
    if not os.path.exists(value_dir):
        os.makedirs(value_dir)
    plt.savefig(os.path.join(value_dir, f"value_map_t_{episode_id}.png"))


def plot_simulation(learner, episode_id, record=False):
    learner.env.reset(learner.env.start_state)
    terminated = False
    time_step = 0
    while not terminated:
        action_id = learner.choose_action(learner.env.state)
        old_state = learner.env.state
        new_state, reward, terminated, info = learner.env.step(action_id)
        if np.array_equal(old_state, new_state):
            plt.scatter(old_state[1], old_state[0], c='red', s=120)
        else:
            plt.scatter(old_state[1], old_state[0], c='gray', s=120)
            plt.scatter(new_state[1], new_state[0], c='orange', s=120)

        if record:
            save_simulation(learner, plt, episode_id, time_step)
            time_step += 1
        else:
            plt.pause(0.05)


def save_simulation(learner, plt, episode_id, time_step):
    sim_dir = os.path.join(get_records_folder(learner),
                           f"simulation_e_{episode_id}"
                           )
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    plt.savefig(os.path.join(sim_dir, f"run_t_{time_step}.png"))


def get_records_folder(learner):
    return os.path.join(get_project_dir(),
                        "data",
                        "value_evolution",
                        learner.env.__class__.__name__,
                        learner.env.obstacles.__class__.__name__,
                        learner.__class__.__name__)
