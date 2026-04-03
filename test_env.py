import time

import matplotlib.pyplot as plt

import ogbench

# Make an environment and datasets (they will be automatically downloaded).
dataset_name = 'pointymaze-multipath-navigate-v0'  # The dataset name specifies the environment and dataset type.
env = ogbench.make_env_and_datasets(
    dataset_name,
    env_only=True,
    render_mode='rgb_array',
    width=512,
    height=512,
)

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
img_artist = None

# Train your offline goal-conditioned RL agent on the dataset.
# ...

# Evaluate the agent.
for task_id in [1, 2, 3, 4, 5]:
    # Reset the environment and set the evaluation task.
    ob, info = env.reset(
        options=dict(
            task_id=task_id,  # Set the evaluation task. Each environment provides five
                              # evaluation goals, and `task_id` must be in [1, 5].
            render_goal=True,  # Set to `True` to get a rendered goal image (optional).
        )
    )

    goal = info['goal']  # Get the goal observation to pass to the agent.
    goal_rendered = info['goal_rendered']  # Get the rendered goal image (optional).

    # Force an initial draw for this episode.
    frame = env.render()
    if img_artist is None:
        img_artist = ax.imshow(frame)
        ax.set_title('OGBench PointMaze (top-down)')
        ax.axis('off')
    else:
        img_artist.set_data(frame)
    fig.canvas.draw_idle()
    plt.pause(0.001)

    done = False
    while not done:
        action = env.action_space.sample()  # Replace this with your agent's action.
        ob, reward, terminated, truncated, info = env.step(action)  # Gymnasium-style step.
        # If the agent reaches the goal, `terminated` will be `True`. If the episode length
        # exceeds the maximum length without reaching the goal, `truncated` will be `True`.
        # `reward` is 1 if the agent reaches the goal and 0 otherwise.
        done = terminated or truncated
        frame = env.render()  # Returns an RGB frame in 'rgb_array' mode.
        img_artist.set_data(frame)
        fig.canvas.draw_idle()
        plt.pause(0.001)
        time.sleep(0.03)

    success = info['success']  # Whether the agent reached the goal (0 or 1).
                               # `terminated` also indicates this.

    env.close()
    plt.ioff()
    plt.show(block=False)
    input('Done. Press Enter to exit...')