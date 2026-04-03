import matplotlib.pyplot as plt
import time
from pathlib import Path

from ogbench.procgen.maze_json_interface import make_pointymaze_env_from_json


# Specify the maze JSON file to visualize.
maze_json_path = Path(__file__).parent / 'ogbench' / 'procgen' / 'maze_jsons' / '8x8_empty_room_0.json'

# Build a pointymaze environment directly from the JSON maze.
env, parsed_maze = make_pointymaze_env_from_json(
	maze_json_path,
	json_origin='top_left',
	render_mode='rgb_array',
	width=512,
	height=512,
	ob_type='states',
	add_noise_to_goal=False,
)

print(f'Loaded maze: {maze_json_path}')
print(f'Task ID: {parsed_maze.task_id}')
print(f'Dimensions (W,H): ({parsed_maze.width}, {parsed_maze.height})')
print(parsed_maze.to_ascii(add_boundary_walls=True))

# Render the pointymaze env.
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
img_artist = None

ob, info = env.reset(options=dict(task_id=1, render_goal=True))

done = False
steps = 0
while not done and steps < parsed_maze.max_steps:
	frame = env.render()
	if img_artist is None:
		img_artist = ax.imshow(frame)
		ax.set_title(f'Custom PointyMaze: {parsed_maze.task_id}')
		ax.axis('off')
	else:
		img_artist.set_data(frame)

	fig.canvas.draw_idle()
	plt.pause(0.001)

	action = env.action_space.sample()
	ob, reward, terminated, truncated, info = env.step(action)
	done = terminated or truncated
	steps += 1
	time.sleep(0.03)

print(f'Episode done: success={info.get("success", 0.0)} steps={steps}')
env.close()

plt.tight_layout()
plt.show()