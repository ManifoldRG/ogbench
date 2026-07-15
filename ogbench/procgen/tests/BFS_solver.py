"""Test helper that adapts procgen maze JSON specs to the mazegen BFS solver."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]
_V2_SRC = _REPO_ROOT / 'src' / 'v2'
if str(_V2_SRC) not in sys.path:
	sys.path.insert(0, str(_V2_SRC))

from automatic_maze_generation.mazegen.models import Door, Gate, Key, MazeInstance, Switch
from automatic_maze_generation.mazegen.solver import solve_maze


def _point(value):
	return tuple(value)


def _maze_instance_from_spec(spec):
	maze = spec['maze']
	mechanisms = spec.get('mechanisms', {})
	width, height = maze['dimensions']

	return MazeInstance(
		width=width,
		height=height,
		walls={_point(wall) for wall in maze.get('walls', [])},
		start=_point(maze['start']),
		goal=_point(maze['goal']),
		keys=[
			Key(
				id=key['id'],
				position=_point(key['position']),
				color=key['color'],
			)
			for key in mechanisms.get('keys', [])
		],
		doors=[
			Door(
				id=door['id'],
				position=_point(door['position']),
				requires_key=door.get('requires_key', door.get('color')),
				initial_state=door.get('initial_state', 'locked'),
			)
			for door in mechanisms.get('doors', [])
		],
		switches=[
			Switch(
				id=switch['id'],
				position=_point(switch['position']),
				controls=list(switch.get('controls', [])),
				switch_type=switch.get('switch_type', 'toggle'),
				initial_state=switch.get('initial_state', 'off'),
			)
			for switch in mechanisms.get('switches', [])
		],
		gates=[
			Gate(
				id=gate['id'],
				position=_point(gate['position']),
				initial_state=gate.get('initial_state', 'closed'),
			)
			for gate in mechanisms.get('gates', [])
		],
		metadata=dict(spec.get('metadata', {})),
	)


def solve(spec):
	"""Return the mazegen BFS solver result for a procgen maze JSON spec."""
	return solve_maze(_maze_instance_from_spec(spec))


def find_all_paths(spec):
	"""Return the BFS solver path in the legacy list-of-paths test-helper shape."""
	result = solve(spec)
	if not result['is_solvable']:
		return []
	return [result['path']]
