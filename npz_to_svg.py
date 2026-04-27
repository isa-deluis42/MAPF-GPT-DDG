"""
Convert a held-out trajectory .npz into an animated SVG.

Does not require re-running inference — reconstructs the animation
directly from the saved positions, obstacles, and goals.

Usage:
    python npz_to_svg.py <path/to/episode.npz> [output.svg]

Example:
    python npz_to_svg.py dataset/held_out/ckpt_0/maze_ms128_ss1000_na16.npz
"""

import sys
import cppimport.import_hook
import numpy as np
from pathlib import Path

from pogema import AnimationConfig, GridConfig
from pogema.wrappers.persistence import AgentState

from finetuning.scenario_generators import make_pogema_maze_instance, make_pogema_random_instance
from utils.svg_utils import create_multi_animation


def npz_to_svg(npz_path: str, output_path: str = None):
    data = np.load(npz_path, allow_pickle=True)

    map_type      = str(data["map_type"])
    map_seed      = int(data["map_seed"])
    scenario_seed = int(data["scenario_seed"])
    num_agents    = int(data["num_agents"])
    positions     = data["positions"]     # (T, N, 2) as (row, col)
    goals         = data["goals"]         # (N, 2)    as (row, col)
    obstacles     = data["obstacles"]     # (H, W)
    segment_diffs = data["segment_diffs"] # (S,)

    # Recreate env just to get grid_config — no inference
    make_env = make_pogema_maze_instance if map_type == "maze" else make_pogema_random_instance
    env = make_env(num_agents=num_agents, map_seed=map_seed, scenario_seed=scenario_seed)
    grid_config = env.grid_config

    # AgentState uses cropped coordinates: pos - (obs_radius - 1)
    wr = grid_config.obs_radius - 1

    T, N, _ = positions.shape
    history = []
    for n in range(N):
        goal_r, goal_c = int(goals[n][0]) - wr, int(goals[n][1]) - wr
        agent_states = []
        for t in range(T):
            r, c = int(positions[t][n][0]) - wr, int(positions[t][n][1]) - wr
            agent_states.append(AgentState(x=r, y=c, tx=goal_r, ty=goal_c, step=t, active=True))
        history.append(agent_states)

    if output_path is None:
        output_path = str(Path(npz_path).with_suffix(".svg"))

    create_multi_animation(obstacles, [history], grid_config, name=output_path)

    print(f"Saved: {output_path}")
    print(f"Episode length: {T} steps, {N} agents, {len(segment_diffs)} segments")
    print(f"Segment diffs: {segment_diffs.tolist()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npz_to_svg.py <episode.npz> [output.svg]")
        sys.exit(1)
    npz_to_svg(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
