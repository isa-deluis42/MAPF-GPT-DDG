"""
Convert held-out trajectory .npz files into animated SVGs.

Only saves SVGs for episodes where segment_diffs contains 1, 2, or 3.
"""

import sys
import argparse
import cppimport.import_hook
import numpy as np
from pathlib import Path

from pogema.wrappers.persistence import AgentState

from finetuning.scenario_generators import (
    make_pogema_maze_instance,
    make_pogema_random_instance,
)
from utils.svg_utils import create_multi_animation


def npz_to_svg(npz_path: str, output_path: str = None, allowed_diffs={1, 2, 3}):
    data = np.load(npz_path, allow_pickle=True)

    segment_diffs = data["segment_diffs"]

    if not any(int(x) in allowed_diffs for x in segment_diffs):
        print(f"Skipping: {npz_path}")
        print(f"Segment diffs: {segment_diffs.tolist()}")
        return False

    map_type      = str(data["map_type"])
    map_seed      = int(data["map_seed"])
    scenario_seed = int(data["scenario_seed"])
    num_agents    = int(data["num_agents"])
    positions     = data["positions"]
    goals         = data["goals"]
    obstacles     = data["obstacles"]

    make_env = make_pogema_maze_instance if map_type == "maze" else make_pogema_random_instance
    env = make_env(
        num_agents=num_agents,
        map_seed=map_seed,
        scenario_seed=scenario_seed,
    )
    grid_config = env.grid_config

    wr = grid_config.obs_radius - 1

    T, N, _ = positions.shape
    history = []

    for n in range(N):
        goal_r, goal_c = int(goals[n][0]) - wr, int(goals[n][1]) - wr
        agent_states = []

        for t in range(T):
            r, c = int(positions[t][n][0]) - wr, int(positions[t][n][1]) - wr
            agent_states.append(
                AgentState(
                    x=r,
                    y=c,
                    tx=goal_r,
                    ty=goal_c,
                    step=t,
                    active=True,
                )
            )

        history.append(agent_states)

    if output_path is None:
        output_path = str(Path(npz_path).with_suffix(".svg"))

    create_multi_animation(obstacles, [history], grid_config, name=output_path)

    print(f"Saved: {output_path}")
    print(f"Episode length: {T} steps, {N} agents, {len(segment_diffs)} segments")
    print(f"Segment diffs: {segment_diffs.tolist()}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert one .npz file or a folder of .npz files into SVG animations."
    )

    parser.add_argument(
        "input",
        help="Path to a .npz file or a folder containing .npz files",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Folder to save SVG files. Defaults to same folder as each .npz file.",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search for .npz files recursively inside the input folder.",
    )

    parser.add_argument(
        "--allowed-diffs",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Only save files where segment_diffs contains at least one of these values.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    allowed_diffs = set(args.allowed_diffs)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    saved = 0
    skipped = 0
    failed = 0

    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"Input file must be a .npz file: {input_path}")

        if output_dir is None:
            output_path = input_path.with_suffix(".svg")
        else:
            output_path = output_dir / input_path.with_suffix(".svg").name

        result = npz_to_svg(
            str(input_path),
            str(output_path),
            allowed_diffs=allowed_diffs,
        )

        if result:
            saved += 1
        else:
            skipped += 1

    elif input_path.is_dir():
        npz_files = sorted(
            input_path.rglob("*.npz") if args.recursive else input_path.glob("*.npz")
        )

        if not npz_files:
            print(f"No .npz files found in {input_path}")
            sys.exit(0)

        print(f"Found {len(npz_files)} .npz files")

        for npz_file in npz_files:
            if output_dir is None:
                output_path = npz_file.with_suffix(".svg")
            else:
                output_path = output_dir / npz_file.with_suffix(".svg").name

            try:
                result = npz_to_svg(
                    str(npz_file),
                    str(output_path),
                    allowed_diffs=allowed_diffs,
                )

                if result:
                    saved += 1
                else:
                    skipped += 1

            except Exception as e:
                failed += 1
                print(f"Failed on {npz_file}: {e}")

    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"Done. Saved: {saved}, Skipped: {skipped}, Failed: {failed}")