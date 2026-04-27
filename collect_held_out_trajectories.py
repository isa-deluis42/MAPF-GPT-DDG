"""
Collect held-out trajectory data from a MAPF-GPT checkpoint.

For each held-out episode:
  1. Run MAPF-GPT, record agent positions at every timestep
  2. Run LaCAM(2s) probes at each 16-step segment boundary (parallel)
  3. Compute per-segment diffs
  4. Save as .npz

Usage:
    python finetuning/collect_held_out_trajectories.py \
        --checkpoint checkpoints/ckpt_ddg_500.pt \
        --output dataset/held_out \
        --device cuda
"""

import argparse
import cppimport.import_hook
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

from pogema_toolbox.run_episode import run_episode
from pogema_toolbox.registry import ToolboxRegistry
from pogema.wrappers.metrics import RuntimeMetricWrapper

from lacam.inference import LacamInference, LacamInferenceConfig
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from utils.wrappers import UnrollWrapper
from finetuning.scenario_generators import make_pogema_maze_instance, make_pogema_random_instance
from held_out_seed_set import (
    iter_held_out_configs, STEPS_DELTA, MAX_EPISODE_STEPS,
)

FAST_TIME_LIMIT = 2
NUM_PROBE_WORKERS = 8


def _run_lacam_probe(args):
    """Run LaCAM(2s) from unroll_steps. Module-level for Pool pickling."""
    env, unroll_steps = args
    env = deepcopy(env)
    solver = LacamInference(LacamInferenceConfig(time_limit=FAST_TIME_LIMIT, timeouts=[FAST_TIME_LIMIT]))
    env.set_unroll_steps(unroll_steps)
    results = run_episode(env, solver)
    return unroll_steps, int(results.get("makespan", MAX_EPISODE_STEPS))


def collect_episode(
    map_type: str,
    map_seed: int,
    scenario_seed: int,
    num_agents: int,
    algo: MAPFGPTInference,
    checkpoint_iter: int,
) -> dict | None:
    if map_type == "maze":
        env = make_pogema_maze_instance(
            num_agents=num_agents, max_episode_steps=MAX_EPISODE_STEPS,
            map_seed=map_seed, scenario_seed=scenario_seed,
        )
    else:
        env = make_pogema_random_instance(
            num_agents=num_agents, max_episode_steps=MAX_EPISODE_STEPS,
            map_seed=map_seed, scenario_seed=scenario_seed,
        )

    env = RuntimeMetricWrapper(env)
    env = UnrollWrapper(env)

    algo.reset_states()
    obs, _ = env.reset()

    # global_xy is (row, col) in pogema convention
    obstacles = obs[0]["global_obstacles"].copy().astype(np.int8)   # (H, W)
    goals = np.array([o["global_target_xy"] for o in obs], dtype=np.int16)  # (N, 2) as (row, col)

    all_positions = []
    for _ in range(MAX_EPISODE_STEPS):
        all_positions.append(np.array([o["global_xy"] for o in obs], dtype=np.int16))  # (N, 2) as (row, col)
        actions = algo.act(obs)
        obs, _, terminated, truncated, _ = env.step(actions)
        if all(terminated) or all(truncated):
            break

    episode_length = len(all_positions)
    num_segments = episode_length // STEPS_DELTA

    if num_segments < 2:
        return None

    # Probe at each segment boundary: 0, 16, 32, ..., num_segments * 16
    probe_steps = [i * STEPS_DELTA for i in range(num_segments + 1)]

    with Pool(processes=NUM_PROBE_WORKERS) as pool:
        probe_results = pool.map(_run_lacam_probe, [(env, s) for s in probe_steps])

    makespans = {step: makespan for step, makespan in probe_results}
    segment_diffs = np.array(
        [makespans[probe_steps[i + 1]] - makespans[probe_steps[i]] for i in range(num_segments)],
        dtype=np.int16,
    )

    return {
        "obstacles": obstacles,
        "positions": np.stack(all_positions),   # (T, N, 2) as (x, y)
        "goals": goals,
        "segment_diffs": segment_diffs,
        "episode_length": np.int32(episode_length),
        "num_agents": np.int32(num_agents),
        "map_seed": np.int32(map_seed),
        "scenario_seed": np.int32(scenario_seed),
        "map_type": map_type,
        "checkpoint_iter": np.int32(checkpoint_iter),
    }


def _parse_checkpoint_iter(checkpoint_path: str) -> int:
    stem = Path(checkpoint_path).stem          # e.g. "ckpt_ddg_500"
    return int(stem.split("_")[-1])


def main():
    parser = argparse.ArgumentParser(description="Collect held-out trajectories from a MAPF-GPT checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to MAPF-GPT checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Root output directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ToolboxRegistry.setup_logger("INFO")

    checkpoint_iter = _parse_checkpoint_iter(args.checkpoint)
    output_dir = Path(args.output) / f"ckpt_{checkpoint_iter}"
    output_dir.mkdir(parents=True, exist_ok=True)

    algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=args.checkpoint, device=args.device))

    configs = list(iter_held_out_configs())
    for i, (map_type, map_seed, scenario_seed, num_agents) in enumerate(configs):
        out_path = output_dir / f"{map_type}_ms{map_seed}_ss{scenario_seed}_na{num_agents}.npz"
        if out_path.exists():
            continue

        ToolboxRegistry.info(
            f"[{i + 1}/{len(configs)}] {map_type} ms={map_seed} ss={scenario_seed} na={num_agents}"
        )
        data = collect_episode(map_type, map_seed, scenario_seed, num_agents, algo, checkpoint_iter)

        if data is None:
            ToolboxRegistry.info("  skipped (episode too short)")
            continue

        np.savez_compressed(str(out_path), **data)

    ToolboxRegistry.info(f"Done. Episodes saved to {output_dir}")


if __name__ == "__main__":
    main()
