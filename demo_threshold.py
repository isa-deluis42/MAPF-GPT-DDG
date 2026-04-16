"""
Runs fast_solver_delta on a batch of envs and prints, per env, the fast-LaCAM
makespan curve, the consecutive diffs, the max diff, and whether that max
crossed diff_threshold (i.e. whether the expert would be queried today).

Requirements to actually run:
  - hf_weights/model-2M-DDG.pt  (student weights)
  - lacam built so LacamInference can load it

Usage:
  python demo_threshold.py --start_seed 0 --num_envs 8 --threshold 3
"""

import argparse
from pathlib import Path

from pogema_toolbox.registry import ToolboxRegistry

from finetuning.delta_data_generator import fast_solver_delta, FastSolverDeltaConfig
from finetuning.scenario_generators import make_pogema_maze_instance
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from lacam.inference import LacamInference, LacamInferenceConfig


FAST_LACAM_TIMELIMIT = 2
EXPERT_LACAM_TIMELIMIT = 10
STEPS_DELTA = 16


def build_batch(start_seed: int, num_envs: int, num_agents: int):
    return [
        make_pogema_maze_instance(
            num_agents=num_agents,
            max_episode_steps=256,
            map_seed=start_seed + i,
            scenario_seed=start_seed + i,
        )
        for i in range(num_envs)
    ]


def summarize(logs, threshold: int) -> None:
    triggered = 0
    for log in logs:
        map_name = log["map_name"]
        fast = log["fast_expert_results"]
        steps = sorted(fast.keys())
        makespans = [fast[s]["makespan"] for s in steps]
        diffs = [makespans[i] - makespans[i - 1] for i in range(1, len(makespans))]
        max_diff = max(diffs) if diffs else None
        crossed = max_diff is not None and max_diff > threshold
        triggered += int(crossed)

        print(f"\n{map_name}")
        print(f"  student ep_length : {log['gpt_results']['ep_length']}")
        print(f"  checkpoint steps  : {steps}")
        print(f"  fast makespans    : {makespans}")
        print(f"  consecutive diffs : {diffs}")
        print(f"  max diff          : {max_diff}")
        print(f"  threshold ({threshold})    : {'CROSSED -> expert queried' if crossed else 'not reached'}")
        if crossed:
            worst_idx = diffs.index(max_diff)
            bad_window = (steps[worst_idx], steps[worst_idx + 1])
            print(f"  worst window      : steps {bad_window[0]} -> {bad_window[1]}")
            print(f"  expert_results    : {log['expert_results']}")

    print(f"\n{triggered}/{len(logs)} envs crossed the threshold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_agents", type=int, default=32)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--weights", type=str, default="hf_weights/model-2M-DDG.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_svg", action="store_true",
                        help="Writes animations for crossed envs to renders/")
    args = parser.parse_args()

    ToolboxRegistry.setup_logger("INFO")

    if not Path(args.weights).exists():
        raise FileNotFoundError(
            f"Student weights not found at {args.weights}. "
            f"Download an hf_weights/model-*.pt checkpoint first."
        )

    learnable_algo = MAPFGPTInference(
        MAPFGPTInferenceConfig(path_to_weights=args.weights, device=args.device)
    )
    fast_solver = LacamInference(
        LacamInferenceConfig(time_limit=FAST_LACAM_TIMELIMIT, timeouts=[FAST_LACAM_TIMELIMIT])
    )
    expert_solver = LacamInference(
        LacamInferenceConfig(time_limit=EXPERT_LACAM_TIMELIMIT, timeouts=[EXPERT_LACAM_TIMELIMIT])
    )

    envs = build_batch(args.start_seed, args.num_envs, args.num_agents)

    cfg = FastSolverDeltaConfig(
        steps_delta=STEPS_DELTA,
        steps_saved=32,
        save_debug_svg=args.save_svg,
        diff_threshold=args.threshold,
    )

    _, logs = fast_solver_delta(envs, learnable_algo, fast_solver, expert_solver, cfg)
    summarize(logs, args.threshold)


if __name__ == "__main__":
    main()
