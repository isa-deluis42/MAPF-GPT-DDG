"""
Warehouse-only DDG threshold demo with visual output.

For each scenario seed:
  1. Render MAPF-GPT (student) trajectory as an SVG.
  2. Run fast_solver_delta to probe makespans and compute consecutive diffs.
  3. If max diff > threshold, also render a full expert LaCAM trajectory
     as an SVG so you can compare visually.

Requirements:
  - hf_weights/model-2M-DDG.pt  (student weights)
  - lacam built so LacamInference can load it

Usage:
  python demo_threshold.py --start_seed 0 --num_envs 4 --num_agents 64 --threshold 3
"""

import argparse
from pathlib import Path

import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.run_episode import run_episode

from create_env import create_eval_env
from finetuning.delta_data_generator import fast_solver_delta, FastSolverDeltaConfig
from finetuning.scenario_generators import make_pogema_map_instance
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from lacam.inference import LacamInference, LacamInferenceConfig


FAST_LACAM_TIMELIMIT = 2
EXPERT_LACAM_TIMELIMIT = 10
STEPS_DELTA = 16
WAREHOUSE_MAP_PATH = Path("eval_configs/03-warehouse/maps.yaml")


def load_warehouse_grid() -> str:
    with open(WAREHOUSE_MAP_PATH, "r") as f:
        return yaml.safe_load(f)["wfi_warehouse"]


def render(algo, env_cfg, svg_path: Path):
    env = create_eval_env(env_cfg)
    if hasattr(algo, "reset_states"):
        algo.reset_states()
    metrics = run_episode(env, algo)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    env.save_animation(str(svg_path))
    return metrics


def diffs_from_log(log):
    fast = log["fast_expert_results"]
    steps = sorted(fast.keys())
    makespans = [fast[s]["makespan"] for s in steps]
    diffs = [makespans[i] - makespans[i - 1] for i in range(1, len(makespans))]
    return steps, makespans, diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_agents", type=int, default=64)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--weights", type=str, default="hf_weights/model-2M-DDG.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render_dir", type=str, default="renders")
    args = parser.parse_args()

    ToolboxRegistry.setup_logger("INFO")

    if not Path(args.weights).exists():
        raise FileNotFoundError(
            f"Student weights not found at {args.weights}. "
            f"Download an hf_weights/model-*.pt checkpoint first."
        )

    grid = load_warehouse_grid()
    render_dir = Path(args.render_dir)

    student = MAPFGPTInference(
        MAPFGPTInferenceConfig(path_to_weights=args.weights, device=args.device)
    )
    fast_solver = LacamInference(
        LacamInferenceConfig(time_limit=FAST_LACAM_TIMELIMIT, timeouts=[FAST_LACAM_TIMELIMIT])
    )
    expert_solver_for_ddg = LacamInference(
        LacamInferenceConfig(time_limit=EXPERT_LACAM_TIMELIMIT, timeouts=[EXPERT_LACAM_TIMELIMIT])
    )

    for seed in range(args.start_seed, args.start_seed + args.num_envs):
        stem = f"warehouse-seed-{seed}-agents-{args.num_agents}"
        print(f"\n=== {stem} ===")

        env_cfg = Environment(
            with_animation=True,
            observation_type="MAPF",
            on_target="nothing",
            map=grid,
            max_episode_steps=256,
            num_agents=args.num_agents,
            seed=seed,
            obs_radius=5,
            collision_system="soft",
        )

        student_svg = render_dir / f"{stem}-student.svg"
        student_metrics = render(student, env_cfg, student_svg)
        print(f"  student SVG  -> {student_svg}")
        print(f"  student ep_length={student_metrics.get('ep_length')}, "
              f"makespan={student_metrics.get('makespan')}")

        env_for_ddg = make_pogema_map_instance(
            num_agents=args.num_agents,
            map=grid,
            max_episode_steps=256,
            scenario_seed=seed,
        )
        cfg = FastSolverDeltaConfig(
            steps_delta=STEPS_DELTA,
            steps_saved=32,
            save_debug_svg=False,
            diff_threshold=args.threshold,
        )
        _, logs = fast_solver_delta(
            [env_for_ddg], student, fast_solver, expert_solver_for_ddg, cfg
        )
        log = logs[0]
        steps, makespans, diffs = diffs_from_log(log)
        max_diff = max(diffs) if diffs else None
        crossed = max_diff is not None and max_diff > args.threshold

        print(f"  probed steps  : {steps}")
        print(f"  fast makespans: {makespans}")
        print(f"  diffs         : {diffs}")
        print(f"  max diff      : {max_diff}")
        print(f"  threshold gate: {'CROSSED' if crossed else 'not reached'}")

        if crossed:
            worst = diffs.index(max_diff)
            print(f"  worst window  : steps {steps[worst]} -> {steps[worst + 1]}")
            expert_render_solver = LacamInference(
                LacamInferenceConfig(
                    time_limit=EXPERT_LACAM_TIMELIMIT,
                    timeouts=[EXPERT_LACAM_TIMELIMIT],
                )
            )
            expert_svg = render_dir / f"{stem}-expert.svg"
            render(expert_render_solver, env_cfg, expert_svg)
            print(f"  expert SVG   -> {expert_svg}")


if __name__ == "__main__":
    main()
