"""
Warehouse-only DDG threshold demo with visual output.

For each scenario seed:
  1. Run MAPF-GPT (student) full episode, save full SVG, record actions.
  2. Probe fast-LaCAM makespans via fast_solver_delta, plot the curve.
  3. If max diff > threshold, render two zoomed-in SVGs around the bad window:
       - student-window.svg  : replays the student's moves during the window
       - expert-window.svg   : expert LaCAM takes over from the bad step

Requirements:
  - hf_weights/model-2M-DDG.pt  (student weights)
  - lacam built so LacamInference can load it

Usage:
  python demo_threshold.py --start_seed 0 --num_envs 4 --num_agents 64 --threshold 3
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from pogema import AnimationConfig, AnimationMonitor, pogema_v0
from pogema.wrappers.metrics import RuntimeMetricWrapper
from pogema_toolbox.create_env import Environment
from pogema_toolbox.registry import ToolboxRegistry

from finetuning.delta_data_generator import fast_solver_delta, FastSolverDeltaConfig
from finetuning.scenario_generators import make_pogema_map_instance
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from lacam.inference import LacamInference, LacamInferenceConfig
from utils.wrappers import UnrollWrapper


FAST_LACAM_TIMELIMIT = 2
EXPERT_LACAM_TIMELIMIT = 10
STEPS_DELTA = 16
EVAL_CONFIGS_DIR = Path("eval_configs")


def load_map(map_name: str) -> str:
    """Search every eval_configs/*/maps.yaml for the given map name."""
    for maps_file in EVAL_CONFIGS_DIR.rglob("maps.yaml"):
        with open(maps_file, "r") as f:
            maps = yaml.safe_load(f) or {}
        if map_name in maps:
            return maps[map_name]
    raise KeyError(
        f"Map '{map_name}' not found under {EVAL_CONFIGS_DIR}/. "
        f"Try e.g. wfi_warehouse, validation-random-seed-000, "
        f"validation-mazes-seed-000, or puzzle-00."
    )


def build_env_stack(env_cfg, unroll_actions=None, unroll_steps=0):
    """
    AnimationMonitor(UnrollWrapper(pogema_v0(cfg))) so the fast-forward
    via UnrollWrapper doesn't get recorded by AnimationMonitor.
    """
    base = pogema_v0(env_cfg)
    unroll_env = UnrollWrapper(base)
    if unroll_actions is not None:
        unroll_env._recorded_actions = list(unroll_actions)
        unroll_env._recording_episode = False
        unroll_env.set_unroll_steps(unroll_steps)
    anim_env = AnimationMonitor(unroll_env, AnimationConfig(save_every_idx_episode=None))
    runtime_env = RuntimeMetricWrapper(anim_env)
    return runtime_env, unroll_env, anim_env


def render_student_full(student, env_cfg, svg_path: Path):
    env, unroll_env, anim_env = build_env_stack(env_cfg)
    student.reset_states()
    obs, _ = env.reset()
    infos = [{}]
    while True:
        actions = student.act(obs)
        obs, _, term, trunc, infos = env.step(actions)
        if all(term) or all(trunc):
            break
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    anim_env.save_animation(str(svg_path))
    return list(unroll_env._recorded_actions), infos[0].get("metrics", {})


def render_window_replay(env_cfg, unroll_actions, clip_start, clip_len, svg_path: Path):
    """Replay the given actions during [clip_start, clip_start + clip_len]."""
    env, _, anim_env = build_env_stack(env_cfg, unroll_actions=unroll_actions, unroll_steps=clip_start)
    env.reset()
    for i in range(clip_start, min(clip_start + clip_len, len(unroll_actions))):
        _, _, term, trunc, _ = env.step(unroll_actions[i])
        if all(term) or all(trunc):
            break
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    anim_env.save_animation(str(svg_path))


def render_window_with_algo(algo, env_cfg, unroll_actions, clip_start, clip_len, svg_path: Path):
    """Fast-forward using student actions, then let algo drive for clip_len steps."""
    env, _, anim_env = build_env_stack(env_cfg, unroll_actions=unroll_actions, unroll_steps=clip_start)
    if hasattr(algo, "reset_states"):
        algo.reset_states()
    obs, _ = env.reset()
    for _ in range(clip_len):
        actions = algo.act(obs)
        if hasattr(algo, "solved") and not algo.solved:
            break
        obs, _, term, trunc, _ = env.step(actions)
        if all(term) or all(trunc):
            break
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    anim_env.save_animation(str(svg_path))


def diffs_from_log(log):
    fast = log["fast_expert_results"]
    steps = sorted(fast.keys())
    makespans = [fast[s]["makespan"] for s in steps]
    diffs = [makespans[i] - makespans[i - 1] for i in range(1, len(makespans))]
    return steps, makespans, diffs


def plot_makespan(steps, makespans, diffs, threshold, png_path: Path, title: str):
    fig, (ax_m, ax_d) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax_m.plot(steps, makespans, marker="o", color="#1f77b4", label="fast-LaCAM makespan")
    ax_m.set_ylabel("remaining makespan")
    ax_m.grid(True, alpha=0.3)
    ax_m.legend(loc="upper right")

    bar_w = max(1, (steps[1] - steps[0]) * 0.7) if len(steps) > 1 else 1
    bar_colors = ["#d62728" if d > threshold else "#7f7f7f" for d in diffs]
    ax_d.bar(steps[1:], diffs, width=bar_w, color=bar_colors, label="consecutive diff")
    ax_d.axhline(threshold, color="#d62728", linestyle="--", label=f"threshold = {threshold}")
    ax_d.set_xlabel("checkpoint step")
    ax_d.set_ylabel("makespan delta")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="upper right")

    if diffs:
        worst = diffs.index(max(diffs))
        for ax in (ax_m, ax_d):
            ax.axvspan(steps[worst], steps[worst + 1], color="#d62728", alpha=0.15)

    fig.suptitle(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_agents", type=int, default=64)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--weights", type=str, default="hf_weights/model-2M-DDG.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render_dir", type=str, default="renders")
    parser.add_argument("--window_context", type=int, default=8,
                        help="extra steps before and after the bad window in the clip")
    parser.add_argument("--map_name", type=str, default="wfi_warehouse",
                        help="any map name from eval_configs/*/maps.yaml "
                             "(e.g. validation-random-seed-000, validation-mazes-seed-000, puzzle-00)")
    parser.add_argument("--steps_delta", type=int, default=STEPS_DELTA,
                        help="checkpoint interval for fast-LaCAM probing; lower for short episodes")
    args = parser.parse_args()

    ToolboxRegistry.setup_logger("INFO")

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Student weights not found at {args.weights}.")

    grid = load_map(args.map_name)
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
        stem = f"{args.map_name}-seed-{seed}-agents-{args.num_agents}"
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
        student_actions, student_metrics = render_student_full(student, env_cfg, student_svg)
        print(f"  student SVG   -> {student_svg}")
        print(f"  student ep_length={student_metrics.get('ep_length')}, "
              f"makespan={student_metrics.get('makespan')}")

        env_for_ddg = make_pogema_map_instance(
            num_agents=args.num_agents,
            map=grid,
            max_episode_steps=256,
            scenario_seed=seed,
        )
        cfg = FastSolverDeltaConfig(
            steps_delta=args.steps_delta,
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

        ep_length = student_metrics.get("ep_length") or (steps[-1] if steps else 1)
        plot_path = render_dir / f"{stem}-makespan.png"
        plot_makespan(steps, makespans, diffs, args.threshold, plot_path,
                      title=f"{stem}  (ep_length={ep_length})")
        print(f"  makespan PNG  -> {plot_path}")

        if crossed:
            worst = diffs.index(max_diff)
            window_start, window_end = steps[worst], steps[worst + 1]
            clip_start = max(0, window_start - args.window_context)
            clip_len = (window_end - window_start) + 2 * args.window_context
            print(f"  worst window  : steps {window_start} -> {window_end}")
            print(f"  clip range    : steps {clip_start} -> {clip_start + clip_len}")

            student_window_svg = render_dir / f"{stem}-student-window.svg"
            render_window_replay(env_cfg, student_actions, clip_start, clip_len, student_window_svg)
            print(f"  student clip  -> {student_window_svg}")

            expert_window_svg = render_dir / f"{stem}-expert-window.svg"
            expert_render_solver = LacamInference(
                LacamInferenceConfig(
                    time_limit=EXPERT_LACAM_TIMELIMIT,
                    timeouts=[EXPERT_LACAM_TIMELIMIT],
                )
            )
            render_window_with_algo(expert_render_solver, env_cfg, student_actions,
                                    clip_start, clip_len, expert_window_svg)
            print(f"  expert clip   -> {expert_window_svg}")


if __name__ == "__main__":
    main()
