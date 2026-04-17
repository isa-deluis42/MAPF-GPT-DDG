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
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap
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


def replay_trajectories(env_cfg, student_actions):
    """Replay student actions, return obstacles, goals, per-step positions, completion times."""
    base = pogema_v0(env_cfg)
    obs, _ = base.reset()
    num_agents = len(obs)
    obstacles = obs[0]["global_obstacles"]
    goals = [tuple(o["global_target_xy"]) for o in obs]
    positions_per_step = [[tuple(o["global_xy"]) for o in obs]]
    completion_time = [None] * num_agents
    for i, (pos, goal) in enumerate(zip(positions_per_step[0], goals)):
        if pos == goal:
            completion_time[i] = 0

    for step_idx, actions in enumerate(student_actions, start=1):
        obs, _, term, trunc, _ = base.step(actions)
        positions = [tuple(o["global_xy"]) for o in obs]
        positions_per_step.append(positions)
        for i, (pos, goal) in enumerate(zip(positions, goals)):
            if completion_time[i] is None and pos == goal:
                completion_time[i] = step_idx
        if all(term) or all(trunc):
            break

    ep_length = len(positions_per_step) - 1
    for i in range(num_agents):
        if completion_time[i] is None:
            completion_time[i] = ep_length

    return {
        "obstacles": obstacles,
        "goals": goals,
        "positions_per_step": positions_per_step,
        "completion_time": completion_time,
        "num_agents": num_agents,
        "ep_length": ep_length,
    }


def compute_bottleneck(traj):
    """Extract bottleneck agent's path from a replay trajectory."""
    idx = max(range(traj["num_agents"]), key=lambda i: traj["completion_time"][i])
    path = [traj["positions_per_step"][t][idx] for t in range(len(traj["positions_per_step"]))]
    return {
        "obstacles": traj["obstacles"],
        "bottleneck_idx": idx,
        "bottleneck_path": path,
        "bottleneck_start": traj["positions_per_step"][0][idx],
        "bottleneck_goal": traj["goals"][idx],
        "completion_time": traj["completion_time"],
    }


def bfs_distance_map(obstacles, goal):
    """Shortest-path distance from `goal` to every free cell. Unreachable cells = -1."""
    H, W = obstacles.shape
    dist = np.full((H, W), -1, dtype=int)
    gr, gc = int(goal[0]), int(goal[1])
    if not (0 <= gr < H and 0 <= gc < W) or obstacles[gr, gc]:
        return dist
    dist[gr, gc] = 0
    q = deque([(gr, gc)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not obstacles[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def compute_fleet_stats(traj):
    """
    Classify each (agent, step) as advance/wait/detour/at-goal and
    return a per-step count of stalled agents.
    """
    positions_per_step = traj["positions_per_step"]
    goals = traj["goals"]
    obstacles = traj["obstacles"]
    num_agents = traj["num_agents"]

    dist_maps = [bfs_distance_map(obstacles, g) for g in goals]

    # 0 = at goal, 1 = advance (dist to goal decreased)
    # 2 = wait (same cell), 3 = detour (moved but dist didn't decrease)
    T = len(positions_per_step)
    state_matrix = np.zeros((num_agents, T), dtype=int)
    for i in range(num_agents):
        state_matrix[i, 0] = 0 if positions_per_step[0][i] == goals[i] else 1

    stall_per_step = [0]
    for t in range(1, T):
        stalls = 0
        for i in range(num_agents):
            pos = positions_per_step[t][i]
            prev = positions_per_step[t - 1][i]
            if pos == goals[i]:
                state_matrix[i, t] = 0
                continue
            if pos == prev:
                state_matrix[i, t] = 2
                stalls += 1
                continue
            dm = dist_maps[i]
            d_now = dm[pos[0], pos[1]] if dm[pos[0], pos[1]] >= 0 else 10**6
            d_prev = dm[prev[0], prev[1]] if dm[prev[0], prev[1]] >= 0 else 10**6
            if d_now < d_prev:
                state_matrix[i, t] = 1
            else:
                state_matrix[i, t] = 3
                stalls += 1
        stall_per_step.append(stalls)

    return {"state_matrix": state_matrix, "stall_per_step": stall_per_step}


def plot_fleet_stall_timeline(stall_per_step, num_agents, png_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 3))
    steps = list(range(len(stall_per_step)))
    ax.fill_between(steps, 0, stall_per_step, color="#d62728", alpha=0.4)
    ax.plot(steps, stall_per_step, color="#d62728", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("# agents stalled")
    ax.set_ylim(0, num_agents)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def plot_gantt(state_matrix, completion_time, png_path: Path, title: str):
    order = sorted(range(len(completion_time)), key=lambda i: completion_time[i])
    sorted_matrix = state_matrix[order]

    # 0 at-goal (green), 1 advance (blue), 2 wait (gray), 3 detour (orange)
    cmap = ListedColormap(["#2ca02c", "#a7c8ff", "#9e9e9e", "#ff9933"])

    fig_h = max(3, len(order) * 0.15)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(sorted_matrix, aspect="auto", cmap=cmap,
                   vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_xlabel("step")
    ax.set_ylabel("agent (sorted by completion time)")
    label_stride = max(1, len(order) // 30)
    ax.set_yticks(range(0, len(order), label_stride))
    ax.set_yticklabels([str(order[i]) for i in range(0, len(order), label_stride)], fontsize=7)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.04)
    cbar.ax.set_yticklabels(["at goal", "advance", "wait", "detour"])
    ax.set_title(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def compute_expert_path(env_cfg, target_agent_idx, time_limit=EXPERT_LACAM_TIMELIMIT):
    """Run LaCAM from step 0; return (positions, completion_time) for target agent."""
    solver = LacamInference(LacamInferenceConfig(time_limit=time_limit, timeouts=[time_limit]))
    base = pogema_v0(env_cfg)
    obs, _ = base.reset()
    target_goal = tuple(obs[target_agent_idx]["global_target_xy"])
    positions = [tuple(obs[target_agent_idx]["global_xy"])]
    completion = None
    if positions[0] == target_goal:
        completion = 0

    while True:
        actions = solver.act(obs)
        if not solver.solved:
            return None, None
        obs, _, term, trunc, _ = base.step(actions)
        pos = tuple(obs[target_agent_idx]["global_xy"])
        positions.append(pos)
        if completion is None and pos == target_goal:
            completion = len(positions) - 1
        if all(term) or all(trunc):
            break
    if completion is None:
        completion = len(positions) - 1
    return positions, completion


def plot_bottleneck_path(info, png_path: Path, title: str, expert_path=None, expert_completion=None):
    obstacles = info["obstacles"]
    path = info["bottleneck_path"]
    start = info["bottleneck_start"]
    goal = info["bottleneck_goal"]
    idx = info["bottleneck_idx"]
    completion = info["completion_time"][idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(obstacles, cmap="Greys", origin="upper", interpolation="nearest")

    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    ax.plot(cols, rows, color="#d62728", linewidth=2, alpha=0.85,
            label=f"student agent {idx} ({completion} steps)")

    if expert_path is not None:
        erows = [p[0] for p in expert_path]
        ecols = [p[1] for p in expert_path]
        ax.plot(ecols, erows, color="#2ca02c", linewidth=2, alpha=0.85, linestyle="--",
                label=f"expert agent {idx} ({expert_completion} steps)")

    ax.scatter([start[1]], [start[0]], marker="o", color="#d62728", s=120,
               edgecolor="white", zorder=3, label="start")
    ax.scatter([goal[1]], [goal[0]], marker="*", color="#2ca02c", s=220,
               edgecolor="white", zorder=3, label="goal")

    stride = max(1, len(path) // 12)
    for t in range(0, len(path), stride):
        ax.annotate(str(t), (path[t][1], path[t][0]),
                    color="#1f77b4", fontsize=7, ha="center", va="center",
                    xytext=(0, 0), textcoords="offset points")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


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

        traj = replay_trajectories(env_cfg, student_actions)
        bn = compute_bottleneck(traj)
        fleet = compute_fleet_stats(traj)

        stall_png = render_dir / f"{stem}-fleet-stall.png"
        plot_fleet_stall_timeline(
            fleet["stall_per_step"], traj["num_agents"], stall_png,
            title=f"{stem}  fleet stall timeline",
        )
        print(f"  stall PNG     -> {stall_png}  "
              f"(peak {max(fleet['stall_per_step'])}/{traj['num_agents']} agents stalled)")

        gantt_png = render_dir / f"{stem}-gantt.png"
        plot_gantt(fleet["state_matrix"], traj["completion_time"], gantt_png,
                   title=f"{stem}  agent state Gantt")
        print(f"  Gantt PNG     -> {gantt_png}")

        expert_path, expert_completion = compute_expert_path(env_cfg, bn["bottleneck_idx"])
        if expert_path is None:
            print(f"  note: LaCAM failed to solve full episode; skipping expert overlay")
        bottleneck_png = render_dir / f"{stem}-bottleneck.png"
        plot_bottleneck_path(
            bn, bottleneck_png,
            title=(f"{stem}  agent {bn['bottleneck_idx']}  "
                   f"student={bn['completion_time'][bn['bottleneck_idx']]}"
                   + (f"  expert={expert_completion}" if expert_completion is not None else "")),
            expert_path=expert_path,
            expert_completion=expert_completion,
        )
        print(f"  bottleneck PNG-> {bottleneck_png}  "
              f"(student {bn['completion_time'][bn['bottleneck_idx']]}"
              + (f", expert {expert_completion}" if expert_completion is not None else "")
              + " steps)")

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
