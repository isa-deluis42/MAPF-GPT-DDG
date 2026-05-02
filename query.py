
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


EPS = 1e-8


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def binary_entropy(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def load_existing_annotations(output_path):
    output_path = Path(output_path)

    if not output_path.exists():
        return []

    try:
        with open(output_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        raise ValueError("Expected annotation JSON to contain a list.")
    except Exception as e:
        print(f"Warning: could not load existing annotations from {output_path}: {e}")
        return []


def append_annotation(output_path, entry):
    output_path = Path(output_path)

    annotations = load_existing_annotations(output_path)
    annotations.append(entry)

    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)


def already_labeled_episode(annotations, scenario_id):
    return any(ann.get("scenario_id") == scenario_id for ann in annotations)


# ---------------------------------------------------------------------
# Segment handling
# ---------------------------------------------------------------------

def infer_segment_ranges(num_steps, num_segments):
    """
    Infer segment timestep ranges from the number of segment_diffs.

    This follows the same assumption as the replay viewer:
    if the .npz has S segment_diffs, split the T timesteps into S
    approximately equal ranges.

    If your dataset later stores exact segment boundaries, replace this
    function with those stored boundaries.
    """
    if num_segments <= 1:
        return [(0, num_steps - 1)]

    boundaries = np.linspace(0, num_steps, num_segments + 1).astype(int)

    ranges = []
    for i in range(num_segments):
        start = int(boundaries[i])
        end = int(boundaries[i + 1] - 1)
        end = max(start, end)
        ranges.append((start, end))

    return ranges


def extract_segment_arrays(positions, segment_range):
    start, end = segment_range
    return positions[start:end + 1]


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def shortest_path_overlap_proxy(segment_positions, goals):
    """
    Approximate shortest-path overlap / shared corridor pressure.

    This is intentionally lightweight and solver-free:
    for each timestep, estimate how many agents share rows or columns
    with other agents while moving toward goals.

    Higher values roughly indicate more shared corridors and possible
    congestion pressure.
    """
    T, N, _ = segment_positions.shape

    if N <= 1:
        return 0.0

    overlap_count = 0
    possible_pairs = T * (N * (N - 1) / 2)

    for t in range(T):
        for i in range(N):
            r_i, c_i = segment_positions[t, i]
            gr_i, gc_i = goals[i]

            for j in range(i + 1, N):
                r_j, c_j = segment_positions[t, j]
                gr_j, gc_j = goals[j]

                same_row_pressure = (
                    r_i == r_j
                    and min(c_i, gc_i) <= c_j <= max(c_i, gc_i)
                ) or (
                    r_j == r_i
                    and min(c_j, gc_j) <= c_i <= max(c_j, gc_j)
                )

                same_col_pressure = (
                    c_i == c_j
                    and min(r_i, gr_i) <= r_j <= max(r_i, gr_i)
                ) or (
                    c_j == c_i
                    and min(r_j, gr_j) <= r_i <= max(r_j, gr_j)
                )

                if same_row_pressure or same_col_pressure:
                    overlap_count += 1

    return float(overlap_count / max(possible_pairs, 1))


def extract_segment_features(positions, goals, obstacles, segment_ranges):
    """
    Return:
        features: np.ndarray, shape (S, D)
        feature_names: list[str]

    Features are deliberately simple and interpretable. They are meant to
    support active query selection, not replace your eventual 3D CNN feature
    representation.
    """
    features = []

    obstacle_density = float(np.mean(obstacles > 0))
    H, W = obstacles.shape

    for seg_idx, segment_range in enumerate(segment_ranges):
        seg = extract_segment_arrays(positions, segment_range)
        T_seg, N, _ = seg.shape

        # Movement between consecutive timesteps
        if T_seg > 1:
            deltas = np.abs(np.diff(seg, axis=0)).sum(axis=2)  # (T-1, N)
            moved = deltas > 0
            wait_fraction = float(np.mean(~moved))
            mean_move = float(np.mean(deltas))
            max_move = float(np.max(deltas))
        else:
            wait_fraction = 1.0
            mean_move = 0.0
            max_move = 0.0

        # Goal distance
        final_positions = seg[-1]
        goal_distances = np.abs(final_positions - goals).sum(axis=1)
        mean_goal_distance = float(np.mean(goal_distances))
        max_goal_distance = float(np.max(goal_distances))

        # Repeated positions / oscillation proxy
        if T_seg > 2:
            backtrack_count = 0
            total_checks = 0

            for t in range(2, T_seg):
                backtrack_count += np.sum(np.all(seg[t] == seg[t - 2], axis=1))
                total_checks += N

            oscillation_fraction = float(backtrack_count / max(total_checks, 1))
        else:
            oscillation_fraction = 0.0

        # Crowding proxy: agents adjacent or on same cell
        crowding_values = []
        same_cell_values = []

        for t in range(T_seg):
            pos_t = seg[t]
            close_pairs = 0
            same_cell_pairs = 0
            total_pairs = max(N * (N - 1) / 2, 1)

            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.abs(pos_t[i] - pos_t[j]).sum()

                    if dist <= 1:
                        close_pairs += 1

                    if dist == 0:
                        same_cell_pairs += 1

            crowding_values.append(close_pairs / total_pairs)
            same_cell_values.append(same_cell_pairs / total_pairs)

        mean_crowding = float(np.mean(crowding_values))
        max_crowding = float(np.max(crowding_values))
        mean_same_cell = float(np.mean(same_cell_values))
        max_same_cell = float(np.max(same_cell_values))

        # Spatial spread: low spread can mean clustering/congestion
        row_std = float(np.mean(np.std(seg[:, :, 0], axis=1)))
        col_std = float(np.mean(np.std(seg[:, :, 1], axis=1)))
        spatial_spread = row_std + col_std

        # Path-overlap / corridor-pressure proxy
        path_overlap = shortest_path_overlap_proxy(seg, goals)

        feature_vector = [
            wait_fraction,
            mean_move,
            max_move,
            mean_goal_distance / max(H + W, 1),
            max_goal_distance / max(H + W, 1),
            oscillation_fraction,
            mean_crowding,
            max_crowding,
            mean_same_cell,
            max_same_cell,
            spatial_spread / max(H + W, 1),
            path_overlap,
            obstacle_density,
            T_seg / max(len(positions), 1),
        ]

        features.append(feature_vector)

    feature_names = [
        "wait_fraction",
        "mean_move",
        "max_move",
        "mean_goal_distance_norm",
        "max_goal_distance_norm",
        "oscillation_fraction",
        "mean_crowding",
        "max_crowding",
        "mean_same_cell",
        "max_same_cell",
        "spatial_spread_norm",
        "path_overlap_proxy",
        "obstacle_density",
        "segment_length_fraction",
    ]

    return np.array(features, dtype=np.float32), feature_names


def normalize_features(features):
    """
    Per-episode feature normalization for pair scoring.

    This makes ||phi_A - phi_B|| less sensitive to feature scale.
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)

    return (features - mean) / (std + EPS)


# ---------------------------------------------------------------------
# Optional model scoring
# ---------------------------------------------------------------------

def load_torch_model(model_path):
    """
    Optional model loading.

    Expects a PyTorch model saved with torch.save(model, path), where:
        model(features_tensor) -> scores

    If your project saves state_dicts instead, adapt this function to
    recreate your model class before loading weights.
    """
    if model_path is None:
        return None

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to use --model-path.")

    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model


def score_segments(features, model=None):
    """
    Return one scalar score per segment.

    If no model is provided, return zeros. That makes p=0.5 for every pair,
    so pair selection is driven by feature distance.
    """
    if model is None:
        return np.zeros(features.shape[0], dtype=np.float32)

    import torch

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        scores = model(x)

        if isinstance(scores, tuple):
            scores = scores[0]

        scores = scores.squeeze(-1).detach().cpu().numpy()

    return scores.astype(np.float32)


# ---------------------------------------------------------------------
# Candidate pair selection
# ---------------------------------------------------------------------

def build_candidate_pairs(num_segments):
    return list(combinations(range(num_segments), 2))


def choose_best_pair(features, scores):
    """
    Pick pair with max:
        H(sigmoid(sA - sB)) * ||phi_A - phi_B||
    """
    if len(features) < 2:
        return None

    norm_features = normalize_features(features)

    best = None
    best_score = -math.inf

    for a, b in build_candidate_pairs(len(features)):
        p = sigmoid(scores[a] - scores[b])
        entropy = binary_entropy(p)
        distance = np.linalg.norm(norm_features[a] - norm_features[b])
        eig_proxy = float(entropy * distance)

        if eig_proxy > best_score:
            best_score = eig_proxy
            best = {
                "segment_a": int(a),
                "segment_b": int(b),
                "model_score_a": float(scores[a]),
                "model_score_b": float(scores[b]),
                "preference_probability_a_worse": float(p),
                "entropy": float(entropy),
                "feature_distance": float(distance),
                "eig_proxy": float(eig_proxy),
            }

    return best


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def draw_segment(ax, positions, goals, obstacles, segment_range, title, agent_colors):
    H, W = obstacles.shape
    start, end = segment_range

    ax.set_facecolor("#f7f7f7")

    # Grid
    for r in range(H):
        for c in range(W):
            color = "#2f2f2f" if obstacles[r, c] else "#ffffff"
            rect = patches.Rectangle(
                (c, r),
                1,
                1,
                facecolor=color,
                edgecolor="#d0d0d0",
                linewidth=0.5,
            )
            ax.add_patch(rect)

    # Goals
    for i, (gr, gc) in enumerate(goals):
        color = agent_colors[i % len(agent_colors)]

        goal = patches.Rectangle(
            (gc + 0.18, gr + 0.18),
            0.64,
            0.64,
            facecolor="none",
            edgecolor=color,
            linewidth=2.0,
            linestyle="--",
        )
        ax.add_patch(goal)

        ax.text(
            gc + 0.5,
            gr + 0.5,
            str(i),
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            weight="bold",
        )

    # Trajectories over the segment
    segment_positions = positions[start:end + 1]

    for i in range(positions.shape[1]):
        color = agent_colors[i % len(agent_colors)]

        xs = segment_positions[:, i, 1] + 0.5
        ys = segment_positions[:, i, 0] + 0.5

        ax.plot(xs, ys, linewidth=1.5, alpha=0.7, color=color)

        # Start marker
        ax.scatter(
            xs[0],
            ys[0],
            s=45,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

        # End marker
        ax.scatter(
            xs[-1],
            ys[-1],
            s=70,
            marker="s",
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )

        ax.text(
            xs[-1],
            ys[-1],
            str(i),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            weight="bold",
            zorder=5,
        )

    ax.set_title(title, fontsize=11, weight="bold", pad=10)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(0, W + 1, 1))
    ax.set_yticks(np.arange(0, H + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)


def show_pair(npz_path, positions, goals, obstacles, segment_ranges, segment_diffs, pair_info):
    a = pair_info["segment_a"]
    b = pair_info["segment_b"]

    num_agents = positions.shape[1]
    agent_colors = plt.cm.tab20(np.linspace(0, 1, max(num_agents, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    a_start, a_end = segment_ranges[a]
    b_start, b_end = segment_ranges[b]

    a_diff = int(segment_diffs[a]) if segment_diffs is not None else None
    b_diff = int(segment_diffs[b]) if segment_diffs is not None else None

    draw_segment(
        axes[0],
        positions,
        goals,
        obstacles,
        segment_ranges[a],
        title=f"A: segment {a} | t={a_start}-{a_end} | diff={a_diff}",
        agent_colors=agent_colors,
    )

    draw_segment(
        axes[1],
        positions,
        goals,
        obstacles,
        segment_ranges[b],
        title=f"B: segment {b} | t={b_start}-{b_end} | diff={b_diff}",
        agent_colors=agent_colors,
    )

    fig.suptitle(
        (
            f"{Path(npz_path).name}\n"
            f"EIG proxy={pair_info['eig_proxy']:.4f}, "
            f"p(A worse)={pair_info['preference_probability_a_worse']:.3f}, "
            f"H(p)={pair_info['entropy']:.3f}, "
            f"feature distance={pair_info['feature_distance']:.3f}"
        ),
        fontsize=12,
        weight="bold",
    )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Interactive query loop
# ---------------------------------------------------------------------

def prompt_user_for_label():
    print()
    print("Which segment has worse congestion?")
    print("  a = Segment A is worse")
    print("  b = Segment B is worse")
    print("  u = unsure / skip")
    print("  s = skip")
    print("  q = quit")

    while True:
        answer = input("Your choice [a/b/u/s/q]: ").strip().lower()

        if answer in {"a", "b", "u", "s", "q"}:
            return answer

        print("Please enter one of: a, b, u, s, q")


def process_npz(npz_path, output_path, model=None, skip_already_labeled=True):
    annotations = load_existing_annotations(output_path)

    scenario_id = Path(npz_path).stem

    if skip_already_labeled and already_labeled_episode(annotations, scenario_id):
        print(f"Skipping already-labeled episode: {scenario_id}")
        return "skipped_existing"

    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["positions", "goals", "obstacles", "segment_diffs"]
    missing = [key for key in required_keys if key not in data]

    if missing:
        print(f"Skipping {npz_path}: missing keys {missing}")
        return "failed"

    positions = data["positions"].astype(int)
    goals = data["goals"].astype(int)
    obstacles = data["obstacles"].astype(int)
    segment_diffs = data["segment_diffs"].astype(int)

    num_steps = positions.shape[0]
    num_segments = len(segment_diffs)

    if num_segments < 2:
        print(f"Skipping {npz_path}: need at least 2 segments, found {num_segments}")
        return "failed"

    segment_ranges = infer_segment_ranges(num_steps, num_segments)

    features, feature_names = extract_segment_features(
        positions=positions,
        goals=goals,
        obstacles=obstacles,
        segment_ranges=segment_ranges,
    )

    scores = score_segments(features, model=model)
    pair_info = choose_best_pair(features, scores)

    if pair_info is None:
        print(f"Skipping {npz_path}: no valid candidate pairs")
        return "failed"

    print("\n" + "=" * 80)
    print(f"Episode: {scenario_id}")
    print(f"File: {npz_path}")
    print(f"Num steps: {num_steps}")
    print(f"Num segments: {num_segments}")
    print(f"Segment diffs: {segment_diffs.tolist()}")
    print()
    print(
        "Selected pair:",
        f"A={pair_info['segment_a']},",
        f"B={pair_info['segment_b']},",
        f"EIG proxy={pair_info['eig_proxy']:.4f}",
    )

    show_pair(
        npz_path=npz_path,
        positions=positions,
        goals=goals,
        obstacles=obstacles,
        segment_ranges=segment_ranges,
        segment_diffs=segment_diffs,
        pair_info=pair_info,
    )

    answer = prompt_user_for_label()

    if answer == "q":
        return "quit"

    if answer in {"u", "s"}:
        chosen_worse_segment = None
        label = "unsure_or_skipped"
    elif answer == "a":
        chosen_worse_segment = pair_info["segment_a"]
        label = "a_worse"
    elif answer == "b":
        chosen_worse_segment = pair_info["segment_b"]
        label = "b_worse"
    else:
        raise RuntimeError(f"Unexpected answer: {answer}")

    a = pair_info["segment_a"]
    b = pair_info["segment_b"]

    entry = {
        "scenario_id": scenario_id,
        "npz_path": str(Path(npz_path)),
        "segment_a": int(a),
        "segment_b": int(b),
        "chosen_worse_segment": (
            int(chosen_worse_segment)
            if chosen_worse_segment is not None
            else None
        ),
        "label": label,
        "segment_a_range": [
            int(segment_ranges[a][0]),
            int(segment_ranges[a][1]),
        ],
        "segment_b_range": [
            int(segment_ranges[b][0]),
            int(segment_ranges[b][1]),
        ],
        "segment_a_diff": int(segment_diffs[a]),
        "segment_b_diff": int(segment_diffs[b]),
        "query_metadata": {
            "model_score_a": pair_info["model_score_a"],
            "model_score_b": pair_info["model_score_b"],
            "preference_probability_a_worse": pair_info[
                "preference_probability_a_worse"
            ],
            "entropy": pair_info["entropy"],
            "feature_distance": pair_info["feature_distance"],
            "eig_proxy": pair_info["eig_proxy"],
            "feature_names": feature_names,
            "feature_a": features[a].astype(float).tolist(),
            "feature_b": features[b].astype(float).tolist(),
        },
    }

    append_annotation(output_path, entry)
    print(f"Saved annotation to {output_path}")

    return "labeled"


def run_query_loop(
    input_folder,
    output_path="annotation_elicitation.json",
    model_path=None,
    recursive=False,
    skip_already_labeled=True,
):
    input_folder = Path(input_folder)
    output_path = Path(output_path)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    npz_files = sorted(
        input_folder.rglob("*.npz") if recursive else input_folder.glob("*.npz")
    )

    if not npz_files:
        print(f"No .npz files found in {input_folder}")
        return

    model = load_torch_model(model_path)

    print(f"Found {len(npz_files)} .npz files")
    print(f"Writing annotations to {output_path}")

    labeled = 0
    skipped = 0
    failed = 0

    for npz_path in npz_files:
        result = process_npz(
            npz_path=npz_path,
            output_path=output_path,
            model=model,
            skip_already_labeled=skip_already_labeled,
        )

        if result == "quit":
            print("Stopping query loop.")
            break
        elif result == "labeled":
            labeled += 1
        elif result == "skipped_existing":
            skipped += 1
        else:
            failed += 1

    print()
    print("Done.")
    print(f"Labeled: {labeled}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed/skipped invalid: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactively query segment-pair congestion labels from .npz MAPF episodes."
    )

    parser.add_argument(
        "input_folder",
        help="Folder containing .npz files.",
    )

    parser.add_argument(
        "--output",
        default="annotation_elicitation.json",
        help="Path to output JSON file.",
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional PyTorch model path. If omitted, scores default to 0.",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search recursively for .npz files.",
    )

    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Do not skip episodes that already have an annotation in the output JSON.",
    )

    args = parser.parse_args()

    run_query_loop(
        input_folder=args.input_folder,
        output_path=args.output,
        model_path=args.model_path,
        recursive=args.recursive,
        skip_already_labeled=not args.relabel,
    )


if __name__ == "__main__":
    main()
