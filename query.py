
import argparse
import json
import math
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np


EPS = 1e-8
SEGMENT_LEN = 16


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def binary_entropy(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def collect_npz_files(input_folder, manifest, recursive):
    """Return the list of .npz paths to consider, sourced from either a manifest
    file (one path per line) or a folder walk. Exactly one source must be given."""
    if (input_folder is None) == (manifest is None):
        raise ValueError("Provide exactly one of input_folder or --manifest.")

    if manifest is not None:
        manifest = Path(manifest)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {manifest}")
        paths = []
        for raw in manifest.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line))
        return paths

    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    return sorted(
        input_folder.rglob("*.npz") if recursive else input_folder.glob("*.npz")
    )


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


def already_labeled_pair(annotations, scenario_id, seg_a, seg_b):
    """True iff the exact (scenario_id, {seg_a, seg_b}) pair has been queried before."""
    target = frozenset((int(seg_a), int(seg_b)))
    for ann in annotations:
        if ann.get("scenario_id") != scenario_id:
            continue
        ann_a, ann_b = ann.get("segment_a"), ann.get("segment_b")
        if ann_a is None or ann_b is None:
            continue
        if frozenset((int(ann_a), int(ann_b))) == target:
            return True
    return False


def labels_per_scenario(annotations):
    counts = {}
    for ann in annotations:
        sid = ann.get("scenario_id")
        if sid is not None:
            counts[sid] = counts.get(sid, 0) + 1
    return counts


# Segment handling


def compute_segment_ranges(num_segments):
    return [
        (i * SEGMENT_LEN, (i + 1) * SEGMENT_LEN - 1)
        for i in range(num_segments)
    ]


def extract_segment_arrays(positions, segment_range):
    start, end = segment_range
    return positions[start:end + 1]

# Feature extraction


def shortest_path_overlap_proxy(segment_positions, goals):
    """
    Approximate shortest-path overlap / shared corridor pressure.

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


# For SHANE: model scoring

def load_torch_model(model_path):
    """

    Expects a PyTorch model saved with torch.save(model, path), where:
        model(features_tensor) -> scores

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


# Candidate pair selection
def build_candidate_pairs(num_segments):
    return list(combinations(range(num_segments), 2))


# Pool-mode helpers
# -----------------
# In pool mode we featurize every segment of every eligible episode up front,
# then rank all (episode, segA, segB) combos by entropy * distance with
# *globally* normalized features. The greedy selector pops the top-ranked
# candidate, accepts it unless the episode has hit its per-episode cap, and
# stops when the label budget is exhausted.

def load_episode_for_pool(npz_path):
    """Load + featurize one episode for pool-based selection.

    Returns a dict with everything needed to render and label the picked pairs
    later, or None if the episode is unusable.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  skip {npz_path}: load failed ({e})")
        return None

    required_keys = ["positions", "goals", "obstacles", "segment_diffs"]
    if any(k not in data for k in required_keys):
        return None

    positions = data["positions"].astype(int)
    goals = data["goals"].astype(int)
    obstacles = data["obstacles"].astype(int)
    segment_diffs = data["segment_diffs"].astype(int)

    num_segments = len(segment_diffs)
    if num_segments < 2:
        return None

    segment_ranges = compute_segment_ranges(num_segments)
    features, feature_names = extract_segment_features(
        positions=positions,
        goals=goals,
        obstacles=obstacles,
        segment_ranges=segment_ranges,
    )

    return {
        "npz_path": npz_path,
        "scenario_id": Path(npz_path).stem,
        "positions": positions,
        "goals": goals,
        "obstacles": obstacles,
        "segment_diffs": segment_diffs,
        "segment_ranges": segment_ranges,
        "features": features,
        "feature_names": feature_names,
    }


def gather_pool_candidates(episodes, model, prior_annotations, random_mode=False, rng=None):
    """Score every pair across every episode against globally-normalized features.

    Returns a list of dicts. By default, sorted by eig_proxy descending (active
    learning). If `random_mode=True`, the list is shuffled instead — used as
    the random-sampling baseline that shares everything with AL except the
    selection criterion. Pairs already present in prior_annotations are excluded.
    """
    import random as _random
    if not episodes:
        return []

    # Per-episode model scores. Score in one pass per episode rather than one
    # giant tensor since episode-level shapes match the model's expectation.
    per_episode_scores = [score_segments(ep["features"], model=model) for ep in episodes]

    # Global feature normalization: stack all segment features and normalize
    # against the joint mean/std. This makes ||phi_A - phi_B|| comparable
    # across episodes, which is the whole point of pool mode.
    all_features = np.concatenate([ep["features"] for ep in episodes], axis=0)
    mean = all_features.mean(axis=0, keepdims=True)
    std = all_features.std(axis=0, keepdims=True)
    normalizer = lambda f: (f - mean) / (std + EPS)

    candidates = []
    for ep_idx, ep in enumerate(episodes):
        scores = per_episode_scores[ep_idx]
        norm_features = normalizer(ep["features"])
        for a, b in build_candidate_pairs(len(ep["features"])):
            if already_labeled_pair(prior_annotations, ep["scenario_id"], a, b):
                continue
            p = sigmoid(scores[a] - scores[b])
            entropy = binary_entropy(p)
            distance = float(np.linalg.norm(norm_features[a] - norm_features[b]))
            eig_proxy = float(entropy * distance)
            candidates.append({
                "episode_idx": ep_idx,
                "scenario_id": ep["scenario_id"],
                "segment_a": int(a),
                "segment_b": int(b),
                "model_score_a": float(scores[a]),
                "model_score_b": float(scores[b]),
                "preference_probability_a_worse": float(p),
                "entropy": float(entropy),
                "feature_distance": distance,
                "eig_proxy": eig_proxy,
            })

    if random_mode:
        (rng or _random).shuffle(candidates)
    else:
        candidates.sort(key=lambda c: c["eig_proxy"], reverse=True)
    return candidates


def select_pool_with_cap(candidates, budget, per_episode_cap, prior_counts):
    """Greedy top-K selection respecting a per-episode cap.

    prior_counts already counts labels written in earlier sessions, so resuming
    a partial run still honors the cap.
    """
    selected = []
    counts = dict(prior_counts)
    for cand in candidates:
        if budget is not None and len(selected) >= budget:
            break
        sid = cand["scenario_id"]
        if per_episode_cap is not None and counts.get(sid, 0) >= per_episode_cap:
            continue
        selected.append(cand)
        counts[sid] = counts.get(sid, 0) + 1
    return selected


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


# Visualization
# ------------
# Each labelling step renders the two segments as animated SVGs (via pogema's
# AnimationDrawer) and embeds them side-by-side in a tiny HTML page. The browser
# tab is opened once per session and rewritten between pairs — refresh to see
# the next pair. Heavy pogema/cppimport deps are imported lazily so query.py's
# top-level still loads in environments that only need the static-feature path.

_HTML_PATH = Path(tempfile.gettempdir()) / "query_current.html"
_SVG_DIR = _HTML_PATH.parent
_BROWSER_OPENED = False


def _render_segment_svg(npz_path, segment_range, output_path):
    """Slice positions to segment_range and emit an animated SVG to output_path."""
    import cppimport.import_hook  # noqa: F401  required for pogema's compiled bits
    from pogema.wrappers.persistence import AgentState
    from finetuning.scenario_generators import (
        make_pogema_maze_instance,
        make_pogema_random_instance,
    )
    from utils.svg_utils import create_multi_animation

    data = np.load(str(npz_path), allow_pickle=True)
    positions = data["positions"]
    goals = data["goals"]
    obstacles = data["obstacles"]
    map_type = str(data["map_type"])
    map_seed = int(data["map_seed"])
    scenario_seed = int(data["scenario_seed"])
    num_agents = int(data["num_agents"])

    make_env = make_pogema_maze_instance if map_type == "maze" else make_pogema_random_instance
    env = make_env(num_agents=num_agents, map_seed=map_seed, scenario_seed=scenario_seed)
    grid_config = env.grid_config
    wr = grid_config.obs_radius - 1

    start, end = segment_range
    T_seg = end - start + 1
    N = positions.shape[1]

    history = []
    for n in range(N):
        goal_r, goal_c = int(goals[n][0]) - wr, int(goals[n][1]) - wr
        states = []
        for t_seg in range(T_seg):
            t_abs = start + t_seg
            r, c = int(positions[t_abs][n][0]) - wr, int(positions[t_abs][n][1]) - wr
            states.append(AgentState(x=r, y=c, tx=goal_r, ty=goal_c, step=t_seg, active=True))
        history.append(states)

    create_multi_animation(obstacles, [history], grid_config, name=str(output_path))


def show_pair(npz_path, positions, goals, obstacles, segment_ranges, segment_diffs, pair_info):
    """Render segment A and B to animated SVGs and embed both in a side-by-side
    HTML page. Browser opens once per session at _HTML_PATH; subsequent calls
    overwrite the file in place — refresh the tab to advance.

    Signature kept compatible with the prior matplotlib version. positions,
    goals, obstacles aren't used directly (the SVG renderer re-opens the npz
    to get pogema env metadata) but are accepted for backwards compatibility.
    """
    global _BROWSER_OPENED

    a = pair_info["segment_a"]
    b = pair_info["segment_b"]
    a_start, a_end = segment_ranges[a]
    b_start, b_end = segment_ranges[b]
    a_diff = int(segment_diffs[a]) if segment_diffs is not None else None
    b_diff = int(segment_diffs[b]) if segment_diffs is not None else None

    svg_a = _SVG_DIR / "query_segment_a.svg"
    svg_b = _SVG_DIR / "query_segment_b.svg"
    _render_segment_svg(npz_path, segment_ranges[a], svg_a)
    _render_segment_svg(npz_path, segment_ranges[b], svg_b)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>query — {Path(npz_path).name}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #f5f5f5; margin: 16px; }}
.summary {{ background: white; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; }}
.summary code {{ background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
.row {{ display: flex; gap: 16px; }}
.col {{ flex: 1; background: white; padding: 12px; border-radius: 6px; }}
h2 {{ margin: 0 0 8px; font-size: 14px; font-weight: 600; }}
object {{ width: 100%; height: auto; display: block; }}
.hint {{ color: #888; font-style: italic; margin-top: 4px; }}
</style></head><body>
<div class="summary">
  <b>{Path(npz_path).name}</b> &middot;
  EIG proxy <b>{pair_info['eig_proxy']:.4f}</b> &middot;
  p(A worse) <b>{pair_info['preference_probability_a_worse']:.3f}</b> &middot;
  H(p) <b>{pair_info['entropy']:.3f}</b> &middot;
  ‖φ_A − φ_B‖ <b>{pair_info['feature_distance']:.3f}</b>
  <div class="hint">watch both, then return to the terminal and press <code>a</code> / <code>b</code> / <code>u</code> / <code>s</code> / <code>q</code></div>
</div>
<div class="row">
  <div class="col">
    <h2>Segment A — t={a_start}-{a_end} &middot; diff={a_diff}</h2>
    <object type="image/svg+xml" data="{svg_a.name}"></object>
  </div>
  <div class="col">
    <h2>Segment B — t={b_start}-{b_end} &middot; diff={b_diff}</h2>
    <object type="image/svg+xml" data="{svg_b.name}"></object>
  </div>
</div>
</body></html>"""
    _HTML_PATH.write_text(html)

    if not _BROWSER_OPENED:
        import webbrowser
        webbrowser.open(f"file://{_HTML_PATH.resolve()}")
        print(f"  (browser opened at {_HTML_PATH}; refresh between pairs)")
        _BROWSER_OPENED = True
    else:
        print(f"  (refresh {_HTML_PATH} to see this pair)")


# Interactive query loop

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


def label_one_pair(episode, pair_info, output_path, feature_names):
    """Show the pair, ask the human, append the annotation. Returns the user's
    raw answer ('a'/'b'/'u'/'s'/'q') so callers can decide whether to keep going."""
    npz_path = episode["npz_path"]
    positions = episode["positions"]
    goals = episode["goals"]
    obstacles = episode["obstacles"]
    segment_diffs = episode["segment_diffs"]
    segment_ranges = episode["segment_ranges"]
    features = episode["features"]

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
        return answer

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
        "scenario_id": episode["scenario_id"],
        "npz_path": str(Path(npz_path)),
        "segment_a": int(a),
        "segment_b": int(b),
        "chosen_worse_segment": (
            int(chosen_worse_segment) if chosen_worse_segment is not None else None
        ),
        "label": label,
        "segment_a_range": [int(segment_ranges[a][0]), int(segment_ranges[a][1])],
        "segment_b_range": [int(segment_ranges[b][0]), int(segment_ranges[b][1])],
        "segment_a_diff": int(segment_diffs[a]),
        "segment_b_diff": int(segment_diffs[b]),
        "query_metadata": {
            "model_score_a": pair_info["model_score_a"],
            "model_score_b": pair_info["model_score_b"],
            "preference_probability_a_worse": pair_info["preference_probability_a_worse"],
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
    return answer


def process_npz(npz_path, output_path, model=None, skip_already_labeled=True):
    annotations = load_existing_annotations(output_path)
    scenario_id = Path(npz_path).stem

    if skip_already_labeled and already_labeled_episode(annotations, scenario_id):
        print(f"Skipping already-labeled episode: {scenario_id}")
        return "skipped_existing"

    episode = load_episode_for_pool(npz_path)
    if episode is None:
        print(f"Skipping {npz_path}: missing keys, <2 segments, or load error")
        return "failed"

    scores = score_segments(episode["features"], model=model)
    pair_info = choose_best_pair(episode["features"], scores)
    if pair_info is None:
        print(f"Skipping {npz_path}: no valid candidate pairs")
        return "failed"

    print("\n" + "=" * 80)
    print(f"Episode: {scenario_id}")
    print(f"File: {npz_path}")
    print(f"Num steps: {episode['positions'].shape[0]}")
    print(f"Num segments: {len(episode['segment_diffs'])}")
    print(f"Segment diffs: {episode['segment_diffs'].tolist()}")
    print()
    print(
        "Selected pair:",
        f"A={pair_info['segment_a']},",
        f"B={pair_info['segment_b']},",
        f"EIG proxy={pair_info['eig_proxy']:.4f}",
    )

    answer = label_one_pair(episode, pair_info, output_path, episode["feature_names"])
    if answer == "q":
        return "quit"
    return "labeled"


def run_query_loop(
    input_folder,
    output_path="annotation_elicitation.json",
    model_path=None,
    recursive=False,
    skip_already_labeled=True,
    manifest=None,
):
    output_path = Path(output_path)
    npz_files = collect_npz_files(input_folder, manifest, recursive)

    if not npz_files:
        src = manifest if manifest is not None else input_folder
        print(f"No .npz files found via {src}")
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


def run_pool_query_loop(
    input_folder,
    output_path,
    model_path,
    budget,
    per_episode_cap,
    recursive=False,
    manifest=None,
    random_mode=False,
    seed=None,
):
    """Pool-based selection over (episode, segA, segB) candidates.

    Default (active learning): rank candidates by H(p)*||phi_A-phi_B|| against
    globally-normalized features, greedy-pick the top `budget` respecting
    `per_episode_cap`, then walk the human through the picked pairs.

    `random_mode=True`: shuffle the candidate pool instead of ranking — used as
    the random-sampling baseline (Stage 2 in the unified comparison). Same
    one-pair-at-a-time labelling protocol; only the selection criterion changes.
    `seed` makes the random choice reproducible across re-runs.

    Resumable: prior annotations in `output_path` exclude their exact pair from
    the pool and count toward each episode's cap.
    """
    import random as _random
    rng = _random.Random(seed) if seed is not None else None
    output_path = Path(output_path)
    npz_files = collect_npz_files(input_folder, manifest, recursive)
    if not npz_files:
        src = manifest if manifest is not None else input_folder
        print(f"No .npz files found via {src}")
        return

    model = load_torch_model(model_path)
    prior_annotations = load_existing_annotations(output_path)
    prior_counts = labels_per_scenario(prior_annotations)
    remaining_budget = budget - sum(prior_counts.values()) if budget is not None else None

    print(f"Pool mode: budget={budget}, per_episode_cap={per_episode_cap}")
    print(f"Found {len(npz_files)} .npz files")
    if prior_annotations:
        print(f"Prior annotations: {len(prior_annotations)} across {len(prior_counts)} episodes")
        if remaining_budget is not None:
            print(f"Remaining budget: {remaining_budget}")
    if remaining_budget is not None and remaining_budget <= 0:
        print("Budget already exhausted by prior annotations; nothing to do.")
        return

    print("Loading + featurizing episodes...")
    episodes = []
    skipped_load = 0
    for npz_path in npz_files:
        ep = load_episode_for_pool(npz_path)
        if ep is None:
            skipped_load += 1
            continue
        # Skip episodes already at cap so we don't waste compute on their pairs.
        if (
            per_episode_cap is not None
            and prior_counts.get(ep["scenario_id"], 0) >= per_episode_cap
        ):
            continue
        episodes.append(ep)
    print(f"  loaded {len(episodes)} eligible episodes ({skipped_load} skipped at load)")

    if not episodes:
        print("No eligible episodes; nothing to do.")
        return

    feature_names = episodes[0]["feature_names"]

    print("Sampling candidate pairs..." if random_mode else "Ranking candidate pairs...")
    candidates = gather_pool_candidates(
        episodes, model, prior_annotations,
        random_mode=random_mode, rng=rng,
    )
    print(f"  {len(candidates)} candidate pairs")

    selected = select_pool_with_cap(candidates, remaining_budget, per_episode_cap, prior_counts)
    print(f"  selected {len(selected)} pairs from {len(set(c['scenario_id'] for c in selected))} episodes")
    print(f"Writing annotations to {output_path}")

    labeled = 0
    quit_early = False
    for i, cand in enumerate(selected, start=1):
        ep = episodes[cand["episode_idx"]]
        print("\n" + "=" * 80)
        print(f"[{i}/{len(selected)}] Episode: {cand['scenario_id']}")
        print(f"File: {ep['npz_path']}")
        print(f"Num segments: {len(ep['segment_diffs'])}")
        print(f"Segment diffs: {ep['segment_diffs'].tolist()}")
        print(
            "Selected pair:",
            f"A={cand['segment_a']},",
            f"B={cand['segment_b']},",
            f"EIG proxy={cand['eig_proxy']:.4f}",
            f"(entropy={cand['entropy']:.3f}, distance={cand['feature_distance']:.3f})",
        )
        answer = label_one_pair(ep, cand, output_path, feature_names)
        if answer == "q":
            quit_early = True
            break
        if answer in {"a", "b"}:
            labeled += 1

    print()
    print("Done." if not quit_early else "Stopped early.")
    print(f"Labeled this session: {labeled}")
    print(f"Total annotations on disk: {len(load_existing_annotations(output_path))}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactively query segment-pair congestion labels from .npz MAPF episodes."
    )

    parser.add_argument(
        "input_folder",
        nargs="?",
        default=None,
        help="Folder containing .npz files. Mutually exclusive with --manifest.",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a manifest file (one .npz path per line; blank lines and "
             "lines starting with '#' are ignored). Used in place of input_folder "
             "to define the elicitation pool — useful for restricting Stage 3/4 "
             "to a curated 'filtered' subset of ckpt_*/ rollouts without "
             "physically copying files into a separate directory.",
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

    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Pool-based active learning: total label budget. When set, all "
             "(episode, segment_a, segment_b) pairs are pool-ranked by "
             "H(p)*||phi_A-phi_B|| with globally-normalized features, then the "
             "top --budget pairs are queried (subject to --per-episode-cap). "
             "If unset, falls back to the per-episode top-1 query loop.",
    )

    parser.add_argument(
        "--per-episode-cap",
        type=int,
        default=None,
        help="Pool mode: max queries per episode (default: no cap). Combine "
             "with --budget to balance budget efficiency against episode "
             "concentration risk.",
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Pool mode: skip the H(p)*||phi_A-phi_B|| acquisition and "
             "uniformly sample candidates instead. Same one-pair-at-a-time "
             "labelling protocol — used as the random-sampling baseline "
             "against which AL acquisition is judged. Requires --budget.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random mode (makes the sampled subset "
             "reproducible across re-runs).",
    )

    args = parser.parse_args()

    if args.input_folder is None and args.manifest is None:
        parser.error("must provide either input_folder or --manifest")
    if args.random and args.budget is None:
        parser.error("--random requires --budget (random sampling without a "
                     "budget would label every candidate pair).")

    in_pool_mode = (
        args.budget is not None or args.per_episode_cap is not None or args.random
    )
    if in_pool_mode:
        run_pool_query_loop(
            input_folder=args.input_folder,
            output_path=args.output,
            model_path=args.model_path,
            budget=args.budget,
            per_episode_cap=args.per_episode_cap,
            recursive=args.recursive,
            manifest=args.manifest,
            random_mode=args.random,
            seed=args.seed,
        )
    else:
        run_query_loop(
            input_folder=args.input_folder,
            output_path=args.output,
            model_path=args.model_path,
            recursive=args.recursive,
            skip_already_labeled=not args.relabel,
            manifest=args.manifest,
        )


if __name__ == "__main__":
    main()
