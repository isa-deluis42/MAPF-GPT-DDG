"""
Compare segment-picking heuristics against trained CNN checkpoints, reporting
the same four per-epoch metrics emitted by train_segment_classifier.py:

  • top-1±1   — argmax(score) within ±1 of argmax(segment_diffs)        [VAL_MAP_SEEDS]
  • top-3     — argmax(segment_diffs) is in the top-3 scored segments    [VAL_MAP_SEEDS]
  • hum-tr    — pairwise (worst > clean) on --annotations                [training-label sanity]
  • hum-val   — pairwise (worst > clean) on --val-annotations            [held-out human signal]

If a simple heuristic matches the CNN, the CNN isn't finding much beyond
that signal. If the CNN clearly beats them, it's learning real
spatiotemporal structure.

Usage:
    # Heuristics only, auto-label metrics
    python evaluate_baselines.py --data dataset/held_out

    # Heuristics + multiple checkpoints, all four metrics
    python evaluate_baselines.py --data dataset/held_out \\
        --cnn out/segment_classifier/baseline.pt \\
              out/segment_classifier/round_1.pt \\
              out/segment_classifier/round_2.pt \\
              out/segment_classifier/round_3.pt \\
        --annotations annotation_iterative.json \\
        --val-annotations annotation_val_map.json
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from held_out_seed_set import STEPS_DELTA, VAL_MAP_SEEDS
from train_segment_classifier import (
    Segment3DCNN,
    featurize_segment,
    load_annotations,
)


# ---------------------------------------------------------------------------
# Heuristics — each takes (data_dict, segment_idx) and returns a scalar score
# ---------------------------------------------------------------------------

def random_score(data, s):
    return np.random.rand()


def middle_segment_score(data, s):
    S = len(data["segment_diffs"])
    return -abs(s - S / 2)


def stuck_score(data, s):
    """Count (agent, timestep) pairs where the agent didn't move."""
    pos = data["positions"]
    a = s * STEPS_DELTA
    b = min(a + STEPS_DELTA, len(pos))
    if b - a < 2:
        return 0
    seg = pos[a:b]
    return int(np.sum(np.all(seg[1:] == seg[:-1], axis=-1)))


def no_progress_score(data, s):
    """Negated mean Manhattan progress toward goals across the segment."""
    pos = data["positions"]
    goals = data["goals"]
    a = s * STEPS_DELTA
    b = min(a + STEPS_DELTA, len(pos)) - 1
    if b <= a:
        return 0.0
    start_dist = np.abs(pos[a] - goals).sum(axis=-1)
    end_dist = np.abs(pos[b] - goals).sum(axis=-1)
    return float(-(start_dist - end_dist).mean())


def local_density_score(data, s):
    """Max agents in any 3x3 spatial window, summed across timesteps."""
    pos = data["positions"]
    obstacles = data["obstacles"]
    H, W = obstacles.shape
    a = s * STEPS_DELTA
    b = min(a + STEPS_DELTA, len(pos))
    total = 0.0
    for t in range(a, b):
        grid = np.zeros((H, W), dtype=np.float32)
        for xy in pos[t]:
            r, c = int(xy[0]), int(xy[1])
            if 0 <= r < H and 0 <= c < W:
                grid[r, c] += 1.0
        kernel = np.ones((3, 3), dtype=np.float32)
        from scipy.signal import convolve2d
        local = convolve2d(grid, kernel, mode="same")
        total += float(local.max())
    return total


def stuck_plus_no_progress(data, s):
    return stuck_score(data, s) + 5.0 * no_progress_score(data, s)


# ---------------------------------------------------------------------------
# Score-function adapters: data → np.array of length S with one score per segment.
# Auto and pairwise metrics share this interface so heuristics and the CNN go
# through the same evaluator code paths.
# ---------------------------------------------------------------------------

ScoreFn = Callable[[dict], np.ndarray]


def heuristic_score_fn(per_segment) -> ScoreFn:
    def fn(data):
        S = len(data["segment_diffs"])
        return np.array([per_segment(data, s) for s in range(S)], dtype=np.float64)
    return fn


def cnn_score_fn(model: Segment3DCNN, context_segments: int, device: str) -> ScoreFn:
    @torch.no_grad()
    def fn(data):
        S = len(data["segment_diffs"])
        feats = np.stack([
            featurize_segment(
                data["obstacles"], data["positions"], data["goals"], s,
                context_segments=context_segments,
            )
            for s in range(S)
        ])
        return model(torch.from_numpy(feats).to(device)).cpu().numpy()
    return fn


# ---------------------------------------------------------------------------
# Metrics — mirror evaluate_metrics + evaluate_human_pairs in
# train_segment_classifier.py so numbers are directly comparable to the
# trainer's per-epoch log lines.
# ---------------------------------------------------------------------------

def auto_metrics(score_fn: ScoreFn, episode_paths, map_seed_filter, tolerance: int = 1):
    """Returns (top1_pm1, top3, n_episodes) on episodes whose map_seed ∈ filter."""
    pm1 = top3_hits = total = 0
    for path in sorted(episode_paths):
        data = np.load(str(path), allow_pickle=True)
        if int(data["map_seed"]) not in map_seed_filter:
            continue
        diffs = data["segment_diffs"]
        S = len(diffs)
        if S < 2:
            continue
        scores = score_fn(data)
        true_best = int(np.argmax(diffs))
        pred_best = int(np.argmax(scores))
        top3 = set(np.argsort(scores)[-3:].tolist())
        pm1 += int(abs(pred_best - true_best) <= tolerance)
        top3_hits += int(true_best in top3)
        total += 1
    if total == 0:
        return 0.0, 0.0, 0
    return pm1 / total, top3_hits / total, total


def pairwise_acc(
    score_fn: ScoreFn,
    annotations: Dict[str, List[Tuple[int, int]]],
    context_segments: int = 1,
):
    """Pairwise (worst > clean) accuracy across all labeled pairs.

    Mirrors evaluate_human_pairs in train_segment_classifier: out-of-window
    indices (segment_idx ≥ S - (ctx-1)) are silently skipped, and rollouts
    whose npz is missing on disk are skipped too.
    """
    correct = total = 0
    for path_key, pair_list in annotations.items():
        path = Path(path_key)
        if not path.exists():
            continue
        data = np.load(str(path), allow_pickle=True)
        S = len(data["segment_diffs"]) - (context_segments - 1)
        scores = score_fn(data)
        for wi, ci in pair_list:
            if not (0 <= wi < S and 0 <= ci < S):
                continue
            correct += int(scores[wi] > scores[ci])
            total += 1
    return (correct / total if total else 0.0), total


# ---------------------------------------------------------------------------
# Episode filtering (mirrors --min_checkpoint in train_segment_classifier.py)
# ---------------------------------------------------------------------------

def filter_min_checkpoint(paths: List[Path], min_checkpoint: int) -> List[Path]:
    if min_checkpoint <= 0:
        return paths
    out = []
    for p in paths:
        ckpt_dir = p.parent.name  # e.g. 'ckpt_500'
        if ckpt_dir.startswith("ckpt_"):
            try:
                if int(ckpt_dir[5:]) >= min_checkpoint:
                    out.append(p)
            except ValueError:
                out.append(p)
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cnn", nargs="+", default=[],
                        help="One or more CNN checkpoint paths to evaluate alongside the heuristics.")
    parser.add_argument("--annotations", default=None,
                        help="Train-map labels file (--annotations in the trainer); reported as hum-tr.")
    parser.add_argument("--val-annotations", default=None,
                        help="Val-map labels file (--val-annotations in the trainer); reported as hum-val.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=int, default=1,
                        help="Top-1 ±tolerance window for the auto metric (default: 1, matches trainer).")
    parser.add_argument("--min_checkpoint", type=int, default=0,
                        help="Skip episodes from MAPF-GPT ckpt_*/ folders below this iter (default: 0 = use all).")
    args = parser.parse_args()

    np.random.seed(args.seed)

    episode_paths = list(Path(args.data).rglob("*.npz"))
    episode_paths = filter_min_checkpoint(episode_paths, args.min_checkpoint)

    train_annotations: Dict[str, List[Tuple[int, int]]] = {}
    val_annotations: Dict[str, List[Tuple[int, int]]] = {}
    if args.annotations:
        train_annotations = load_annotations(args.annotations)
    if args.val_annotations:
        val_annotations = load_annotations(args.val_annotations)

    print(f"Episodes:           {len(episode_paths)} found under {args.data}")
    print(f"Val map seeds:      {sorted(VAL_MAP_SEEDS)}")
    if train_annotations:
        n_pairs = sum(len(p) for p in train_annotations.values())
        print(f"Train annotations:  {len(train_annotations)} rollouts, {n_pairs} pairs (--annotations)")
    if val_annotations:
        n_pairs = sum(len(p) for p in val_annotations.values())
        print(f"Val annotations:    {len(val_annotations)} rollouts, {n_pairs} pairs (--val-annotations)")
    print(f"Metric: top-1 ±{args.tolerance} (DDG-aligned) on val maps; pairwise on annotations")
    print()

    name_w = 36
    header = f"  {'Scorer':<{name_w}} {'top-1±1':>8} {'top-3':>8} {'hum-tr':>8} {'hum-val':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    auto_n_ref = None  # auto-metric episode count, for the legend below

    def report(name: str, score_fn: ScoreFn, ctx: int = 1):
        nonlocal auto_n_ref
        pm1, top3, n_ep = auto_metrics(score_fn, episode_paths, VAL_MAP_SEEDS, args.tolerance)
        auto_n_ref = n_ep
        cells = [f"{pm1:.3f}", f"{top3:.3f}"]
        if train_annotations:
            acc, _ = pairwise_acc(score_fn, train_annotations, ctx)
            cells.append(f"{acc:.3f}")
        else:
            cells.append("  -  ")
        if val_annotations:
            acc, _ = pairwise_acc(score_fn, val_annotations, ctx)
            cells.append(f"{acc:.3f}")
        else:
            cells.append("  -  ")
        print(f"  {name:<{name_w}} " + " ".join(c.rjust(8) for c in cells))

    heuristics = [
        ("Random",                    random_score),
        ("Middle-segment",            middle_segment_score),
        ("Agents-stuck count",        stuck_score),
        ("Lack-of-progress",          no_progress_score),
        ("Local-density (3x3 max)",   local_density_score),
        ("Stuck + no-progress",       stuck_plus_no_progress),
    ]
    for name, fn in heuristics:
        report(name, heuristic_score_fn(fn), ctx=1)

    for cnn_path in args.cnn:
        ckpt = torch.load(cnn_path, map_location=args.device, weights_only=False)
        model = Segment3DCNN(
            in_channels=ckpt.get("in_channels", 4),
            base_ch=ckpt["base_ch"],
        ).to(args.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        ctx = ckpt.get("context_segments", 1)
        save_by = ckpt.get("save_by", "?")
        report(f"CNN ({Path(cnn_path).name}) [save_by={save_by}]",
               cnn_score_fn(model, ctx, args.device), ctx=ctx)

    print()
    print(f"  n_episodes (val maps) = {auto_n_ref}")
    if train_annotations:
        print(f"  n_pairs (hum-tr)      = {sum(len(p) for p in train_annotations.values())}  [random ≈ 0.500]")
    if val_annotations:
        print(f"  n_pairs (hum-val)     = {sum(len(p) for p in val_annotations.values())}  [random ≈ 0.500]")


if __name__ == "__main__":
    main()
