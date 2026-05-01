"""
Compare segment-picking heuristics against the CNN's val top-3 accuracy.

If a simple heuristic matches the CNN, the CNN isn't finding much beyond
that signal. If the CNN clearly beats them, it's learning real
spatiotemporal structure.

Usage:
    python evaluate_baselines.py --data dataset/held_out
    python evaluate_baselines.py --data dataset/held_out --cnn out/segment_classifier.pt
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from held_out_seed_set import STEPS_DELTA, VAL_MAP_SEEDS


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
# Top-1 ±1 accuracy: model's argmax pick within ±1 of argmax(segment_diffs)
# ---------------------------------------------------------------------------

def pm1_acc(scorer, episode_paths, map_seed_filter, tolerance: int = 1):
    correct = total = 0
    for path in sorted(episode_paths):
        data = np.load(str(path), allow_pickle=True)
        if int(data["map_seed"]) not in map_seed_filter:
            continue
        diffs = data["segment_diffs"]
        S = len(diffs)
        if S < 2:
            continue
        scores = np.array([scorer(data, s) for s in range(S)])
        true_best = int(np.argmax(diffs))
        pred_best = int(np.argmax(scores))
        correct += int(abs(pred_best - true_best) <= tolerance)
        total += 1
    return correct / total if total > 0 else 0.0, total


def cnn_pm1_acc(model_path, episode_paths, map_seed_filter, device, tolerance: int = 1):
    from train_segment_classifier import Segment3DCNN, featurize_segment
    ckpt = torch.load(model_path, map_location=device)
    model = Segment3DCNN(in_channels=ckpt["in_channels"], base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ctx = ckpt.get("context_segments", 1)

    correct = total = 0
    with torch.no_grad():
        for path in sorted(episode_paths):
            data = np.load(str(path), allow_pickle=True)
            if int(data["map_seed"]) not in map_seed_filter:
                continue
            diffs = data["segment_diffs"]
            S = len(diffs)
            if S < 2:
                continue
            feats = np.stack([
                featurize_segment(data["obstacles"], data["positions"], data["goals"], s, context_segments=ctx)
                for s in range(S)
            ])
            scores = model(torch.from_numpy(feats).to(device)).cpu().numpy()
            true_best = int(np.argmax(diffs))
            pred_best = int(np.argmax(scores))
            correct += int(abs(pred_best - true_best) <= tolerance)
            total += 1
    return correct / total if total > 0 else 0.0, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cnn", default=None, help="Optional path to trained CNN checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=int, default=1,
                        help="Pass if predicted top is within ±tolerance of true_best (default: 1)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    episode_paths = list(Path(args.data).rglob("*.npz"))

    heuristics = [
        ("Random",                    random_score),
        ("Middle-segment",            middle_segment_score),
        ("Agents-stuck count",        stuck_score),
        ("Lack-of-progress",          no_progress_score),
        ("Local-density (3x3 max)",   local_density_score),
        ("Stuck + no-progress",       stuck_plus_no_progress),
    ]

    print(f"Val map seeds: {sorted(VAL_MAP_SEEDS)}")
    print(f"Metric: top-1 ±{args.tolerance} (DDG-aligned)")
    print()
    print(f"  {'Heuristic':<28} acc         (n)")
    print(f"  {'-'*28} ----------- -----")

    n_total = None
    for name, fn in heuristics:
        acc, n = pm1_acc(fn, episode_paths, VAL_MAP_SEEDS, tolerance=args.tolerance)
        n_total = n
        print(f"  {name:<28} {acc:.3f}       {n}")

    if args.cnn:
        acc, _ = cnn_pm1_acc(args.cnn, episode_paths, VAL_MAP_SEEDS, args.device, tolerance=args.tolerance)
        print(f"  {'CNN (' + Path(args.cnn).name + ')':<28} {acc:.3f}       {n_total}")


if __name__ == "__main__":
    main()
