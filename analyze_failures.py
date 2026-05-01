"""
Analyze top-3 failures of a trained segment classifier on val episodes.

Categorizes failures as:
  - Ambiguous: the diff margin between true_best and the kth-best diff is small,
    so even a perfect model would essentially tie.
  - Confident: true_best had a clear advantage in diff but the model still missed it.

Also breaks down failure rate by checkpoint_iter, num_agents, and map_type so you
can see which slices the model struggles on.

Usage:
    python analyze_failures.py --data dataset/held_out --cnn out/segment_classifier.pt --device mps
"""

import argparse
import numpy as np
import torch
from collections import Counter
from pathlib import Path

from held_out_seed_set import VAL_MAP_SEEDS
from train_segment_classifier import Segment3DCNN, featurize_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cnn", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=int, default=1,
                        help="Episode passes if model's top pick is within ±tolerance of true_best (default: 1)")
    parser.add_argument("--margin_threshold", type=float, default=1.0,
                        help="Failures with diff(true_best) - diff(2nd_best) >= this are 'confident'")
    parser.add_argument("--show", type=int, default=15, help="How many top confident failures to print")
    args = parser.parse_args()

    ckpt = torch.load(args.cnn, map_location=args.device)
    model = Segment3DCNN(in_channels=ckpt["in_channels"], base_ch=ckpt["base_ch"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ctx = ckpt.get("context_segments", 1)

    episode_paths = sorted(Path(args.data).rglob("*.npz"))

    successes, failures = [], []

    with torch.no_grad():
        for path in episode_paths:
            data = np.load(str(path), allow_pickle=True)
            if int(data["map_seed"]) not in VAL_MAP_SEEDS:
                continue
            diffs = np.array(data["segment_diffs"], dtype=np.float32)
            S = len(diffs)
            if S < 2:
                continue

            feats = np.stack([
                featurize_segment(data["obstacles"], data["positions"], data["goals"], s, context_segments=ctx)
                for s in range(S)
            ])
            scores = model(torch.from_numpy(feats).to(args.device)).cpu().numpy()

            true_best = int(np.argmax(diffs))
            pred_best = int(np.argmax(scores))
            position_offset = abs(pred_best - true_best)
            top3_idx = set(np.argsort(scores)[-3:].tolist())
            ranking = np.argsort(-scores)
            true_best_rank = int(np.where(ranking == true_best)[0][0]) + 1  # 1-indexed

            sorted_diffs = np.sort(diffs)[::-1]
            margin = float(sorted_diffs[0] - sorted_diffs[1])  # diff(best) - diff(2nd best)

            entry = {
                "path": path,
                "map_type": str(data["map_type"]),
                "map_seed": int(data["map_seed"]),
                "scenario_seed": int(data["scenario_seed"]),
                "num_agents": int(data["num_agents"]),
                "checkpoint_iter": int(data["checkpoint_iter"]) if "checkpoint_iter" in data.files else -1,
                "S": S,
                "diffs": diffs.tolist(),
                "scores": scores.tolist(),
                "true_best": true_best,
                "pred_best": pred_best,
                "position_offset": position_offset,
                "true_best_rank": true_best_rank,
                "in_top3": true_best in top3_idx,
                "margin": margin,
                "passed": position_offset <= args.tolerance,
            }
            (successes if entry["passed"] else failures).append(entry)

    n_total = len(successes) + len(failures)
    n_top3 = sum(1 for e in successes + failures if e["in_top3"])
    print(f"\nTotal val episodes: {n_total}")
    print(f"top-1 ±{args.tolerance} (DDG metric): {len(successes)} / {n_total} = {len(successes)/n_total:.3f}")
    print(f"top-3 (legacy):              {n_top3} / {n_total} = {n_top3/n_total:.3f}")
    print(f"Failed (top-1 ±{args.tolerance}): {len(failures)} ({len(failures)/n_total:.1%})")

    confident = [f for f in failures if f["margin"] >= args.margin_threshold]
    ambiguous = [f for f in failures if f["margin"] < args.margin_threshold]
    print(f"\nFailure breakdown (margin threshold = {args.margin_threshold}):")
    print(f"  Confident (margin ≥ {args.margin_threshold}): {len(confident)} ({len(confident)/max(1,n_total):.1%} of all)")
    print(f"  Ambiguous (margin < {args.margin_threshold}): {len(ambiguous)} ({len(ambiguous)/max(1,n_total):.1%} of all)")

    # Slice breakdowns
    def rate_by(key):
        fail = Counter(f[key] for f in failures)
        total = Counter(e[key] for e in successes + failures)
        return [(k, fail[k], total[k]) for k in sorted(total.keys())]

    print("\nFailure rate by checkpoint_iter:")
    for k, nf, nt in rate_by("checkpoint_iter"):
        print(f"  ckpt {k:>5}: {nf:>3}/{nt:<3} = {nf/nt:.1%}")

    print("\nFailure rate by num_agents:")
    for k, nf, nt in rate_by("num_agents"):
        print(f"  {k:>3} agents: {nf:>3}/{nt:<3} = {nf/nt:.1%}")

    print("\nFailure rate by map_type:")
    for k, nf, nt in rate_by("map_type"):
        print(f"  {k:<7}: {nf:>3}/{nt:<3} = {nf/nt:.1%}")

    # Top confident failures
    confident.sort(key=lambda f: -f["margin"])
    print(f"\nTop {args.show} most confident failures (genuine model errors):")
    print(f"  {'off':<4} {'margin':<7} {'agents':<7} {'map':<8} {'mseed':<6} {'sseed':<6} {'ckpt':<6} {'file'}")
    print(f"  {'-'*4} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*30}")
    for f in confident[:args.show]:
        print(f"  {f['position_offset']:<4} {f['margin']:<7.2f} {f['num_agents']:<7} {f['map_type']:<8} "
              f"{f['map_seed']:<6} {f['scenario_seed']:<6} {f['checkpoint_iter']:<6} {f['path'].name}")

    # Distribution of position offset for failures
    print(f"\nHow far off was the model's top pick from true_best (in segments)?")
    print(f"(failures = offset > {args.tolerance})")
    offsets = Counter(f["position_offset"] for f in failures)
    for o in sorted(offsets.keys()):
        bar = "#" * offsets[o]
        print(f"  offset {o:>2}: {offsets[o]:>3} {bar}")


if __name__ == "__main__":
    main()
