"""
Train a 3D CNN segment classifier on held-out trajectory data.

The classifier takes a 16-step trajectory segment as input and outputs a scalar
congestion score. Training uses RankNet pairwise loss: within each episode,
segments with higher LaCAM diff should receive higher scores.

Four-stage workflow (controlled by --annotations):

  Stage 1 — auto labels only (omit --annotations, or pass it with
            --hold_out_every 1 so all annotations stay in val).
  Stage 2 — auto + hand-curated annotations.json (Option B overrides).
  Stage 3 — auto + warm-start preference elicitation (annotations produced
            by `query.py --model-path <stage-1-ckpt>`; uncertainty × diversity).
  Stage 4 — auto + cold-start preference elicitation (annotations produced
            by `query.py` with no --model-path; pure feature-distance diversity).

Stages 3 and 4 reuse --annotations: load_annotations auto-detects the
elicitation list format produced by query.py.

Stages 2/3/4 also support a two-phase fine-tune mode (--human-only): the
train set is restricted to rollouts with human labels and auto pairs are
disabled. Pair with --init-from <stage1.pt> so the auto-trained backbone
is fine-tuned on pure human signal instead of being diluted by ~99% auto
pairs.

Usage:
    # Stage 1 (auto only)
    python train_segment_classifier.py --data dataset/held_out \\
        --output out/stage1.pt --epochs 30

    # Stages 2/3/4 (two-phase fine-tune)
    python train_segment_classifier.py --data dataset/held_out \\
        --output out/stage2.pt --annotations annotations.json \\
        --human-only --init-from out/stage1.pt --epochs 30
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from held_out_seed_set import (
    STEPS_DELTA, GRID_PAD_SIZE, TRAIN_MAP_SEEDS, VAL_MAP_SEEDS,
)

# Pair-weighting: a pair (i, j) with diff[i] > diff[j] gets weight
# min(diff[i] - diff[j], MAX_PAIR_GAP) / MAX_PAIR_GAP. Gaps at or above this
# clip to weight 1.0; smaller gaps ramp linearly down to 0.
MAX_PAIR_GAP = 5


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def _resolve_annotation_path(npz_path: Optional[str]) -> Optional[str]:
    """Resolve a recorded npz_path to a stable absolute-path string.

    Annotations record paths relative to the repo root (the cwd where labeling
    was run). Resolving against the current cwd gives a stable string we can
    string-compare against the trainer's resolved episode paths."""
    if not npz_path:
        return None
    return str(Path(npz_path).resolve())


def load_annotations(path) -> Dict[str, List[Tuple[int, int]]]:
    """Read annotations file → {resolved_abs_npz_path: [(worst_idx, clean_idx), ...]}.

    Two formats are accepted, auto-detected by JSON shape. Both record an
    ``npz_path`` field per entry; that path (resolved to absolute) is the
    override key. Same scenario_id labeled across different rollouts (e.g.
    ``ckpt_500/<sid>.npz`` vs ``ckpt_1500/<sid>.npz``) coexist as separate
    annotations.

      • Original (Stage 2): dict keyed by scenario_id, with keys
        ``npz_path``, ``worst_congestion_failure_segment_index``,
        ``clearly_clean_segment_index``. Always produces a single-element
        list per path.

      • Elicitation (Stages 3–4, produced by query.py): list of per-pair
        entries with ``npz_path``, ``segment_a``, ``segment_b``,
        ``chosen_worse_segment``, ``label``. Entries labelled
        ``unsure_or_skipped`` are dropped. Multiple labels for the same
        rollout (pool-mode AL with --per-episode-cap > 1) accumulate.

    Entries with missing indices, missing npz_path, or worst==clean are dropped.
    """
    with open(path) as f:
        raw = json.load(f)

    out: Dict[str, List[Tuple[int, int]]] = {}

    if isinstance(raw, list):
        for entry in raw:
            key = _resolve_annotation_path(entry.get("npz_path"))
            chosen = entry.get("chosen_worse_segment")
            a = entry.get("segment_a")
            b = entry.get("segment_b")
            if key is None or chosen is None or a is None or b is None:
                continue
            a, b, chosen = int(a), int(b), int(chosen)
            if a == b or chosen not in (a, b):
                continue
            worst = chosen
            clean = b if chosen == a else a
            out.setdefault(key, []).append((worst, clean))
        return out

    for _sid, entry in raw.items():
        key = _resolve_annotation_path(entry.get("npz_path"))
        wi = entry.get("worst_congestion_failure_segment_index")
        ci = entry.get("clearly_clean_segment_index")
        if key is None or wi is None or ci is None or int(wi) == int(ci):
            continue
        out.setdefault(key, []).append((int(wi), int(ci)))
    return out


def split_annotations(
    annotations: Dict[str, List[Tuple[int, int]]], hold_out_every: int = 4
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, List[Tuple[int, int]]]]:
    """Deterministic split: every `hold_out_every`-th annotation (sorted by id)
    becomes val; the rest become train. Splits at the scenario level so that
    all pairs for a given episode go to the same side."""
    train, val = {}, {}
    for i, sid in enumerate(sorted(annotations)):
        (val if i % hold_out_every == 0 else train)[sid] = annotations[sid]
    return train, val


def resolve_path(path: Path) -> str:
    """Resolve an episode path to the same canonical string used as the override key."""
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

def featurize_segment(
    obstacles: np.ndarray,   # (H, W) int8
    positions: np.ndarray,   # (T_ep, N, 2) int16 — (x, y) = (col, row)
    goals: np.ndarray,       # (N, 2) int16 — (x, y)
    segment_idx: int,
    history_steps: int = 4,
    context_segments: int = 1,
) -> np.ndarray:
    """
    Build a (4, STEPS_DELTA * context_segments, GRID_PAD_SIZE, GRID_PAD_SIZE) tensor.

    context_segments=1 covers just this segment (16 frames).
    context_segments=2 also includes the next segment (32 frames) as future context.
    Frames past episode end are zero-padded.

    Channels:
      0 — agent density per timestep within the window
      1 — obstacle map (broadcast across T)
      2 — goal density (broadcast across T)
      3 — pre-segment agent history density (broadcast across T)
    """
    H, W = obstacles.shape
    seg_start = segment_idx * STEPS_DELTA
    T = STEPS_DELTA * context_segments

    def place(grid, xy_array):
        """Add 1 at each position; global_xy is (row, col)."""
        for xy in xy_array:
            r, c = int(xy[0]), int(xy[1])
            if 0 <= r < H and 0 <= c < W:
                grid[r, c] += 1.0

    # Channel 0: agent density, one frame per timestep
    agent_density = np.zeros((T, H, W), dtype=np.float32)
    for t_idx in range(T):
        t = seg_start + t_idx
        if t < len(positions):
            place(agent_density[t_idx], positions[t])

    # Channel 1: obstacles (static)
    obs_ch = np.broadcast_to(
        obstacles.astype(np.float32)[np.newaxis], (T, H, W)
    ).copy()

    # Channel 2: goal density (static)
    goal_grid = np.zeros((H, W), dtype=np.float32)
    place(goal_grid, goals)
    goal_ch = np.broadcast_to(goal_grid[np.newaxis], (T, H, W)).copy()

    # Channel 3: pre-segment history density (averaged, broadcast)
    history_start = max(0, seg_start - history_steps)
    hist_grid = np.zeros((H, W), dtype=np.float32)
    n_hist = seg_start - history_start
    if n_hist > 0:
        for t in range(history_start, seg_start):
            place(hist_grid, positions[t])
        hist_grid /= n_hist
    hist_ch = np.broadcast_to(hist_grid[np.newaxis], (T, H, W)).copy()

    # Stack → (4, T, H, W)
    tensor = np.stack([agent_density, obs_ch, goal_ch, hist_ch], axis=0)

    # Pad spatial dims to GRID_PAD_SIZE × GRID_PAD_SIZE
    pad_h = GRID_PAD_SIZE - H
    pad_w = GRID_PAD_SIZE - W
    if pad_h > 0 or pad_w > 0:
        tensor = np.pad(tensor, ((0, 0), (0, 0), (0, max(pad_h, 0)), (0, max(pad_w, 0))))

    return tensor.astype(np.float32)


def _augment(feat_a: np.ndarray, feat_b: np.ndarray):
    """Apply the same random spatial augmentation to both segment tensors."""
    flip_h = np.random.rand() > 0.5
    flip_w = np.random.rand() > 0.5
    k = np.random.randint(0, 4)  # 0/90/180/270 degree rotation

    def aug(t):
        if flip_h:
            t = t[:, :, ::-1, :].copy()
        if flip_w:
            t = t[:, :, :, ::-1].copy()
        if k:
            t = np.rot90(t, k=k, axes=(2, 3)).copy()
        return t

    return aug(feat_a), aug(feat_b)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_pairs(segment_diffs: np.ndarray, context_segments: int = 1) -> List[Tuple[int, int, float]]:
    """
    Return (i, j, weight) pairs where segment i should score higher than j,
    weighted by the size of the diff gap.

    The last (context_segments - 1) segments are excluded from training pairs
    because their forward context is partially zero-padded. Segments with
    diff == 0 are excluded entirely: that value is *corrupt*, not "no change"
    — it usually arises when LaCAM's makespan was identical at both segment
    boundaries (both probes timed out hitting MAX_EPISODE_STEPS, OR both saw
    a trivial residual problem because most agents had already reached
    goals). Either way, diff=0 reflects "LaCAM saw the same problem twice"
    rather than a real signal.

    Weight: min(diff[i] - diff[j], MAX_PAIR_GAP) / MAX_PAIR_GAP. Larger gaps
    are confident orderings (weight 1.0); small gaps are weak signal.
    """
    pairs = []
    S = len(segment_diffs) - (context_segments - 1)
    for i in range(S):
        if segment_diffs[i] == 0:
            continue
        for j in range(S):
            if i == j or segment_diffs[j] == 0:
                continue
            gap = int(segment_diffs[i]) - int(segment_diffs[j])
            if gap <= 0:
                continue
            w = min(gap, MAX_PAIR_GAP) / MAX_PAIR_GAP
            pairs.append((i, j, w))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentPairDataset(Dataset):
    def __init__(
        self,
        episode_paths: List[Path],
        map_seed_filter: set,
        augment: bool = False,
        context_segments: int = 1,
        human_overrides: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        human_only: bool = False,
    ):
        self.episodes = []
        self.pairs: List[Tuple[int, int, int, float]] = []  # (ep_idx, seg_i, seg_j, weight)
        self.augment = augment
        self.context_segments = context_segments
        human_overrides = human_overrides or {}
        self.human_only = human_only
        self.n_overridden = 0  # episodes whose pair list got replaced by human pair(s)
        self.n_human_pairs = 0  # total human-pair training items added

        for path in sorted(episode_paths):
            data = np.load(str(path), allow_pickle=True)
            if int(data["map_seed"]) not in map_seed_filter:
                continue

            path_key = resolve_path(path)
            has_human = path_key in human_overrides

            # Two-phase fine-tune mode: skip episodes without human labels
            # entirely (no auto pairs anywhere in the train set).
            if human_only and not has_human:
                continue

            ep_idx = len(self.episodes)
            self.episodes.append({
                "obstacles": data["obstacles"],
                "positions": data["positions"],
                "goals": data["goals"],
                "segment_diffs": data["segment_diffs"],
            })

            # If this exact rollout has human verdict(s), use the human pair(s)
            # instead of the auto-generated ones. The override key is the
            # resolved absolute path, so the same scenario_id labeled at
            # different checkpoints (e.g. ckpt_500/X vs ckpt_1500/X) only
            # overrides the rollout the human actually watched.
            if has_human:
                S = len(data["segment_diffs"]) - (context_segments - 1)
                added_any = False
                for wi, ci in human_overrides[path_key]:
                    if 0 <= wi < S and 0 <= ci < S:
                        self.pairs.append((ep_idx, wi, ci, 1.0))
                        self.n_human_pairs += 1
                        added_any = True
                if added_any:
                    self.n_overridden += 1
                continue  # skip auto pairs for this episode

            for i, j, w in generate_pairs(data["segment_diffs"], context_segments):
                self.pairs.append((ep_idx, i, j, w))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ep_idx, seg_i, seg_j, weight = self.pairs[idx]
        ep = self.episodes[ep_idx]
        feat_i = featurize_segment(ep["obstacles"], ep["positions"], ep["goals"], seg_i, context_segments=self.context_segments)
        feat_j = featurize_segment(ep["obstacles"], ep["positions"], ep["goals"], seg_j, context_segments=self.context_segments)
        if self.augment:
            feat_i, feat_j = _augment(feat_i, feat_j)
        return (
            torch.from_numpy(feat_i),
            torch.from_numpy(feat_j),
            torch.tensor(weight, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Segment3DCNN(nn.Module):
    """
    Small 3D CNN: (B, 4, 16, 24, 24) → scalar score per segment.
    """
    def __init__(self, in_channels: int = 4, base_ch: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, 3, padding=1),
            nn.GroupNorm(4, base_ch),
            nn.ReLU(),
            nn.MaxPool3d(2),                              # → (B, 16, 8, 12, 12)

            nn.Conv3d(base_ch, base_ch * 2, 3, padding=1),
            nn.GroupNorm(4, base_ch * 2),
            nn.ReLU(),
            nn.MaxPool3d(2),                              # → (B, 32, 4, 6, 6)

            nn.Conv3d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.GroupNorm(4, base_ch * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),                      # → (B, 64, 1, 1, 1)
        )
        self.head = nn.Linear(base_ch * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).flatten(1)).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Validation: weighted pairwise ranking accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_human_pairs(
    model: "Segment3DCNN",
    annotations: Dict[str, List[Tuple[int, int]]],
    device: str,
    context_segments: int = 1,
) -> Tuple[float, int]:
    """Pairwise accuracy: fraction of (worst, clean) pairs where score(worst) > score(clean).
    Each rollout contributes one entry per labeled pair (so rollouts with
    multiple pool-mode labels weigh more)."""
    model.eval()
    correct = total = 0
    for path_key, pair_list in annotations.items():
        path = Path(path_key)
        if not path.exists():
            continue
        data = np.load(str(path), allow_pickle=True)
        S = len(data["segment_diffs"]) - (context_segments - 1)
        for wi, ci in pair_list:
            if not (0 <= wi < S and 0 <= ci < S):
                continue
            feat_w = featurize_segment(data["obstacles"], data["positions"], data["goals"], wi, context_segments=context_segments)
            feat_c = featurize_segment(data["obstacles"], data["positions"], data["goals"], ci, context_segments=context_segments)
            x = torch.from_numpy(np.stack([feat_w, feat_c])).to(device)
            scores = model(x).cpu().numpy()
            correct += int(scores[0] > scores[1])
            total += 1
    return (correct / total if total > 0 else 0.0), total


@torch.no_grad()
def evaluate_pair_accuracy(model: Segment3DCNN, episode_paths: List[Path], map_seed_filter: set, device: str, context_segments: int = 1) -> float:
    """
    Weighted pairwise ranking accuracy on val-map auto-pairs.

    For each val episode, generate pairs the same way training does (gap-clipped
    weights, diff==0 excluded), score every segment, and accumulate
        weight * 1[score[i] > score[j]]
    Return the weighted accuracy: (sum of weight on correctly-ordered pairs) /
    (total weight). Matches training distribution and weighting, so it's the
    most direct val-side analog of training loss.
    """
    model.eval()
    correct_w = total_w = 0.0

    for path in sorted(episode_paths):
        data = np.load(str(path), allow_pickle=True)
        if int(data["map_seed"]) not in map_seed_filter:
            continue

        diffs = data["segment_diffs"]
        pairs = generate_pairs(diffs, context_segments=context_segments)
        if not pairs:
            continue

        S = len(diffs) - (context_segments - 1)
        feats = np.stack([
            featurize_segment(data["obstacles"], data["positions"], data["goals"], s, context_segments=context_segments)
            for s in range(S)
        ])
        scores = model(torch.from_numpy(feats).to(device)).cpu().numpy()

        for i, j, w in pairs:
            total_w += w
            if scores[i] > scores[j]:
                correct_w += w

    if total_w == 0.0:
        return 0.0
    return correct_w / total_w


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train segment congestion classifier")
    parser.add_argument("--data", required=True, help="Root directory containing ckpt_*/  episode folders")
    parser.add_argument("--output", required=True, help="Output path for saved model (.pt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context_segments", type=int, default=1, choices=[1, 2],
                        help="Number of 16-step segments in the temporal window (1=this segment only, 2=this+next)")
    parser.add_argument("--base_ch", type=int, default=16,
                        help="CNN base channel width; must be divisible by 4 (default: 16)")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Path to a human annotations file with labels on TRAIN_MAP_SEEDS rollouts. "
                             "If set, ALL labels are used as Option-B training overrides (no within-train "
                             "holdout). Held-out human val signal comes from --val-annotations only.")
    parser.add_argument("--min_checkpoint", type=int, default=0,
                        help="Skip episodes from MAPF-GPT checkpoint_iter < this (default: 0 = use all)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "plateau"],
                        help="LR schedule: 'none' (constant), 'cosine' (CosineAnnealingLR), 'plateau' (ReduceLROnPlateau on pair_acc)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the train dataset, print pair-count and weight-distribution stats, "
                             "then exit before training. Useful for sanity-checking pair generation.")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to a .pt checkpoint to initialize weights from before training. "
                             "Used for warm-start fine-tuning across Stages 2/3/4 — initializes from "
                             "Stage 1's baseline so human-label training fine-tunes the auto-trained "
                             "backbone rather than training from scratch. Architecture (base_ch, "
                             "context_segments, in_channels) must match the CLI args.")
    parser.add_argument("--val-annotations", type=str, default=None,
                        help="Path to a SECOND annotations file containing human labels on "
                             "VAL_MAP_SEEDS rollouts (the held-out maps the trainer never sees). "
                             "Eval-only — never used as training overrides. Adds a separate "
                             "'val-map' human metric each epoch (pairwise + argmax accuracy) and "
                             "saves a best-by-val-map-pairwise checkpoint as {stem}.val_map_human.pt.")
    parser.add_argument("--human-only", action="store_true",
                        help="Two-phase fine-tune mode: train only on human-labeled pairs. "
                             "Drops episodes without annotations from the train set and disables "
                             "auto pair generation entirely, so 100%% of gradient signal comes "
                             "from human labels. Requires --annotations; pair with --init-from "
                             "<stage1.pt> to fine-tune the auto-trained backbone.")
    args = parser.parse_args()

    if args.human_only and not args.annotations:
        parser.error("--human-only requires --annotations (no human labels = no train pairs)")
    if args.human_only and not args.init_from:
        print("⚠ --human-only without --init-from: fine-tuning from scratch on a tiny "
              "human-only dataset will likely underfit. Consider passing a Stage 1 ckpt.")

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    episode_paths = list(Path(args.data).rglob("*.npz"))
    print(f"Found {len(episode_paths)} episode files under {args.data}")

    # Apply min-checkpoint filter (only affects ckpt_*/ files; everything else is kept regardless).
    if args.min_checkpoint > 0:
        filtered = []
        for p in episode_paths:
            ckpt_dir = p.parent.name  # e.g. 'ckpt_500'
            if ckpt_dir.startswith("ckpt_"):
                try:
                    if int(ckpt_dir[5:]) >= args.min_checkpoint:
                        filtered.append(p)
                except ValueError:
                    filtered.append(p)
            else:
                filtered.append(p)
        episode_paths = filtered
        print(f"Filtered to {len(episode_paths)} episodes (ckpt_*/ kept only if iter >= {args.min_checkpoint})")

    # Train-map human annotations: all go into training overrides (no within-train holdout).
    # Held-out human val signal comes from --val-annotations only (val-map rollouts).
    # Annotation keys are resolved absolute npz paths; the override key matches a loaded
    # rollout's resolved path.
    train_annotations: Dict[str, List[Tuple[int, int]]] = {}
    if args.annotations:
        train_annotations = load_annotations(args.annotations)
        n_missing = sum(1 for k in train_annotations if not Path(k).exists())
        print(f"Train-map annotations: {len(train_annotations)} rollouts "
              f"({n_missing} not on disk; all used as training overrides)")

    # Val-map human annotations: eval-only, never overrides. The actual human val signal.
    val_map_annotations: Dict[str, List[Tuple[int, int]]] = {}
    if args.val_annotations:
        val_map_annotations = load_annotations(args.val_annotations)
        n_missing_vm = sum(1 for k in val_map_annotations if not Path(k).exists())
        print(f"Val-map annotations: {len(val_map_annotations)} rollouts "
              f"({n_missing_vm} not on disk; eval-only)")

    train_dataset = SegmentPairDataset(
        episode_paths, TRAIN_MAP_SEEDS,
        augment=True, context_segments=args.context_segments,
        human_overrides=train_annotations or None,
        human_only=args.human_only,
    )
    val_dataset = SegmentPairDataset(
        episode_paths, VAL_MAP_SEEDS,
        augment=False, context_segments=args.context_segments, human_overrides=None,
    )
    if len(train_dataset) == 0:
        raise SystemExit(
            "Train dataset is empty. "
            + ("--human-only mode but no TRAIN_MAP_SEEDS rollout has human labels."
               if args.human_only else "Check --data and seed filters.")
        )
    print(f"Train pairs: {len(train_dataset)}  |  Val pairs: {len(val_dataset)}")
    if args.annotations:
        if args.human_only:
            print(
                f"  ↳ Two-phase: {train_dataset.n_human_pairs} human pair(s) from "
                f"{train_dataset.n_overridden} rollouts (auto pairs disabled)"
            )
        else:
            print(
                f"  ↳ {train_dataset.n_overridden} train episodes had auto pairs replaced "
                f"by {train_dataset.n_human_pairs} human pair(s) total"
            )

    if args.dry_run:
        from collections import Counter
        n_eps = len(train_dataset.episodes)
        per_ep = Counter(p[0] for p in train_dataset.pairs)
        counts = np.array([per_ep.get(i, 0) for i in range(n_eps)], dtype=int)
        weights = np.array([p[3] for p in train_dataset.pairs], dtype=float)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(weights, bins=bins + [1.0001])
        total_w = float(weights.sum())

        print("\n=== Dry run: pair stats (TRAIN_MAP_SEEDS) ===")
        print(f"Episodes loaded:      {n_eps}")
        print(f"Episodes w/ ≥1 pair:  {(counts > 0).sum()}  ({(counts == 0).sum()} contributed zero)")
        if (counts > 0).any():
            nz = counts[counts > 0]
            print(f"Pairs/episode (nonzero only): min={nz.min()}  median={int(np.median(nz))}  "
                  f"mean={nz.mean():.1f}  max={nz.max()}")
        print(f"Total pairs:          {len(weights)}")
        print(f"Total weight (sum):   {total_w:.1f}  "
              f"(mean weight {weights.mean():.3f})")
        print("Weight histogram:")
        for lo, hi, c in zip(bins, bins[1:] + [1.0], hist):
            bar = "#" * int(40 * c / max(hist.max(), 1))
            print(f"  ({lo:.1f}, {hi:.1f}]  {c:7d}  {bar}")
        print("=== Dry run complete; exiting before training. ===")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = Segment3DCNN(base_ch=args.base_ch).to(device)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        ckpt_base_ch = ckpt.get("base_ch")
        ckpt_ctx = ckpt.get("context_segments", 1)
        if ckpt_base_ch is not None and ckpt_base_ch != args.base_ch:
            raise SystemExit(
                f"--init-from architecture mismatch: ckpt base_ch={ckpt_base_ch} "
                f"vs --base_ch {args.base_ch}"
            )
        if ckpt_ctx != args.context_segments:
            raise SystemExit(
                f"--init-from architecture mismatch: ckpt context_segments={ckpt_ctx} "
                f"vs --context_segments {args.context_segments}"
            )
        model.load_state_dict(ckpt["state_dict"])
        print(f"Initialized weights from {args.init_from} (base_ch={ckpt_base_ch}, "
              f"ctx={ckpt_ctx}, save_by={ckpt.get('save_by')})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    else:
        scheduler = None

    # We track two best-checkpoint criteria, written to up to two .pt files:
    #   args.output                              — best human_val pairwise on val-map labels (the human val signal)
    #   {stem}.pair_acc.pt                       — best weighted pairwise accuracy on val-map auto labels
    #
    # If --val-annotations isn't passed, only pair_acc is saved (no human val signal).
    primary_path = Path(args.output)
    pair_acc_path = primary_path.with_name(primary_path.stem + ".pair_acc" + primary_path.suffix)

    def _save_ckpt(path: Path, criterion: str, value: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "base_ch": args.base_ch,
            "in_channels": 4,
            "context_segments": args.context_segments,
            "save_by": criterion,
            "best_metric": value,
        }, str(path))

    train_losses, val_pair_accs = [], []
    hpair_trains = []  # pairwise sanity on training labels
    hpair_vals = []    # pairwise on val-map labels — the actual human val signal
    best_pair_acc = 0.0
    best_human_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for feat_i, feat_j, weights in train_loader:
            feat_i, feat_j, weights = feat_i.to(device), feat_j.to(device), weights.to(device)
            loss = (-F.logsigmoid(model(feat_i) - model(feat_j)) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        pair_acc = evaluate_pair_accuracy(model, episode_paths, VAL_MAP_SEEDS, device, context_segments=args.context_segments)
        train_losses.append(avg_loss)
        val_pair_accs.append(pair_acc)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(pair_acc)
            else:
                scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        msg = f"Epoch {epoch:3d} | lr {cur_lr:.2e} | loss {avg_loss:.4f} | pair_acc {pair_acc:.3f}"

        # Train-side sanity: pairwise accuracy on the rollouts we trained on.
        # Should saturate near 1.0 once training converges; useful as a debug signal.
        hp_train = 0.0
        if train_annotations:
            hp_train, _ = evaluate_human_pairs(model, train_annotations, device, args.context_segments)
            hpair_trains.append(hp_train)

        # Held-out human val (val-map rollouts + human labels). The actual generalization signal.
        hp_val = 0.0
        if val_map_annotations:
            hp_val, _ = evaluate_human_pairs(model, val_map_annotations, device, args.context_segments)
            hpair_vals.append(hp_val)

        if train_annotations or val_map_annotations:
            msg += f" | human_val tr/val {hp_train:.3f}/{hp_val:.3f}"

        print(msg)

        # Always save best-pair_acc checkpoint (no annotation dependency — uses auto val).
        if pair_acc > best_pair_acc:
            best_pair_acc = pair_acc
            _save_ckpt(pair_acc_path, "pair_acc", best_pair_acc)
            print(f"  → saved {pair_acc_path.name} (best pair_acc: {best_pair_acc:.3f})")

        # Human-val-driven checkpoint: only fires when --val-annotations is set.
        if val_map_annotations and hpair_vals:
            if hpair_vals[-1] > best_human_val:
                best_human_val = hpair_vals[-1]
                _save_ckpt(primary_path, "human_val", best_human_val)
                print(f"  → saved {primary_path.name} (best human_val: {best_human_val:.3f})")

    print(f"Training complete.")
    print(f"  best pair_acc (val map seeds 144-147): {best_pair_acc:.3f}")
    if val_map_annotations:
        print(f"  best human_val (pairwise, val-map labels):           {best_human_val:.3f}  [random=0.5]")

    import matplotlib.pyplot as plt
    has_human = bool(train_annotations or val_map_annotations)
    n_panels = 3 if has_human else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.6 * n_panels), sharex=True)
    epochs = range(1, args.epochs + 1)
    axes[0].plot(epochs, train_losses); axes[0].set_ylabel("Train Loss"); axes[0].grid(True)
    axes[1].plot(epochs, val_pair_accs, color="steelblue", label="weighted pair acc")
    axes[1].set_ylabel("Val Acc (auto)"); axes[1].grid(True); axes[1].legend(loc="lower right")

    if has_human:
        if hpair_trains: axes[2].plot(epochs, hpair_trains, label="train")
        if hpair_vals:   axes[2].plot(epochs, hpair_vals,   label="val")
        axes[2].axhline(0.5, color="red", linestyle=":", alpha=0.4, label="chance")
        axes[2].set_ylabel("human_val"); axes[2].grid(True); axes[2].legend(loc="lower right")

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    plot_path = Path(args.output).with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    print(f"Loss curve saved to {plot_path}")


if __name__ == "__main__":
    main()
