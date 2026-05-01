"""
Train a 3D CNN segment classifier on held-out trajectory data.

The classifier takes a 16-step trajectory segment as input and outputs a scalar
congestion score. Training uses RankNet pairwise loss: within each episode,
segments with higher LaCAM diff should receive higher scores.

Usage:
    python finetuning/train_segment_classifier.py \
        --data dataset/held_out \
        --output out/segment_classifier.pt \
        --epochs 30
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

# Diff thresholds (matching DDG's FastSolverDeltaConfig)
DIFF_CONFIDENT_POS = 3
DIFF_CONFIDENT_NEG = 1


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def load_annotations(path) -> Dict[str, Tuple[int, int]]:
    """Read annotations.json → {scenario_id: (worst_idx, clean_idx)}.
    Drops entries with missing indices or worst==clean."""
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, Tuple[int, int]] = {}
    for sid, entry in raw.items():
        wi = entry.get("worst_congestion_failure_segment_index")
        ci = entry.get("clearly_clean_segment_index")
        if wi is None or ci is None or int(wi) == int(ci):
            continue
        out[sid] = (int(wi), int(ci))
    return out


def split_annotations(
    annotations: Dict[str, Tuple[int, int]], hold_out_every: int = 4
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
    """Deterministic split: every `hold_out_every`-th annotation (sorted by id)
    becomes val; the rest become train."""
    train, val = {}, {}
    for i, sid in enumerate(sorted(annotations)):
        (val if i % hold_out_every == 0 else train)[sid] = annotations[sid]
    return train, val


def is_annotated_npz(path: Path) -> bool:
    """True iff `path` lives in filtered_npzs/annotated/ — where the actual
    annotated rollouts were moved during the labeling pass.
    These are the only files that the human verdict applies to (other ckpt_*/
    files with the same scenario_id are *different rollouts* of the same scenario)."""
    return "annotated" in path.parts


def annotation_path_for(scenario_id: str, episode_paths: List[Path]) -> Optional[Path]:
    """Find the filtered_npzs/annotated/<scenario_id>.npz the human verdict applies to."""
    for p in episode_paths:
        if p.stem == scenario_id and is_annotated_npz(p):
            return p
    return None


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
    Return (i, j, weight) pairs where segment i should score higher than j.

    The last (context_segments - 1) segments are excluded from training pairs
    because their forward context is partially zero-padded.

    Weights:
      confident_pos vs confident_neg  → 1.0
      one confident, one midrange     → 0.5
      both midrange                   → skipped (0.0)
    """
    pairs = []
    S = len(segment_diffs) - (context_segments - 1)
    for i in range(S):
        for j in range(S):
            if i == j or segment_diffs[i] <= segment_diffs[j]:
                continue
            bi = "pos" if segment_diffs[i] > DIFF_CONFIDENT_POS else (
                 "neg" if segment_diffs[i] < DIFF_CONFIDENT_NEG else "mid")
            bj = "pos" if segment_diffs[j] > DIFF_CONFIDENT_POS else (
                 "neg" if segment_diffs[j] < DIFF_CONFIDENT_NEG else "mid")
            if bi == "pos" and bj == "neg":
                w = 1.0
            elif (bi == "pos" and bj == "mid") or (bi == "mid" and bj == "neg"):
                w = 0.5
            else:
                continue
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
        human_overrides: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        self.episodes = []
        self.pairs: List[Tuple[int, int, int, float]] = []  # (ep_idx, seg_i, seg_j, weight)
        self.augment = augment
        self.context_segments = context_segments
        human_overrides = human_overrides or {}
        self.n_overridden = 0  # episodes whose pair list got replaced by a human pair

        for path in sorted(episode_paths):
            data = np.load(str(path), allow_pickle=True)
            if int(data["map_seed"]) not in map_seed_filter:
                continue
            ep_idx = len(self.episodes)
            self.episodes.append({
                "obstacles": data["obstacles"],
                "positions": data["positions"],
                "goals": data["goals"],
                "segment_diffs": data["segment_diffs"],
            })

            # Option B: if this is an annotated rollout with a human verdict,
            # replace all auto-generated pairs from this episode with the single human pair.
            sid = path.stem
            if is_annotated_npz(path) and sid in human_overrides:
                wi, ci = human_overrides[sid]
                S = len(data["segment_diffs"]) - (context_segments - 1)
                if 0 <= wi < S and 0 <= ci < S:
                    self.pairs.append((ep_idx, wi, ci, 1.0))
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
# Validation: top-3 agreement
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_human_pairs(
    model: "Segment3DCNN",
    episode_paths: List[Path],
    annotations: Dict[str, Tuple[int, int]],
    device: str,
    context_segments: int = 1,
) -> Tuple[float, int]:
    """Pairwise accuracy: fraction of (worst, clean) pairs where score(worst) > score(clean)."""
    model.eval()
    correct = total = 0
    for sid, (wi, ci) in annotations.items():
        path = annotation_path_for(sid, episode_paths)
        if path is None:
            continue
        data = np.load(str(path), allow_pickle=True)
        S = len(data["segment_diffs"]) - (context_segments - 1)
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
def evaluate_human_worst_match(
    model: "Segment3DCNN",
    episode_paths: List[Path],
    annotations: Dict[str, Tuple[int, int]],
    device: str,
    context_segments: int = 1,
) -> Tuple[float, float, float, int]:
    """For each annotated episode, score every segment and check how close
    argmax(scores) is to the human's worst_idx.

    Returns (top1, top1_pm1, top3, n_episodes_evaluated).
      top1     — argmax(scores) == human worst_idx
      top1_pm1 — |argmax(scores) - human worst_idx| ≤ 1   (deployment-shape; mirrors the auto top-1±1)
      top3     — human worst_idx ∈ top-3 by score
    """
    model.eval()
    t1 = pm1 = t3 = total = 0
    for sid, (wi, _ci) in annotations.items():
        path = annotation_path_for(sid, episode_paths)
        if path is None:
            continue
        data = np.load(str(path), allow_pickle=True)
        S = len(data["segment_diffs"]) - (context_segments - 1)
        if S < 2 or not (0 <= wi < S):
            continue
        feats = np.stack([
            featurize_segment(data["obstacles"], data["positions"], data["goals"], s, context_segments=context_segments)
            for s in range(S)
        ])
        scores = model(torch.from_numpy(feats).to(device)).cpu().numpy()
        pred = int(np.argmax(scores))
        t1 += int(pred == wi)
        pm1 += int(abs(pred - wi) <= 1)
        t3 += int(wi in set(np.argsort(scores)[-3:].tolist()))
        total += 1
    if total == 0:
        return 0.0, 0.0, 0.0, 0
    return t1 / total, pm1 / total, t3 / total, total


@torch.no_grad()
def evaluate_metrics(model: Segment3DCNN, episode_paths: List[Path], map_seed_filter: set, device: str, context_segments: int = 1) -> Tuple[float, float]:
    """
    For each val episode compute two metrics:
      top1_pm1 — model's argmax score is within ±1 of argmax(segment_diffs) [DDG-aligned]
      top3     — argmax(segment_diffs) is in the CNN's top-3 scored segments
    Returns (top1_pm1, top3).
    """
    model.eval()
    pm1_correct = top3_correct = total = 0

    for path in sorted(episode_paths):
        data = np.load(str(path), allow_pickle=True)
        if int(data["map_seed"]) not in map_seed_filter:
            continue

        diffs = data["segment_diffs"]
        S = len(diffs)
        if S < 2:
            continue

        feats = np.stack([
            featurize_segment(data["obstacles"], data["positions"], data["goals"], s, context_segments=context_segments)
            for s in range(S)
        ])
        scores = model(torch.from_numpy(feats).to(device)).cpu().numpy()

        true_best = int(np.argmax(diffs))
        pred_best = int(np.argmax(scores))
        top3 = set(np.argsort(scores)[-3:].tolist())

        pm1_correct += int(abs(pred_best - true_best) <= 1)
        top3_correct += int(true_best in top3)
        total += 1

    if total == 0:
        return 0.0, 0.0
    return pm1_correct / total, top3_correct / total


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
                        help="Path to annotations.json. If set, enables Option-B human-label override "
                             "during training and reports human-pair accuracy each epoch.")
    parser.add_argument("--hold_out_every", type=int, default=4,
                        help="Every Nth annotation (sorted by scenario_id) is held out from training. "
                             "Default 4 → ~7/28 held out for val human-pair accuracy.")
    parser.add_argument("--min_checkpoint", type=int, default=0,
                        help="Skip episodes from MAPF-GPT checkpoint_iter < this (default: 0 = use all)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "plateau"],
                        help="LR schedule: 'none' (constant), 'cosine' (CosineAnnealingLR), 'plateau' (ReduceLROnPlateau on top-1±1)")
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    episode_paths = list(Path(args.data).rglob("*.npz"))
    n_annotated = sum(1 for p in episode_paths if is_annotated_npz(p))
    print(f"Found {len(episode_paths)} episode files ({n_annotated} under filtered_npzs/annotated/)")

    # Apply min-checkpoint filter (only affects ckpt_*/ files; annotated/ and filtered_npzs/ are kept regardless).
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
                # Keep filtered_npzs/ and annotated/ files (they don't carry a per-checkpoint tag in the path).
                filtered.append(p)
        episode_paths = filtered
        print(f"Filtered to {len(episode_paths)} episodes (ckpt_*/ kept only if iter >= {args.min_checkpoint})")

    # Load annotations for Stage-1 measurement and (optionally) Stage-2 training overrides.
    all_annotations: Dict[str, Tuple[int, int]] = {}
    train_annotations: Dict[str, Tuple[int, int]] = {}
    val_annotations: Dict[str, Tuple[int, int]] = {}
    if args.annotations:
        all_annotations = load_annotations(args.annotations)
        train_annotations, val_annotations = split_annotations(all_annotations, args.hold_out_every)
        print(f"Annotations: {len(all_annotations)} total → {len(train_annotations)} train + {len(val_annotations)} val "
              f"(hold_out_every={args.hold_out_every})")

    train_overrides = train_annotations if args.annotations else None
    train_dataset = SegmentPairDataset(
        episode_paths, TRAIN_MAP_SEEDS,
        augment=True, context_segments=args.context_segments, human_overrides=train_overrides,
    )
    val_dataset = SegmentPairDataset(
        episode_paths, VAL_MAP_SEEDS,
        augment=False, context_segments=args.context_segments, human_overrides=None,
    )
    print(f"Train pairs: {len(train_dataset)}  |  Val pairs: {len(val_dataset)}")
    if args.annotations:
        print(f"  ↳ {train_dataset.n_overridden} train episodes had auto pairs replaced by human pair")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = Segment3DCNN(base_ch=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    else:
        scheduler = None

    # We track four best-checkpoint criteria simultaneously and write up to four .pt files:
    #   args.output                              — best human_val (pairwise vs clean), the original criterion
    #   {stem}.argmax_pm1.pt                     — best DDG-aligned top-1±1 on val map seeds (lots of data, noisy auto-label)
    #   {stem}.human_argmax.pt                   — best human-argmax±1 on val annotations (gold label, ±1 tolerance)
    #   {stem}.human_argmax_top1.pt              — best human-argmax exact-match (gold label, strictest)
    #
    # If --annotations isn't passed, only the argmax_pm1 checkpoint is written.
    primary_path = Path(args.output)
    pm1_path             = primary_path.with_name(primary_path.stem + ".argmax_pm1"        + primary_path.suffix)
    human_argmax_path    = primary_path.with_name(primary_path.stem + ".human_argmax"      + primary_path.suffix)
    human_top1_path      = primary_path.with_name(primary_path.stem + ".human_argmax_top1" + primary_path.suffix)

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

    train_losses, val_pm1s, val_top3s = [], [], []
    hpair_alls, hpair_trains, hpair_vals = [], [], []
    ha_t1_alls, ha_t1_trains, ha_t1_vals = [], [], []      # human-argmax exact-match trajectories
    ha_pm1_alls, ha_pm1_trains, ha_pm1_vals = [], [], []   # human-argmax±1 trajectories
    best_human_val = 0.0
    best_pm1 = 0.0
    best_human_argmax_pm1 = 0.0
    best_human_argmax_top1 = 0.0

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
        pm1, top3 = evaluate_metrics(model, episode_paths, VAL_MAP_SEEDS, device, context_segments=args.context_segments)
        train_losses.append(avg_loss)
        val_pm1s.append(pm1)
        val_top3s.append(top3)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(pm1)
            else:
                scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        msg = f"Epoch {epoch:3d} | lr {cur_lr:.2e} | loss {avg_loss:.4f} | top-1±1 {pm1:.3f} | top-3 {top3:.3f}"

        # Human-aligned metrics — pairwise (existing) + argmax top-1 + argmax±1 over annotation subsets.
        ha_t1_val = 0.0
        ha_pm1_val = 0.0
        if all_annotations:
            hp_all, n_all = evaluate_human_pairs(model, episode_paths, all_annotations, device, args.context_segments)
            hp_train, n_train = evaluate_human_pairs(model, episode_paths, train_annotations, device, args.context_segments)
            hp_val, n_val = evaluate_human_pairs(model, episode_paths, val_annotations, device, args.context_segments)
            hpair_alls.append(hp_all); hpair_trains.append(hp_train); hpair_vals.append(hp_val)

            ha_t1_all,   ha_pm1_all,   _, _ = evaluate_human_worst_match(model, episode_paths, all_annotations,   device, args.context_segments)
            ha_t1_train, ha_pm1_train, _, _ = evaluate_human_worst_match(model, episode_paths, train_annotations, device, args.context_segments)
            ha_t1_val,   ha_pm1_val,   _, _ = evaluate_human_worst_match(model, episode_paths, val_annotations,   device, args.context_segments)
            ha_t1_alls.append(ha_t1_all);   ha_t1_trains.append(ha_t1_train);   ha_t1_vals.append(ha_t1_val)
            ha_pm1_alls.append(ha_pm1_all); ha_pm1_trains.append(ha_pm1_train); ha_pm1_vals.append(ha_pm1_val)

            msg += (f" | hp all/tr/val {hp_all:.3f}/{hp_train:.3f}/{hp_val:.3f}"
                    f" | h-arg top1 a/t/v {ha_t1_all:.3f}/{ha_t1_train:.3f}/{ha_t1_val:.3f}"
                    f" | h-arg ±1 a/t/v {ha_pm1_all:.3f}/{ha_pm1_train:.3f}/{ha_pm1_val:.3f}")

        print(msg)

        # Always save best-pm1 checkpoint (no annotation dependency).
        if pm1 > best_pm1:
            best_pm1 = pm1
            _save_ckpt(pm1_path, "argmax_pm1", best_pm1)
            print(f"  → saved {pm1_path.name} (best argmax_pm1: {best_pm1:.3f})")

        # Save the three annotation-driven checkpoints when annotations are present.
        if args.annotations and hpair_vals:
            if hpair_vals[-1] > best_human_val:
                best_human_val = hpair_vals[-1]
                _save_ckpt(primary_path, "human_val", best_human_val)
                print(f"  → saved {primary_path.name} (best human_val: {best_human_val:.3f})")
            if ha_pm1_val > best_human_argmax_pm1:
                best_human_argmax_pm1 = ha_pm1_val
                _save_ckpt(human_argmax_path, "human_argmax_pm1", best_human_argmax_pm1)
                print(f"  → saved {human_argmax_path.name} (best human_argmax±1: {best_human_argmax_pm1:.3f})")
            if ha_t1_val > best_human_argmax_top1:
                best_human_argmax_top1 = ha_t1_val
                _save_ckpt(human_top1_path, "human_argmax_top1", best_human_argmax_top1)
                print(f"  → saved {human_top1_path.name} (best human_argmax top-1: {best_human_argmax_top1:.3f})")

    print(f"Training complete.")
    print(f"  best argmax_pm1 (DDG-aligned, val map seeds 144-147): {best_pm1:.3f}")
    if args.annotations:
        print(f"  best human_val (pairwise, 13 held-out annotations):    {best_human_val:.3f}")
        print(f"  best human_argmax±1 (gold ±1, 13 held-out):             {best_human_argmax_pm1:.3f}  [random≈0.81]")
        print(f"  best human_argmax top-1 (gold exact, 13 held-out):      {best_human_argmax_top1:.3f}  [random≈0.35]")

    import matplotlib.pyplot as plt
    n_panels = 5 if all_annotations else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.6 * n_panels), sharex=True)
    epochs = range(1, args.epochs + 1)
    axes[0].plot(epochs, train_losses); axes[0].set_ylabel("Train Loss"); axes[0].grid(True)
    axes[1].plot(epochs, val_pm1s,  color="orange",    label="top-1±1 (DDG)")
    axes[1].plot(epochs, val_top3s, color="steelblue", linestyle="--", label="top-3")
    axes[1].set_ylabel("Val Acc (auto)"); axes[1].grid(True); axes[1].legend(loc="lower right")
    if all_annotations:
        axes[2].plot(epochs, hpair_alls,   label=f"all ({n_all})")
        axes[2].plot(epochs, hpair_trains, label=f"train ({n_train})")
        axes[2].plot(epochs, hpair_vals,   label=f"val ({n_val})")
        axes[2].axhline(0.5, color="red", linestyle=":", alpha=0.4)
        axes[2].set_ylabel("HP Pairwise"); axes[2].grid(True); axes[2].legend(loc="lower right")
        axes[3].plot(epochs, ha_t1_alls,   label=f"all ({n_all})")
        axes[3].plot(epochs, ha_t1_trains, label=f"train ({n_train})")
        axes[3].plot(epochs, ha_t1_vals,   label=f"val ({n_val})")
        axes[3].axhline(0.354, color="red", linestyle=":", alpha=0.4, label="random")
        axes[3].set_ylabel("HP Argmax (top-1)"); axes[3].grid(True); axes[3].legend(loc="lower right")
        axes[4].plot(epochs, ha_pm1_alls,   label=f"all ({n_all})")
        axes[4].plot(epochs, ha_pm1_trains, label=f"train ({n_train})")
        axes[4].plot(epochs, ha_pm1_vals,   label=f"val ({n_val})")
        axes[4].axhline(0.812, color="red", linestyle=":", alpha=0.4, label="random")
        axes[4].set_ylabel("HP Argmax±1"); axes[4].grid(True); axes[4].legend(loc="lower right")
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    plot_path = Path(args.output).with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    print(f"Loss curve saved to {plot_path}")


if __name__ == "__main__":
    main()
