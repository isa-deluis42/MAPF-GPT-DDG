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
import numpy as np
from pathlib import Path
from typing import List, Tuple

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
    def __init__(self, episode_paths: List[Path], map_seed_filter: set, augment: bool = False, context_segments: int = 1):
        self.episodes = []
        self.pairs: List[Tuple[int, int, int, float]] = []  # (ep_idx, seg_i, seg_j, weight)
        self.augment = augment
        self.context_segments = context_segments

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
    parser.add_argument("--min_checkpoint", type=int, default=0,
                        help="Skip episodes from MAPF-GPT checkpoint_iter < this (default: 0 = use all)")
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    episode_paths = list(Path(args.data).rglob("*.npz"))
    print(f"Found {len(episode_paths)} episode files")

    if args.min_checkpoint > 0:
        filtered = []
        for p in episode_paths:
            ckpt_dir = p.parent.name  # e.g. 'ckpt_500'
            if ckpt_dir.startswith("ckpt_") and int(ckpt_dir[5:]) >= args.min_checkpoint:
                filtered.append(p)
        episode_paths = filtered
        print(f"Filtered to {len(episode_paths)} episodes (checkpoint_iter >= {args.min_checkpoint})")

    train_dataset = SegmentPairDataset(episode_paths, TRAIN_MAP_SEEDS, augment=True, context_segments=args.context_segments)
    val_dataset = SegmentPairDataset(episode_paths, VAL_MAP_SEEDS, augment=False, context_segments=args.context_segments)
    print(f"Train pairs: {len(train_dataset)}  |  Val pairs: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = Segment3DCNN(base_ch=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_pm1s, val_top3s = [], [], []

    best_pm1 = 0.0
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

        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | top-1±1 {pm1:.3f} | top-3 {top3:.3f}")

        if pm1 > best_pm1:
            best_pm1 = pm1
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "base_ch": args.base_ch, "in_channels": 4, "context_segments": args.context_segments}, str(out_path))
            print(f"  → saved (best top-1±1: {best_pm1:.3f})")

    print(f"Training complete. Best val top-1±1 acc: {best_pm1:.3f}")

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    epochs = range(1, args.epochs + 1)
    ax1.plot(epochs, train_losses)
    ax1.set_ylabel("Train Loss")
    ax1.grid(True)
    ax2.plot(epochs, val_pm1s, color="orange", label="top-1±1 (DDG-aligned)")
    ax2.plot(epochs, val_top3s, color="steelblue", linestyle="--", label="top-3")
    ax2.set_ylabel("Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()
    plot_path = Path(args.output).with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    print(f"Loss curve saved to {plot_path}")


if __name__ == "__main__":
    main()
