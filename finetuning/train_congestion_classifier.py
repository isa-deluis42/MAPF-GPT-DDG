"""
Congestion Classifier Training

Phase 1 trains on high-confidence auto-labels and leaves uncertain cases for
human review or later active-learning rounds.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from finetuning.congestion_utils import (
    CongestionClassifier,
    IndexedCongestionDataset,
    load_congestion_arrow,
    select_training_indices,
    split_by_episode,
)


def build_dataloader(dataset: IndexedCongestionDataset, batch_size: int, shuffle: bool) -> DataLoader:
    worker_count = min(4, (torch.get_num_threads() or 1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: str,
    pos_weight: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=device))

    for batch in tqdm(dataloader, desc="Training"):
        inputs, labels, _, _ = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    pos_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=device))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels, _, _ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                else:
                    fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Train congestion classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to congestion dataset (.arrow)")
    parser.add_argument("--output", type=str, default="out/congestion_classifier.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--include_buckets",
        nargs="+",
        default=["confident_negative", "confident_positive"],
        help="Confidence buckets to include in supervised training",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio over episodes")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for episode split")
    args = parser.parse_args()

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    data = load_congestion_arrow(args.data)
    selected_indices = select_training_indices(data["confidence_buckets"], args.include_buckets)

    if len(selected_indices) == 0:
        raise ValueError(f"No samples found for include_buckets={args.include_buckets}")

    selected_episode_ids = data["episode_ids"][selected_indices]
    train_episode_ids, val_episode_ids = split_by_episode(selected_episode_ids, args.val_ratio, args.seed)

    train_indices = selected_indices[np.isin(selected_episode_ids, train_episode_ids)]
    val_indices = selected_indices[np.isin(selected_episode_ids, val_episode_ids)]

    train_dataset = IndexedCongestionDataset(data, train_indices)
    val_dataset = IndexedCongestionDataset(data, val_indices)

    print(f"Loaded {len(data['inputs'])} total samples from {args.data}")
    print(f"Training on {len(train_dataset)} confident samples from {len(np.unique(train_episode_ids))} episodes")
    print(f"Validating on {len(val_dataset)} confident samples from {len(np.unique(val_episode_ids))} episodes")

    train_labels = data["auto_labels"][train_indices]
    unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    label_distribution = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
    print(f"Train label distribution: {label_distribution}")

    num_pos = int((train_labels == 1).sum())
    num_neg = int((train_labels == 0).sum())
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Positive weight: {pos_weight:.2f}")

    train_loader = build_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, shuffle=False)

    model = CongestionClassifier(hidden_dim=args.hidden_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_metrics = {}
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, device, pos_weight)
        scheduler.step()

        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(
            f"       Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}"
        )
        print(
            f"       TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, "
            f"TN: {val_metrics['tn']}, FN: {val_metrics['fn']}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "include_buckets": args.include_buckets,
                },
                output_path,
            )
            print(f"Saved best model to {output_path}")

    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_f1": best_f1,
                "best_metrics": best_metrics,
                "include_buckets": args.include_buckets,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
            },
            f,
            indent=2,
        )

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
