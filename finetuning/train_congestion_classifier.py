"""
Congestion Classifier Training

Phase 1: Train a binary classifier to predict failure (congestion) 
using automatically generated labels from solver diffs.

Usage:
    python finetuning/train_congestion_classifier.py --data dataset/congestion/train.arrow
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import pyarrow as pa


# ============================================================================
# Dataset
# ============================================================================

class CongestionDataset(Dataset):
    """Dataset for congestion classification."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self._load_data()
    
    def _load_data(self):
        """Load data from Arrow file."""
        with pa.memory_map(self.data_path, 'r') as source:
            table = pa.ipc.open_file(source).read_all()
            self.inputs = np.stack(table['inputs'].to_numpy())
            self.labels = table['labels'].to_numpy()
        
        print(f"Loaded {len(self.labels)} samples")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Label distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ============================================================================
# Model
# ============================================================================

class CongestionClassifier(nn.Module):
    """
    Simple classifier on raw 256-dim inputs.
    
    Architecture:
    - Input: 256-dim tensor (same as MAPF-GPT observation)
    - Linear layers with GELU activation
    - Output: 2-class (pass/fail)
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: str,
    pos_weight: float
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Weighted cross entropy for class imbalance
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=device)
    )
    
    for batch in tqdm(dataloader, desc="Training"):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    pos_weight: float
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    tp = 0  # true positives (fail correctly predicted)
    fp = 0  # false positives
    tn = 0  # true negatives (pass correctly predicted)
    fn = 0  # false negatives
    
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=device)
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Confusion matrix
            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                elif pred == 0 and label == 1:
                    fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train congestion classifier')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to congestion dataset (.arrow)')
    parser.add_argument('--output', type=str, default='out/congestion_classifier.pt',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    # Load dataset
    dataset = CongestionDataset(args.data)
    
    # Compute class imbalance ratio for weighted loss
    labels = dataset.labels
    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"\nClass imbalance: {num_neg} negative, {num_pos} positive")
    print(f"Positive weight: {pos_weight:.2f}")
    
    # Create dataloaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = CongestionClassifier(hidden_dim=args.hidden_dim).to(args.device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, pos_weight
        )
        val_metrics = evaluate(
            model, val_loader, args.device, pos_weight
        )
        
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"       Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"       TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, TN: {val_metrics['tn']}, FN: {val_metrics['fn']}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model to {args.output}")
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()