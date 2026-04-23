"""
Contrastive Fine-Tuning for Congestion Classifier

Phase 2: Use human labels to refine the classifier using contrastive learning.

The key idea:
- "Positive pairs" = two timesteps both labeled "fail" by humans
- "Negative pairs" = one labeled "fail", one labeled "pass"
- Loss: maximize similarity for positive, minimize for negative

Usage:
    python finetuning/contrastive_finetune.py \
        --base_model out/congestion_classifier.pt \
        --human_labels dataset/congestion/human_labels.json \
        --output out/congestion_classifier_refined.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from finetuning.congestion_utils import (
    CongestionClassifierWithEmbedding,
    IndexedCongestionDataset,
    load_congestion_arrow,
)


# ============================================================================
# Contrastive Loss
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss.
    
    Given embeddings, pull together samples with same label,
    push apart samples with different labels.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, hidden_dim)
            labels: (batch_size,)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()
        
        # For contrastive learning, we want to contrast against all other samples
        # So we use the full sim matrix but mask out self-comparisons
        diagonal_mask = torch.eye(labels.size(0), device=embeddings.device)
        mask = mask - diagonal_mask  # Remove self-comparisons
        
        # InfoNCE: for each sample, contrast positive vs negative
        # exp(sim) / sum(exp(sim)) for positive pairs
        exp_sim = torch.exp(sim_matrix)
        
        # Denominator: sum of exp(sim) for all pairs
        denom = exp_sim.sum(dim=1, keepdim=True)
        
        # Numerator: only positive pairs
        numerator = (exp_sim * mask).sum(dim=1)
        
        # Loss = -log(numerator/denom)
        loss = -torch.log(numerator / denom.squeeze() + 1e-8)
        
        # Only compute loss where there are positive pairs
        num_positives = mask.sum(dim=1)
        valid_loss = loss[num_positives > 0]
        
        return valid_loss.mean()


class CombinedLoss(nn.Module):
    """Combined cross-entropy + contrastive loss."""
    
    def __init__(self, ce_weight: float = 1.0, contrastive_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight
        self.ce = nn.CrossEntropyLoss()
        self.contrastive = ContrastiveLoss()
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, labels)
        
        if embeddings is not None and self.contrastive_weight > 0:
            cont_loss = self.contrastive(embeddings, labels)
            return self.ce_weight * ce_loss + self.contrastive_weight * cont_loss
        
        return ce_loss


# ============================================================================
# Dataset with Human Labels
# ============================================================================

class HumanLabeledDataset(IndexedCongestionDataset):
    def __init__(self, data_path: str, human_labels_path: str):
        self.data_path = Path(data_path)
        self.human_labels_path = Path(human_labels_path)
        self._load_data()
    
    def _load_data(self):
        data = load_congestion_arrow(str(self.data_path))

        with open(self.human_labels_path, 'r') as f:
            human_data = json.load(f)
        human_label_map = {
            item["input_hash"]: int(item["human_label"])
            for item in human_data
            if "input_hash" in item
        }
        matched_indices = np.where(np.isin(data["input_hashes"], list(human_label_map.keys())))[0]

        super().__init__(data, matched_indices, human_label_map)

        print(f"Loaded {len(data['auto_labels'])} base samples")
        print(f"Found {len(self.indices)} human-labeled samples")


# ============================================================================
# Training
# ============================================================================

def train_contrastive(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: str,
    pos_weight: float,
    use_contrastive: bool = True
) -> Dict[str, float]:
    """Train with combined CE + contrastive loss."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track human vs auto label agreement
    human_agree = 0
    human_disagree = 0
    
    criterion = CombinedLoss(
        ce_weight=1.0, 
        contrastive_weight=0.5 if use_contrastive else 0.0
    ).to(device)
    
    ce_only = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=device)
    )
    
    for batch in tqdm(dataloader, desc="Training"):
        # Handle 2 or 3 tensors (with or without auto labels)
        inputs, human_labels, auto_labels, _ = batch
        
        inputs = inputs.to(device)
        human_labels = human_labels.to(device)
        auto_labels = auto_labels.to(device)
        
        optimizer.zero_grad()
        
        logits, embeddings = model(inputs)
        
        # Use human labels for CE loss
        if use_contrastive:
            loss = criterion(logits, human_labels, embeddings)
        else:
            loss = ce_only(logits, human_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == human_labels).sum().item()
        total += human_labels.size(0)
        
        # Track agreement
        human_agree += ((preds == human_labels) & (auto_labels == human_labels)).sum().item()
        human_disagree += ((preds == human_labels) & (auto_labels != human_labels)).sum().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'human_agree': human_agree,
        'human_disagree': human_disagree
    }


def evaluate_human(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """Evaluate on human-labeled data."""
    model.eval()
    
    correct = 0
    total = 0
    
    # Compare to auto labels
    auto_correct = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, human_labels, auto_labels, _ = batch
            
            inputs = inputs.to(device)
            human_labels = human_labels.to(device)
            
            logits, _ = model(inputs)
            preds = logits.argmax(dim=1)
            
            correct += (preds == human_labels).sum().item()
            total += human_labels.size(0)
            
            auto_correct += (auto_labels == human_labels).sum().item()
    
    return {
        'accuracy': correct / total,
        'auto_accuracy': auto_correct / total,
        'improvement': (correct - auto_correct) / total if total > 0 else 0
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Contrastive fine-tuning')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to base classifier model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to congestion dataset (.arrow)')
    parser.add_argument('--human_labels', type=str, required=True,
                        help='Path to human labels JSON')
    parser.add_argument('--output', type=str, default='out/congestion_classifier_refined.pt',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (smaller than initial training)')
    parser.add_argument('--no_contrastive', action='store_true',
                        help='Disable contrastive loss, use CE only')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    # Load human-labeled dataset
    dataset = HumanLabeledDataset(args.data, args.human_labels)
    
    if len(dataset) == 0:
        print("ERROR: No human-labeled samples found!")
        print(f"Expected format: list of {{'input': [256], 'human_label': 0/1}} in {args.human_labels}")
        return
    
    print(f"\nHuman-labeled samples: {len(dataset)}")
    
    # Compute class distribution
    labels_list = []
    for idx in range(len(dataset)):
        _, hl, _, _ = dataset[idx]
        labels_list.append(hl.item())
    labels_array = np.array(labels_list)
    unique, counts = np.unique(labels_array, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Load base model
    model = CongestionClassifierWithEmbedding().to(args.device)
    checkpoint = torch.load(args.base_model, map_location=args.device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    print(f"\nLoaded base model from {args.base_model}")
    
    # Optimizer (lower LR for fine-tuning)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        train_metrics = train_contrastive(
            model, dataloader, optimizer, args.device,
            pos_weight=1.0,  # Balanced since human-labeled
            use_contrastive=not args.no_contrastive
        )
        
        val_metrics = evaluate_human(model, dataloader, args.device)
        
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}")
        print(f"       Auto baseline: {val_metrics['auto_accuracy']:.4f}")
        print(f"       Improvement: {val_metrics['improvement']:+.4f}")
    
    # Save final model
    torch.save(model.state_dict(), args.output)
    print(f"\nSaved refined model to {args.output}")


if __name__ == '__main__':
    main()
