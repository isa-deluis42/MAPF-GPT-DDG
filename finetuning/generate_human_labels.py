"""
Helper script to generate human labels for congestion classification.

This creates the JSON format needed for contrastive fine-tuning.

Usage:
    python finetuning/generate_human_labels.py --input dataset/congestion/train.arrow --output dataset/congestion/human_labels.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from finetuning.congestion_utils import compute_input_hash, load_congestion_arrow


def main():
    parser = argparse.ArgumentParser(description='Generate human labels template')
    parser.add_argument('--input', type=str, required=True,
                        help='Input Arrow dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file for human labels')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to include for human labeling')
    parser.add_argument('--strategy', type=str, default='uncertain',
                        choices=['random', 'uncertain', 'balanced'],
                        help='Sampling strategy for human labels')
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    data = load_congestion_arrow(args.input)
    inputs = data["inputs"]
    labels = data["auto_labels"]
    confidence_buckets = data["confidence_buckets"]
    diffs = data["diffs"]
    
    print(f"Loaded {len(labels)} samples")
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"  Label distribution: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")
    
    # Select samples for human labeling
    if args.strategy == 'random':
        indices = np.random.choice(len(labels), min(args.num_samples, len(labels)), replace=False)
    elif args.strategy == 'uncertain':
        uncertain_indices = np.where(confidence_buckets == "uncertain")[0]
        if len(uncertain_indices) == 0:
            raise ValueError("No uncertain samples found in dataset.")
        indices = np.random.choice(
            uncertain_indices,
            min(args.num_samples, len(uncertain_indices)),
            replace=False,
        )
    elif args.strategy == 'balanced':
        # Sample equal from each class
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        n_per_class = args.num_samples // 2
        pos_sample = np.random.choice(pos_indices, min(n_per_class, len(pos_indices)), replace=False)
        neg_sample = np.random.choice(neg_indices, min(n_per_class, len(neg_indices)), replace=False)
        indices = np.concatenate([pos_sample, neg_sample])
    
    # Generate human label template
    human_labels = []
    for idx in tqdm(indices, desc="Generating label template"):
        input_hash = compute_input_hash(inputs[idx])
        human_labels.append({
            "input_hash": input_hash,
            "input": inputs[idx].tolist(),
            "human_label": int(labels[idx]),  # Start with auto label as default
            "auto_label": int(labels[idx]),
            "confidence_bucket": str(confidence_buckets[idx]),
            "diff": int(diffs[idx]),
            "notes": ""
        })
    
    # Save template
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(human_labels, f, indent=2)
    
    print(f"\nSaved {len(human_labels)} samples to {output_path}")
    print("\n=== Next Steps ===")
    print("1. Open the JSON file and review the 'human_label' field")
    print("2. Change labels where you disagree with the auto-label")
    print("3. Add optional notes in the 'notes' field")
    print("4. Run contrastive fine-tuning:")
    print(f"   python finetuning/contrastive_finetune.py \\")
    print(f"       --base_model out/congestion_classifier.pt \\")
    print(f"       --data {args.input} \\")
    print(f"       --human_labels {args.output} \\")
    print(f"       --output out/congestion_classifier_refined.pt")


if __name__ == '__main__':
    main()
