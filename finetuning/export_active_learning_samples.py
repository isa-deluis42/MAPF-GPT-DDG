"""
Export human-review candidates for congestion classification.

Priority order:
1. Auto-label/model disagreements
2. Uncertain auto-label bucket
3. Lowest-confidence predictions
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from finetuning.congestion_utils import CongestionClassifier, load_congestion_arrow


def load_model(model_path: str, device: str) -> CongestionClassifier:
    checkpoint = torch.load(model_path, map_location=device)
    hidden_dim = checkpoint.get("hidden_dim", 512) if isinstance(checkpoint, dict) else 512
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = CongestionClassifier(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def score_samples(model: CongestionClassifier, inputs: np.ndarray, device: str, batch_size: int) -> Dict[str, np.ndarray]:
    probabilities: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(inputs), batch_size), desc="Scoring"):
            end = min(start + batch_size, len(inputs))
            batch = torch.tensor(inputs[start:end], dtype=torch.float32, device=device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs)

    probs = np.concatenate(probabilities, axis=0)
    pred_labels = probs.argmax(axis=1).astype(np.int8)
    pred_confidence = probs.max(axis=1)
    pred_entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
    return {
        "probs": probs,
        "pred_labels": pred_labels,
        "pred_confidence": pred_confidence,
        "pred_entropy": pred_entropy,
    }


def priority_tuple(bucket: str, disagreement: bool, confidence: float):
    bucket_rank = {
        "disagreement": 0,
        "uncertain": 1,
        "confident": 2,
    }[bucket]
    return (bucket_rank, 0 if disagreement else 1, confidence)


def main():
    parser = argparse.ArgumentParser(description="Export active-learning samples for human review")
    parser.add_argument("--data", type=str, required=True, help="Path to congestion dataset (.arrow)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained classifier checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of review samples to export")
    parser.add_argument("--batch_size", type=int, default=1024, help="Scoring batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--include_agreeing_samples",
        action="store_true",
        help="Include agreeing samples after disagreement and uncertain cases are exhausted",
    )
    args = parser.parse_args()

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    data = load_congestion_arrow(args.data)
    model = load_model(args.model, device)
    scores = score_samples(model, data["inputs"], device, args.batch_size)

    candidate_rows = []
    for idx in range(len(data["inputs"])):
        auto_label = int(data["auto_labels"][idx])
        pred_label = int(scores["pred_labels"][idx])
        confidence_bucket = str(data["confidence_buckets"][idx])
        disagreement = auto_label in (0, 1) and pred_label != auto_label

        review_bucket = "confident"
        if disagreement:
            review_bucket = "disagreement"
        elif confidence_bucket == "uncertain":
            review_bucket = "uncertain"
        elif not args.include_agreeing_samples:
            continue

        candidate_rows.append(
            {
                "sample_index": int(data["sample_indices"][idx]),
                "episode_id": int(data["episode_ids"][idx]),
                "diff": int(data["diffs"][idx]),
                "confidence_bucket": confidence_bucket,
                "input_hash": str(data["input_hashes"][idx]),
                "input": data["inputs"][idx].astype(int).tolist(),
                "auto_label": auto_label,
                "model_pred": pred_label,
                "model_prob_pass": float(scores["probs"][idx][0]),
                "model_prob_fail": float(scores["probs"][idx][1]),
                "model_confidence": float(scores["pred_confidence"][idx]),
                "model_entropy": float(scores["pred_entropy"][idx]),
                "review_bucket": review_bucket,
                "human_label": auto_label if auto_label in (0, 1) else None,
                "notes": "",
            }
        )

    candidate_rows.sort(
        key=lambda row: priority_tuple(
            row["review_bucket"],
            row["review_bucket"] == "disagreement",
            row["model_confidence"],
        )
    )

    selected = candidate_rows[: args.num_samples]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"Exported {len(selected)} review samples to {output_path}")


if __name__ == "__main__":
    main()
