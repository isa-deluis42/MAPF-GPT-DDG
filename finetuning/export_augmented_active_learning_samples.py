"""
Export augmented active-learning samples for congestion classification.

1. Pick labeled seed examples, usually from the rare class.
2. Generate nearby synthetic examples by perturbing the input.
3. Score synthetic examples with the current classifier.
4. Export the most informative synthetic examples for review.

The exported JSON is intentionally similar to export_active_learning_samples.py
so reviewed labels can later be merged into the training set.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from finetuning.congestion_utils import compute_input_hash, load_congestion_arrow
from finetuning.export_active_learning_samples import load_model


LABEL_NAME_TO_VALUE = {
    "negative": 0,
    "positive": 1,
}


def parse_mutable_indices(raw: Optional[str], input_dim: int) -> np.ndarray:
    """
    Parse mutable feature indices.

    Examples:
        None        -> all features mutable
        "0,1,2"     -> indices [0, 1, 2]
        "0:64"      -> indices [0, 1, ..., 63]
        "0:64,80"   -> indices [0..63, 80]

    For safety, you can later pass a smaller mutable-index list if you know
    which parts of the 256-dimensional observation are semantically safe to
    perturb.
    """
    if raw is None or raw.strip() == "":
        return np.arange(input_dim, dtype=np.int32)

    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            start_raw, end_raw = part.split(":", maxsplit=1)
            start = int(start_raw)
            end = int(end_raw)
            indices.extend(range(start, end))
        else:
            indices.append(int(part))

    unique = np.unique(np.asarray(indices, dtype=np.int32))
    if len(unique) == 0:
        raise ValueError("No mutable indices were parsed.")

    if unique.min() < 0 or unique.max() >= input_dim:
        raise ValueError(f"Mutable indices must be in [0, {input_dim}).")

    return unique


def choose_target_label(auto_labels: np.ndarray, target_label: str) -> int:
    """
    Choose which class to augment.

    'rarest' picks the rarer class among 0 and 1, ignoring uncertain -1 labels.
    """
    if target_label in LABEL_NAME_TO_VALUE:
        return LABEL_NAME_TO_VALUE[target_label]

    if target_label != "rarest":
        raise ValueError("--target_label must be one of: rarest, positive, negative")

    counts = {
        label: int((auto_labels == label).sum())
        for label in (0, 1)
    }

    if counts[0] == 0 and counts[1] == 0:
        raise ValueError("No labeled positive/negative samples found.")

    if counts[0] == 0:
        return 1
    if counts[1] == 0:
        return 0

    return 0 if counts[0] <= counts[1] else 1


def select_seed_indices(
    data: Dict[str, np.ndarray],
    target_label: int,
    max_seeds: int,
    seed: int,
    include_buckets: Sequence[str],
) -> np.ndarray:
    """
    Select real labeled examples to use as augmentation seeds.

    By default, this should use confident labels only, because augmenting noisy
    labels can amplify mistakes.
    """
    label_mask = data["auto_labels"] == target_label
    bucket_mask = np.isin(data["confidence_buckets"], list(include_buckets))
    candidate_indices = np.where(label_mask & bucket_mask)[0]

    if len(candidate_indices) == 0:
        raise ValueError(
            f"No seed samples found for target_label={target_label} "
            f"and include_buckets={list(include_buckets)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(candidate_indices)
    return candidate_indices[:max_seeds]


def augment_input(
    input_vec: np.ndarray,
    rng: np.random.Generator,
    mutable_indices: np.ndarray,
    mutation_rate: float,
    jitter_radius: int,
    min_value: int,
    max_value: int,
) -> np.ndarray:
    """
    Create one augmented feature vector.

    This implementation assumes the 256-dimensional inputs are integer-like.
    It applies sparse integer jitter to a subset of mutable indices, then clips
    back to an int8-compatible range.

    If you later identify semantic feature groups, replace this with
    domain-aware transforms such as rotations, flips, local obstacle edits,
    agent swaps, or target jitter.
    """
    augmented = np.asarray(input_vec, dtype=np.int16).copy()

    mutate_count = max(1, int(round(len(mutable_indices) * mutation_rate)))
    chosen = rng.choice(mutable_indices, size=mutate_count, replace=False)

    deltas = rng.integers(
        low=-jitter_radius,
        high=jitter_radius + 1,
        size=mutate_count,
        dtype=np.int16,
    )

    # Avoid no-op deltas when possible.
    if jitter_radius > 0:
        zero_mask = deltas == 0
        replacement = rng.choice(
            np.asarray([-jitter_radius, jitter_radius], dtype=np.int16),
            size=int(zero_mask.sum()),
        )
        deltas[zero_mask] = replacement

    augmented[chosen] += deltas
    augmented = np.clip(augmented, min_value, max_value)

    return augmented.astype(np.int8)


def generate_augmented_candidates(
    data: Dict[str, np.ndarray],
    seed_indices: np.ndarray,
    candidates_per_seed: int,
    mutable_indices: np.ndarray,
    mutation_rate: float,
    jitter_radius: int,
    min_value: int,
    max_value: int,
    seed: int,
) -> List[Dict]:
    """
    Generate synthetic candidate rows from selected seed examples.
    """
    rng = np.random.default_rng(seed)
    candidates: List[Dict] = []
    seen_hashes = set(str(h) for h in data["input_hashes"])

    synthetic_id = 0
    for source_idx in seed_indices:
        source_input = data["inputs"][source_idx]
        source_hash = str(data["input_hashes"][source_idx])

        for _ in range(candidates_per_seed):
            augmented = augment_input(
                input_vec=source_input,
                rng=rng,
                mutable_indices=mutable_indices,
                mutation_rate=mutation_rate,
                jitter_radius=jitter_radius,
                min_value=min_value,
                max_value=max_value,
            )
            augmented_hash = compute_input_hash(augmented)

            # Drop exact duplicates of existing or already-generated inputs.
            if augmented_hash in seen_hashes:
                continue

            seen_hashes.add(augmented_hash)

            candidates.append(
                {
                    "synthetic_id": synthetic_id,
                    "source_sample_index": int(data["sample_indices"][source_idx]),
                    "source_episode_id": int(data["episode_ids"][source_idx]),
                    "source_diff": int(data["diffs"][source_idx]),
                    "source_confidence_bucket": str(data["confidence_buckets"][source_idx]),
                    "source_input_hash": source_hash,
                    "source_auto_label": int(data["auto_labels"][source_idx]),
                    "input_hash": augmented_hash,
                    "input": augmented.astype(int).tolist(),
                    "synthetic": True,
                    "query_strategy": "augment",
                    "augment_transform": "sparse_integer_jitter",
                    "mutation_rate": float(mutation_rate),
                    "jitter_radius": int(jitter_radius),
                    "human_label": None,
                    "notes": "",
                }
            )
            synthetic_id += 1

    return candidates


def score_augmented_candidates(
    model: torch.nn.Module,
    candidates: List[Dict],
    device: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Score synthetic candidates with the current classifier.

    Returns:
        probs: shape [N, 2]
        pred_labels: shape [N]
        pred_confidence: shape [N]
        pred_entropy: shape [N]
        uncertainty_margin: abs(P(fail) - 0.5), lower is more informative
    """
    if len(candidates) == 0:
        raise ValueError("No augmented candidates to score.")

    inputs = np.asarray([row["input"] for row in candidates], dtype=np.float32)
    probabilities: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(inputs), batch_size), desc="Scoring augmented"):
            end = min(start + batch_size, len(inputs))
            batch = torch.tensor(inputs[start:end], dtype=torch.float32, device=device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs)

    probs = np.concatenate(probabilities, axis=0)
    pred_labels = probs.argmax(axis=1).astype(np.int8)
    pred_confidence = probs.max(axis=1)
    pred_entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
    uncertainty_margin = np.abs(probs[:, 1] - 0.5)

    return {
        "probs": probs,
        "pred_labels": pred_labels,
        "pred_confidence": pred_confidence,
        "pred_entropy": pred_entropy,
        "uncertainty_margin": uncertainty_margin,
    }


def select_augmented_queries(
    candidates: List[Dict],
    scores: Dict[str, np.ndarray],
    num_samples: int,
    selection: str,
) -> List[Dict]:
    """
    Select the best augmented queries for review.

    selection:
        uncertainty -> closest P(fail) to 0.5
        entropy     -> highest predictive entropy
        low_conf    -> lowest max probability
    """
    if selection == "uncertainty":
        order = np.argsort(scores["uncertainty_margin"])
    elif selection == "entropy":
        order = np.argsort(-scores["pred_entropy"])
    elif selection == "low_conf":
        order = np.argsort(scores["pred_confidence"])
    else:
        raise ValueError("--selection must be one of: uncertainty, entropy, low_conf")

    selected = []
    for rank, idx in enumerate(order[:num_samples]):
        row = dict(candidates[int(idx)])
        row.update(
            {
                "rank": int(rank),
                "model_pred": int(scores["pred_labels"][idx]),
                "model_prob_pass": float(scores["probs"][idx][0]),
                "model_prob_fail": float(scores["probs"][idx][1]),
                "model_confidence": float(scores["pred_confidence"][idx]),
                "model_entropy": float(scores["pred_entropy"][idx]),
                "uncertainty_margin": float(scores["uncertainty_margin"][idx]),
                "review_bucket": "augment",
            }
        )
        selected.append(row)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Export augmented active-learning samples for human/expert review"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to congestion dataset (.arrow)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained classifier checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of review samples to export")
    parser.add_argument("--max_seeds", type=int, default=200, help="Maximum number of source examples to augment")
    parser.add_argument("--candidates_per_seed", type=int, default=10, help="Synthetic candidates per seed")
    parser.add_argument(
        "--target_label",
        choices=["rarest", "positive", "negative"],
        default="rarest",
        help="Which source label to augment",
    )
    parser.add_argument(
        "--include_buckets",
        nargs="+",
        default=["confident_negative", "confident_positive"],
        help="Source confidence buckets allowed as augmentation seeds",
    )
    parser.add_argument(
        "--selection",
        choices=["uncertainty", "entropy", "low_conf"],
        default="uncertainty",
        help="How to rank augmented candidates for review",
    )
    parser.add_argument(
        "--mutable_indices",
        type=str,
        default=None,
        help=(
            "Comma/range list of mutable feature indices, e.g. '0:64,80,81'. "
            "Default: all features."
        ),
    )
    parser.add_argument("--mutation_rate", type=float, default=0.05, help="Fraction of mutable features to perturb")
    parser.add_argument("--jitter_radius", type=int, default=1, help="Integer perturbation radius")
    parser.add_argument("--min_value", type=int, default=-128, help="Minimum clipped feature value")
    parser.add_argument("--max_value", type=int, default=127, help="Maximum clipped feature value")
    parser.add_argument("--batch_size", type=int, default=1024, help="Scoring batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--review_round", type=int, default=1, help="Active-learning round ID")
    args = parser.parse_args()

    if args.mutation_rate <= 0.0 or args.mutation_rate > 1.0:
        raise ValueError("--mutation_rate must be in (0, 1].")
    if args.jitter_radius < 1:
        raise ValueError("--jitter_radius must be >= 1.")

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"

    data = load_congestion_arrow(args.data)
    model = load_model(args.model, device)

    input_dim = int(data["inputs"].shape[1])
    mutable_indices = parse_mutable_indices(args.mutable_indices, input_dim)

    target_label = choose_target_label(data["auto_labels"], args.target_label)
    seed_indices = select_seed_indices(
        data=data,
        target_label=target_label,
        max_seeds=args.max_seeds,
        seed=args.seed,
        include_buckets=args.include_buckets,
    )

    print(f"Loaded {len(data['inputs'])} real samples from {args.data}")
    print(f"Selected {len(seed_indices)} augmentation seeds with target_label={target_label}")
    print(f"Using {len(mutable_indices)} mutable feature indices")

    candidates = generate_augmented_candidates(
        data=data,
        seed_indices=seed_indices,
        candidates_per_seed=args.candidates_per_seed,
        mutable_indices=mutable_indices,
        mutation_rate=args.mutation_rate,
        jitter_radius=args.jitter_radius,
        min_value=args.min_value,
        max_value=args.max_value,
        seed=args.seed,
    )

    print(f"Generated {len(candidates)} unique augmented candidates")

    if len(candidates) == 0:
        raise ValueError("No unique augmented candidates were generated.")

    scores = score_augmented_candidates(
        model=model,
        candidates=candidates,
        device=device,
        batch_size=args.batch_size,
    )

    selected = select_augmented_queries(
        candidates=candidates,
        scores=scores,
        num_samples=args.num_samples,
        selection=args.selection,
    )

    for row in selected:
        row["review_round"] = int(args.review_round)
        row["target_label"] = int(target_label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"Exported {len(selected)} augmented review samples to {output_path}")


if __name__ == "__main__":
    main()