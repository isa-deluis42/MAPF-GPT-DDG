import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn
from torch.utils.data import Dataset


CONFIDENT_NEGATIVE = 0
UNCERTAIN_LABEL = -1
CONFIDENT_POSITIVE = 1


def compute_input_hash(input_array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(input_array, dtype=np.int16).tobytes()).hexdigest()


def diff_to_confidence_bucket(
    diff: int,
    low_diff_threshold: int,
    high_diff_threshold: int,
) -> str:
    if diff < low_diff_threshold:
        return "confident_negative"
    if diff > high_diff_threshold:
        return "confident_positive"
    return "uncertain"


def bucket_to_label(bucket: str) -> int:
    if bucket == "confident_negative":
        return CONFIDENT_NEGATIVE
    if bucket == "confident_positive":
        return CONFIDENT_POSITIVE
    return UNCERTAIN_LABEL


def _read_episode_info(arrow_path: Path) -> List[Dict[str, Any]]:
    episode_info_path = arrow_path.parent / "episode_info.json"
    if not episode_info_path.exists():
        return []

    with open(episode_info_path, "r") as f:
        return json.load(f)


def _infer_sample_metadata_from_episode_info(
    episode_info: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(episode_info) == 0:
        empty = np.asarray([], dtype=np.int32)
        return empty, empty.astype(np.int16), empty.astype(np.int8), np.asarray([], dtype=object)

    episode_ids: List[int] = []
    diffs: List[int] = []
    auto_labels: List[int] = []
    buckets: List[str] = []

    for episode_idx, info in enumerate(episode_info):
        num_samples = int(info["num_timesteps"]) * int(info["num_agents"])
        diff = int(info["diff"])
        label = int(info["label"])
        bucket = info.get("confidence_bucket")
        if bucket is None:
            bucket = "confident_positive" if label == 1 else "confident_negative"
        episode_ids.extend([episode_idx] * num_samples)
        diffs.extend([diff] * num_samples)
        auto_labels.extend([label] * num_samples)
        buckets.extend([bucket] * num_samples)

    return (
        np.asarray(episode_ids, dtype=np.int32),
        np.asarray(diffs, dtype=np.int16),
        np.asarray(auto_labels, dtype=np.int8),
        np.asarray(buckets, dtype=object),
    )


def load_congestion_arrow(data_path: str) -> Dict[str, np.ndarray]:
    arrow_path = Path(data_path)
    with pa.memory_map(str(arrow_path), "r") as source:
        table = pa.ipc.open_file(source).read_all()

    inputs = np.stack(table["inputs"].to_numpy()).astype(np.float32)

    if "labels" in table.column_names:
        auto_labels = table["labels"].to_numpy().astype(np.int8)
    elif "auto_labels" in table.column_names:
        auto_labels = table["auto_labels"].to_numpy().astype(np.int8)
    else:
        raise ValueError("Dataset must contain 'labels' or 'auto_labels'.")

    episode_info = _read_episode_info(arrow_path)

    if "episode_ids" in table.column_names:
        episode_ids = table["episode_ids"].to_numpy().astype(np.int32)
    else:
        episode_ids, _, _, _ = _infer_sample_metadata_from_episode_info(episode_info)

    if "diffs" in table.column_names:
        diffs = table["diffs"].to_numpy().astype(np.int16)
    else:
        _, diffs, _, _ = _infer_sample_metadata_from_episode_info(episode_info)

    if "confidence_buckets" in table.column_names:
        confidence_buckets = table["confidence_buckets"].to_numpy(zero_copy_only=False)
    else:
        _, _, _, confidence_buckets = _infer_sample_metadata_from_episode_info(episode_info)

    if len(episode_ids) == 0:
        episode_ids = np.zeros(len(inputs), dtype=np.int32)
    if len(diffs) == 0:
        diffs = np.zeros(len(inputs), dtype=np.int16)
    if len(confidence_buckets) == 0:
        confidence_buckets = np.asarray(
            [
                "confident_positive" if label == 1 else "confident_negative" if label == 0 else "uncertain"
                for label in auto_labels
            ],
            dtype=object,
        )

    if len(episode_ids) != len(inputs):
        raise ValueError("Episode metadata length does not match dataset size.")

    sample_indices = np.arange(len(inputs), dtype=np.int32)
    input_hashes = np.asarray([compute_input_hash(row) for row in inputs], dtype=object)

    return {
        "inputs": inputs,
        "auto_labels": auto_labels,
        "episode_ids": episode_ids,
        "diffs": diffs,
        "confidence_buckets": np.asarray(confidence_buckets, dtype=object),
        "sample_indices": sample_indices,
        "input_hashes": input_hashes,
    }


def select_training_indices(
    confidence_buckets: np.ndarray,
    include_buckets: Optional[Sequence[str]] = None,
) -> np.ndarray:
    include_buckets = include_buckets or ("confident_negative", "confident_positive")
    mask = np.isin(confidence_buckets, list(include_buckets))
    return np.where(mask)[0]


def split_by_episode(
    episode_ids: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    unique_episode_ids = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_episode_ids)

    if len(unique_episode_ids) == 1:
        return unique_episode_ids, unique_episode_ids

    val_episode_count = max(1, int(round(len(unique_episode_ids) * val_ratio)))
    val_episode_count = min(val_episode_count, len(unique_episode_ids) - 1)

    val_episode_ids = unique_episode_ids[:val_episode_count]
    train_episode_ids = unique_episode_ids[val_episode_count:]
    return train_episode_ids, val_episode_ids


class IndexedCongestionDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], indices: Sequence[int], human_labels: Optional[Dict[str, int]] = None):
        self.inputs = data["inputs"]
        self.auto_labels = data["auto_labels"]
        self.sample_indices = data["sample_indices"]
        self.input_hashes = data["input_hashes"]
        self.indices = np.asarray(indices, dtype=np.int32)
        self.human_labels = human_labels or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx])
        input_vec = self.inputs[sample_idx]
        auto_label = int(self.auto_labels[sample_idx])
        input_hash = str(self.input_hashes[sample_idx])
        label = self.human_labels.get(input_hash, auto_label)

        return (
            torch.tensor(input_vec, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(auto_label, dtype=torch.long),
            torch.tensor(sample_idx, dtype=torch.long),
        )


class CongestionClassifier(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CongestionClassifierWithEmbedding(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(x)
        logits = self.head(embedding)
        return logits, embedding
