"""Walk a dataset directory and write two path manifests partitioned by map_seed.

Usage:
    python build_seed_manifests.py
    python build_seed_manifests.py --data-dir dataset/held_out \
        --train-output train_seeds_manifest.txt \
        --val-output val_seeds_manifest.txt

The manifests feed query.py via --manifest. The split (train vs val map seeds)
is the canonical hold-out from held_out_seed_set; rebuild whenever those sets
change.
"""

import argparse
from pathlib import Path

import numpy as np

from held_out_seed_set import TRAIN_MAP_SEEDS, VAL_MAP_SEEDS


def build(data_dir: Path, train_output: Path, val_output: Path):
    train_lines = [
        f"# Train-map manifest. Source for AL/random elicitation on Stages 2/3/4.",
        f"# TRAIN_MAP_SEEDS = {sorted(TRAIN_MAP_SEEDS)}",
    ]
    val_lines = [
        f"# Val-map manifest. Source for held-out human val (random sampling, eval-only).",
        f"# VAL_MAP_SEEDS = {sorted(VAL_MAP_SEEDS)}",
    ]
    n_train = n_val = n_other = n_failed = 0

    for npz in sorted(data_dir.rglob("*.npz")):
        try:
            seed = int(np.load(str(npz), allow_pickle=True)["map_seed"])
        except Exception as e:
            n_failed += 1
            print(f"  failed: {npz} ({e})")
            continue

        if seed in TRAIN_MAP_SEEDS:
            train_lines.append(str(npz))
            n_train += 1
        elif seed in VAL_MAP_SEEDS:
            val_lines.append(str(npz))
            n_val += 1
        else:
            n_other += 1

    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_text("\n".join(train_lines) + "\n")
    val_output.write_text("\n".join(val_lines) + "\n")

    print(f"{train_output}: {n_train} rollouts")
    print(f"{val_output}: {n_val} rollouts")
    if n_other:
        print(f"(skipped {n_other} on map seeds outside both sets)")
    if n_failed:
        print(f"(failed to read {n_failed} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build train and val map-seed manifests from a dataset directory."
    )
    parser.add_argument("--data-dir", default="dataset/held_out", help="Root .npz directory")
    parser.add_argument("--train-output", default="train_seeds_manifest.txt")
    parser.add_argument("--val-output", default="val_seeds_manifest.txt")
    args = parser.parse_args()

    build(Path(args.data_dir), Path(args.train_output), Path(args.val_output))
