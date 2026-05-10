"""
Augment segment-ranking CNN rollout data with label-preserving map symmetries.

This script operates on the checkpoint rollout `.npz` files used by
train_segment_classifier.py, not on the flat congestion Arrow dataset. It applies
spatial transforms to obstacles, positions, and goals while preserving
segment_diffs, so the generated files can be included directly under the
--data root passed to train_segment_classifier.py.

Example:
    python augment_segment_rollouts.py \
        --input dataset/held_out/ckpt _0 \
        --output dataset/held_out_aug/ckpt_0 \
        --transforms hflip vflip rot180
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


CoordTransform = Callable[[np.ndarray, int, int], np.ndarray]


SHAPE_PRESERVING_TRANSFORMS = ("hflip", "vflip", "rot180")
ALL_TRANSFORMS = SHAPE_PRESERVING_TRANSFORMS + ("rot90", "rot270", "transpose")
REQUIRED_KEYS = ("obstacles", "positions", "goals", "segment_diffs")


def collect_npz_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"Input file must be a .npz file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(input_path.glob(pattern))


def transform_hflip(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    out = coords.copy()
    out[..., 1] = width - 1 - out[..., 1]
    return out


def transform_vflip(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    out = coords.copy()
    out[..., 0] = height - 1 - out[..., 0]
    return out


def transform_rot180(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    out = coords.copy()
    out[..., 0] = height - 1 - out[..., 0]
    out[..., 1] = width - 1 - out[..., 1]
    return out


def transform_rot90(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    # np.rot90(..., k=1) is counter-clockwise: old (r, c) -> new (W - 1 - c, r).
    out = coords.copy()
    old_r = coords[..., 0]
    old_c = coords[..., 1]
    out[..., 0] = width - 1 - old_c
    out[..., 1] = old_r
    return out


def transform_rot270(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    # np.rot90(..., k=3) is clockwise: old (r, c) -> new (c, H - 1 - r).
    out = coords.copy()
    old_r = coords[..., 0]
    old_c = coords[..., 1]
    out[..., 0] = old_c
    out[..., 1] = height - 1 - old_r
    return out


def transform_transpose(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    out = coords.copy()
    out[..., 0] = coords[..., 1]
    out[..., 1] = coords[..., 0]
    return out


TRANSFORM_COORDS: Dict[str, CoordTransform] = {
    "hflip": transform_hflip,
    "vflip": transform_vflip,
    "rot180": transform_rot180,
    "rot90": transform_rot90,
    "rot270": transform_rot270,
    "transpose": transform_transpose,
}


def transform_obstacles(obstacles: np.ndarray, transform: str) -> np.ndarray:
    if transform == "hflip":
        return obstacles[:, ::-1].copy()
    if transform == "vflip":
        return obstacles[::-1, :].copy()
    if transform == "rot180":
        return np.rot90(obstacles, k=2).copy()
    if transform == "rot90":
        return np.rot90(obstacles, k=1).copy()
    if transform == "rot270":
        return np.rot90(obstacles, k=3).copy()
    if transform == "transpose":
        return obstacles.T.copy()
    raise ValueError(f"Unknown transform: {transform}")


def validate_episode(data: np.lib.npyio.NpzFile, path: Path) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in data.files]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    obstacles = data["obstacles"]
    positions = data["positions"]
    goals = data["goals"]

    if obstacles.ndim != 2:
        raise ValueError(f"{path}: obstacles must have shape (H, W), got {obstacles.shape}")
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"{path}: positions must have shape (T, N, 2), got {positions.shape}")
    if goals.ndim != 2 or goals.shape[-1] != 2:
        raise ValueError(f"{path}: goals must have shape (N, 2), got {goals.shape}")


def validate_coordinates(coords: np.ndarray, height: int, width: int, label: str, source_path: Path) -> None:
    if coords.size == 0:
        return

    rows = coords[..., 0]
    cols = coords[..., 1]
    if rows.min() < 0 or rows.max() >= height or cols.min() < 0 or cols.max() >= width:
        raise ValueError(
            f"{source_path}: transformed {label} coordinates are out of bounds for "
            f"shape ({height}, {width})"
        )


def augmented_payload(data: np.lib.npyio.NpzFile, transform: str, source_path: Path) -> Dict[str, np.ndarray]:
    validate_episode(data, source_path)

    obstacles = data["obstacles"]
    positions = data["positions"]
    goals = data["goals"]
    height, width = obstacles.shape

    transformed_obstacles = transform_obstacles(obstacles, transform).astype(obstacles.dtype, copy=False)
    transformed_positions = TRANSFORM_COORDS[transform](positions, height, width).astype(positions.dtype, copy=False)
    transformed_goals = TRANSFORM_COORDS[transform](goals, height, width).astype(goals.dtype, copy=False)

    new_height, new_width = transformed_obstacles.shape
    validate_coordinates(transformed_positions, new_height, new_width, "positions", source_path)
    validate_coordinates(transformed_goals, new_height, new_width, "goals", source_path)

    payload = {key: data[key] for key in data.files}
    payload["obstacles"] = transformed_obstacles
    payload["positions"] = transformed_positions
    payload["goals"] = transformed_goals
    payload["augmentation_transform"] = np.asarray(transform)
    payload["augmentation_source_path"] = np.asarray(str(source_path))
    payload["augmentation_label_preserving"] = np.asarray(True)

    return payload


def output_path_for(input_file: Path, input_root: Path, output_root: Path, transform: str, flat: bool) -> Path:
    if flat or input_root.is_file():
        relative_parent = Path()
    else:
        relative_parent = input_file.parent.relative_to(input_root)

    return output_root / relative_parent / f"{input_file.stem}__aug_{transform}.npz"


def should_skip_augmented(data: np.lib.npyio.NpzFile, skip_augmented: bool) -> bool:
    return skip_augmented and "augmentation_transform" in data.files


def augment_file(
    input_file: Path,
    input_root: Path,
    output_root: Path,
    transforms: Iterable[str],
    flat: bool,
    overwrite: bool,
    skip_augmented: bool,
) -> Tuple[int, int]:
    written = 0
    skipped = 0
    with np.load(str(input_file), allow_pickle=True) as data:
        if should_skip_augmented(data, skip_augmented):
            return 0, len(tuple(transforms))

        for transform in transforms:
            out_path = output_path_for(input_file, input_root, output_root, transform, flat)
            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            payload = augmented_payload(data, transform, input_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(out_path), **payload)
            written += 1

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment checkpoint rollout .npz files for the segment-ranking CNN pipeline"
    )
    parser.add_argument("--input", required=True, help="Input .npz file or directory of checkpoint rollout .npz files")
    parser.add_argument("--output", required=True, help="Output directory for augmented .npz files")
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=ALL_TRANSFORMS,
        default=list(SHAPE_PRESERVING_TRANSFORMS),
        help="Spatial transforms to apply. Defaults to shape-preserving flips/rot180.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively for .npz files")
    parser.add_argument("--flat", action="store_true", help="Write all augmented files directly into --output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing augmented files")
    parser.add_argument(
        "--include_augmented_sources",
        action="store_true",
        help="Also augment input files that already contain augmentation_transform metadata",
    )
    parser.add_argument("--manifest", default=None, help="Optional JSON manifest path summarizing written/skipped files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    npz_files = collect_npz_files(input_path, args.recursive)
    if not npz_files:
        raise ValueError(f"No .npz files found under {input_path}")

    total_written = 0
    total_skipped = 0
    errors = []
    manifest_rows = []

    for input_file in npz_files:
        try:
            written, skipped = augment_file(
                input_file=input_file,
                input_root=input_path,
                output_root=output_root,
                transforms=args.transforms,
                flat=args.flat,
                overwrite=args.overwrite,
                skip_augmented=not args.include_augmented_sources,
            )
            total_written += written
            total_skipped += skipped
            manifest_rows.append(
                {
                    "input": str(input_file),
                    "written": int(written),
                    "skipped": int(skipped),
                    "transforms": list(args.transforms),
                }
            )
        except Exception as exc:
            errors.append({"input": str(input_file), "error": str(exc)})
            print(f"ERROR {input_file}: {exc}")

    summary = {
        "input": str(input_path),
        "output": str(output_root),
        "num_input_files": len(npz_files),
        "transforms": list(args.transforms),
        "written": int(total_written),
        "skipped": int(total_skipped),
        "errors": errors,
        "files": manifest_rows,
    }

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
