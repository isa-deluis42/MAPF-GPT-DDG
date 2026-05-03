"""Write a manifest of .npz rollouts whose segment_diffs contain selected values.

Used to define a curated 'filtered' subset of the rollout pool (e.g. MAPF-GPT
failures with segment_diff in {1, 2, 3}) without physically moving files.
The output is a text file, one path per line, consumable by `query.py --manifest`.
"""

import argparse
from pathlib import Path

import numpy as np


def has_allowed_segment_diff(npz_path, allowed_diffs):
    data = np.load(npz_path, allow_pickle=True)
    if "segment_diffs" not in data:
        raise KeyError("Missing 'segment_diffs'")
    segment_diffs = data["segment_diffs"]
    return any(int(x) in allowed_diffs for x in segment_diffs), segment_diffs


def write_manifest(input_dir, output_path, allowed_diffs={1, 2, 3}, recursive=False):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(
        input_dir.rglob("*.npz") if recursive else input_dir.glob("*.npz")
    )
    print(f"Found {len(npz_files)} .npz files under {input_dir}")

    matched = []
    failed = 0
    for npz_file in npz_files:
        try:
            keep, _ = has_allowed_segment_diff(npz_file, allowed_diffs)
            if keep:
                matched.append(npz_file)
        except Exception as e:
            failed += 1
            print(f"Failed: {npz_file} | {e}")

    lines = [
        f"# Manifest written by filter_npzs_by_segment_diff.py",
        f"# Source: {input_dir} (recursive={recursive})",
        f"# Filter: segment_diffs ∩ {sorted(allowed_diffs)} non-empty",
        f"# Matched: {len(matched)} / {len(npz_files)} files",
        "",
    ]
    lines.extend(str(p) for p in matched)
    output_path.write_text("\n".join(lines) + "\n")

    print(f"Matched: {len(matched)}, Skipped: {len(npz_files) - len(matched) - failed}, Failed: {failed}")
    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write a manifest of .npz files whose segment_diffs contain selected values."
    )
    parser.add_argument("input_dir", help="Folder containing .npz files")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the manifest (text file, one .npz path per line).",
    )
    parser.add_argument(
        "--allowed-diffs",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Match files where segment_diffs contains at least one of these values.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Search recursively inside input_dir.",
    )
    args = parser.parse_args()

    write_manifest(
        input_dir=args.input_dir,
        output_path=args.output,
        allowed_diffs=set(args.allowed_diffs),
        recursive=args.recursive,
    )
