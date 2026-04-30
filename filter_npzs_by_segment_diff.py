import argparse
import shutil
from pathlib import Path

import numpy as np


def has_allowed_segment_diff(npz_path, allowed_diffs):
    data = np.load(npz_path, allow_pickle=True)

    if "segment_diffs" not in data:
        raise KeyError("Missing 'segment_diffs'")

    segment_diffs = data["segment_diffs"]

    return any(int(x) in allowed_diffs for x in segment_diffs), segment_diffs


def move_matching_npzs(input_dir, output_dir, allowed_diffs={1, 2, 3}, recursive=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(
        input_dir.rglob("*.npz") if recursive else input_dir.glob("*.npz")
    )

    moved = 0
    skipped = 0
    failed = 0

    print(f"Found {len(npz_files)} .npz files")

    for npz_file in npz_files:
        try:
            keep, segment_diffs = has_allowed_segment_diff(npz_file, allowed_diffs)

            if keep:
                output_path = output_dir / npz_file.name

                # Avoid overwriting if duplicate names exist
                if output_path.exists():
                    output_path = output_dir / f"{npz_file.stem}_copy{npz_file.suffix}"

                shutil.move(str(npz_file), str(output_path))

                moved += 1
                print(f"Moved: {npz_file.name} | segment_diffs={segment_diffs.tolist()}")
            else:
                skipped += 1

        except Exception as e:
            failed += 1
            print(f"Failed: {npz_file} | {e}")

    print(f"\nDone. Moved: {moved}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move .npz files whose segment_diffs contain selected values."
    )

    parser.add_argument(
        "input_dir",
        help="Folder containing .npz files",
    )

    parser.add_argument(
        "output_dir",
        help="Folder to move matching .npz files into",
    )

    parser.add_argument(
        "--allowed-diffs",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Move files where segment_diffs contains at least one of these values.",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search recursively inside input_dir.",
    )

    args = parser.parse_args()

    move_matching_npzs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        allowed_diffs=set(args.allowed_diffs),
        recursive=args.recursive,
    )