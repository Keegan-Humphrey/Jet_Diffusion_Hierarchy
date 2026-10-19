#!/usr/bin/env python
"""Merge chunked pythia snapshot output into a single dataset directory.

The chunked generation writes one directory per chunk
(pythia_gluon_700_part1, ..._part2, ...), each containing splittings_k.npy
for every branching step k. Since events across chunks are independent, the
full dataset for step k is just the concatenation (along axis 0) of that
step's array from every chunk.

This also merges event_ids_k.npy when present, offsetting each chunk's event
ids so they stay unique across chunks (chunk c is shifted by c * chunk_span,
where chunk_span is one more than the max id seen so far). splittings_all.npy
and event_record.npz are NOT merged (not needed by the classification
notebook, and event_record requires re-indexing offsets); regenerate those
separately if you ever need them.

Usage:
    python merge_chunks.py OUTPUT_DIR [--parts GLOB] [--prefix PREFIX]

Examples:
    python merge_chunks.py pythia_gluon_700
    python merge_chunks.py pythia_gluon_700 --parts 'pythia_gluon_700_part*'
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

STEP_RE = re.compile(r"splittings_(\d+)\.npy$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir",
                    help="Directory to write the merged splittings_k.npy into.")
    ap.add_argument("--parts", default=None,
                    help="Glob for the chunk directories "
                         "(default: OUTPUT_DIR + '_part*').")
    ap.add_argument("--prefix", default=None,
                    help="Alternative prefix for chunk dirs if not OUTPUT_DIR_part*.")
    ap.add_argument("--delete-parts", action="store_true",
                    help="Delete the chunk directories after a VERIFIED merge "
                         "(merged row counts must equal the sum of the sources). "
                         "Off by default; nothing is deleted unless the merge "
                         "fully checks out.")
    args = ap.parse_args()

    parts_glob = args.parts or f"{args.prefix or args.output_dir}_part*"
    part_dirs = sorted(d for d in glob.glob(parts_glob) if os.path.isdir(d))
    if not part_dirs:
        raise SystemExit(f"No chunk directories matched: {parts_glob!r}")

    print(f"Merging {len(part_dirs)} chunk directories:")
    for d in part_dirs:
        print(f"    {d}")

    # step -> list of (part_dir, splittings_path) in chunk order
    by_step: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for d in part_dirs:
        for f in glob.glob(os.path.join(d, "splittings_*.npy")):
            m = STEP_RE.search(os.path.basename(f))
            if m:                      # skips splittings_all.npy
                by_step[int(m.group(1))].append((d, f))

    if not by_step:
        raise SystemExit("No splittings_k.npy files found in the chunk dirs.")

    os.makedirs(args.output_dir, exist_ok=True)

    expected_rows: dict[int, int] = {}   # step -> summed source rows, for verify
    total_rows = 0
    for step in sorted(by_step):
        entries = sorted(by_step[step])          # deterministic chunk order
        arrays, id_arrays, id_offset = [], [], 0
        have_ids = True
        for d, f in entries:
            arr = np.load(f)
            arrays.append(arr)
            ids_path = os.path.join(d, f"event_ids_{step}.npy")
            if have_ids and os.path.exists(ids_path):
                ids = np.load(ids_path).astype(np.int64)
                id_arrays.append(ids + id_offset)
                # keep ids unique across chunks
                id_offset += int(ids.max()) + 1 if ids.size else 0
            else:
                have_ids = False

        merged = np.concatenate(arrays, axis=0)
        np.save(os.path.join(args.output_dir, f"splittings_{step}.npy"), merged)
        expected_rows[step] = int(sum(a.shape[0] for a in arrays))
        total_rows += merged.shape[0]

        if have_ids and id_arrays:
            merged_ids = np.concatenate(id_arrays, axis=0)
            np.save(os.path.join(args.output_dir, f"event_ids_{step}.npy"),
                    merged_ids)

        print(f"  step {step:2d}: {merged.shape[0]:8d} rows  "
              f"shape={tuple(merged.shape)}"
              + ("" if have_ids else "   [event_ids skipped: missing in a chunk]"))

    print(f"\nMerged {total_rows} total rows into {args.output_dir}/")
    print("The classification notebook can now load "
          f"{args.output_dir}/splittings_(N-2).npy")

    if args.delete_parts:
        # Verify every merged file reads back with exactly the summed row count
        # before removing any source. If a single step fails, delete nothing.
        print("\nVerifying merge before deleting chunk directories...")
        ok = True
        for step, want in sorted(expected_rows.items()):
            path = os.path.join(args.output_dir, f"splittings_{step}.npy")
            try:
                got = np.load(path, mmap_mode="r").shape[0]
            except Exception as e:                       # noqa: BLE001
                print(f"  step {step:2d}: FAILED to read back ({e})")
                ok = False
                continue
            mark = "ok" if got == want else "MISMATCH"
            if got != want:
                ok = False
            print(f"  step {step:2d}: {got} rows vs {want} expected  [{mark}]")

        if not ok:
            print("\nVerification FAILED -- keeping all chunk directories intact. "
                  "Nothing was deleted.")
            raise SystemExit(1)

        import shutil
        print("\nVerification passed. Deleting chunk directories:")
        for d in part_dirs:
            shutil.rmtree(d)
            print(f"    removed {d}")
        print("Done.")


if __name__ == "__main__":
    main()
