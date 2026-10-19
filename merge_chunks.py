#!/usr/bin/env python
"""Merge chunked pythia snapshot output into a single dataset directory.

The chunked generation writes one directory per chunk
(<base>_part1, <base>_part2, ...), each containing splittings_k.npy for one or
more snapshot steps k (shape (n_k, n_seed + k, 4) of (E, px, py, pz) per
parton).  Since events across chunks are independent, the full dataset for
step k is just the concatenation (along axis 0) of that step's array from
every chunk.

This works for both dataset flavours:

  * per-branching datasets (generate_pythia_flavor.py, ...): many steps k;
  * fixed-multiplicity momenta datasets (generate_pythia_ee_gg_qq.py): a
    single step k = N - 2 holding the N-parton final-state momenta.  For these
    the merged file is additionally written as momenta_N{N}.npy (identical
    contents) for convenience.

Also merged:
  * event_ids_k.npy, offset per chunk so ids stay unique across chunks;
  * metadata.json: the physics configuration (process, ecm, alpha_s, N, ...)
    is checked to be IDENTICAL across chunks (the merge refuses to mix
    e.g. running- and fixed-alpha_s chunks), the event counts are summed, and
    the per-chunk seeds are recorded.

splittings_all.npy and event_record.npz are NOT merged (event_record requires
re-indexing offsets); regenerate those separately if you ever need them.

After merging, the merged files are read back and their row counts checked
against the sum over chunks; only if every step verifies are the chunk
directories deleted (pass --keep-parts to keep them).  A failed verification
deletes nothing and exits nonzero.

Usage:
    python merge_chunks.py OUTPUT_DIR [--parts GLOB] [--prefix PREFIX]
                           [--keep-parts] [--allow-mixed] [--overwrite]

Examples:
    python merge_chunks.py pythia_gluon_700
    python merge_chunks.py pythia_gg_2000_N5_a0.13fix
    python merge_chunks.py merged_gg --parts 'pythia_gg_2000_N5_a0.13run_part*'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np

STEP_RE = re.compile(r"splittings_(\d+)\.npy$")

# metadata keys that must agree across chunks (physics configuration)
CONFIG_KEYS = (
    "ecm_GeV", "n_seed", "tolerance_GeV", "expected_total_4momentum",
    "process", "flavor", "hard_final_state", "quark_flavours",
    "higgs_m0_GeV", "alphas_value", "alphas_order",
    "n_final_partons", "snapshot_step_k", "item",
)
# metadata keys summed across chunks
COUNT_KEYS = ("n_events_requested", "n_events_generated", "n_events_skipped")


def load_metadata(part_dir: str) -> dict | None:
    path = os.path.join(part_dir, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def merge_metadata(part_dirs: list[str], allow_mixed: bool) -> dict | None:
    """Combine the chunks' metadata.json.  Returns None if no chunk has one."""
    metas = [(d, load_metadata(d)) for d in part_dirs]
    metas = [(d, m) for d, m in metas if m is not None]
    if not metas:
        print("  (no metadata.json in any chunk; skipping metadata merge)")
        return None
    if len(metas) != len(part_dirs):
        missing = sorted(set(part_dirs) - {d for d, _ in metas})
        print(f"  WARNING: metadata.json missing in {len(missing)} chunk(s): "
              f"{', '.join(missing)}")

    ref_dir, ref = metas[0]
    config = {k: ref[k] for k in CONFIG_KEYS if k in ref}
    mismatches: list[str] = []
    for d, m in metas[1:]:
        for k in CONFIG_KEYS:
            if m.get(k) != ref.get(k):
                mismatches.append(f"{k}: {ref_dir}={ref.get(k)!r} vs {d}={m.get(k)!r}")
    if mismatches:
        print("  Chunk configuration MISMATCH:")
        for line in mismatches:
            print(f"    {line}")
        if not allow_mixed:
            raise SystemExit("Refusing to merge chunks with different "
                             "configurations (pass --allow-mixed to override).")
        print("  --allow-mixed given: continuing anyway.")

    merged = dict(config)
    for k in COUNT_KEYS:
        merged[k] = int(sum(m.get(k, 0) for _, m in metas))
    merged["n_chunks"] = len(metas)
    merged["chunks"] = [
        {"dir": d, "seed": m.get("seed"),
         "n_events_generated": m.get("n_events_generated"),
         "generation_started": m.get("generation_started"),
         "generation_finished": m.get("generation_finished"),
         "wall_time_s": m.get("wall_time_s"),
         "hostname": m.get("hostname")}
        for d, m in metas
    ]
    merged["seeds"] = [m.get("seed") for _, m in metas]
    if len(set(merged["seeds"])) != len(merged["seeds"]):
        print("  WARNING: duplicate seeds across chunks -- duplicated events!")
    starts = [m["generation_started"] for _, m in metas if m.get("generation_started")]
    ends = [m["generation_finished"] for _, m in metas if m.get("generation_finished")]
    walls = [m["wall_time_s"] for _, m in metas if m.get("wall_time_s") is not None]
    if starts:
        merged["generation_started"] = min(starts)
    if ends:
        merged["generation_finished"] = max(ends)
    if walls:
        merged["total_wall_time_s"] = round(float(sum(walls)), 1)
    merged["merged_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    # carry over any free-text note from the generator
    if "note" in ref:
        merged["generator_note"] = ref["note"]
    return merged


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
    ap.add_argument("--allow-mixed", action="store_true",
                    help="Merge even if the chunks' metadata.json disagree on "
                         "the physics configuration (process, alpha_s, N, ...).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing, non-empty OUTPUT_DIR. By default "
                         "the merge refuses to overwrite an existing dataset.")
    ap.add_argument("--keep-parts", dest="delete_parts", action="store_false",
                    default=True,
                    help="Keep the chunk directories after merging. By default "
                         "they are deleted, but only after a VERIFIED merge "
                         "(merged row counts must equal the sum of the sources); "
                         "nothing is deleted unless the merge fully checks out.")
    args = ap.parse_args()

    parts_glob = args.parts or f"{args.prefix or args.output_dir}_part*"
    part_dirs = sorted(d for d in glob.glob(parts_glob) if os.path.isdir(d))
    if not part_dirs:
        raise SystemExit(f"No chunk directories matched: {parts_glob!r}")

    print(f"Merging {len(part_dirs)} chunk directories:")
    for d in part_dirs:
        print(f"    {d}")

    print("\nChecking chunk metadata...")
    metadata = merge_metadata(part_dirs, args.allow_mixed)

    # step -> list of (part_dir, splittings_path) in chunk order
    by_step: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for d in part_dirs:
        for f in glob.glob(os.path.join(d, "splittings_*.npy")):
            m = STEP_RE.search(os.path.basename(f))
            if m:                      # skips splittings_all.npy
                by_step[int(m.group(1))].append((d, f))

    if not by_step:
        raise SystemExit("No splittings_k.npy files found in the chunk dirs.")

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory {args.output_dir!r} already exists and is not "
                f"empty -- refusing to overwrite an existing dataset. Pass "
                f"--overwrite to replace it, or merge into a different "
                f"OUTPUT_DIR (e.g. '{args.output_dir}_run2' with "
                f"--parts '{parts_glob}')."
            )
        print(f"  --overwrite given: replacing contents of {args.output_dir}/")
    os.makedirs(args.output_dir, exist_ok=True)

    n_seed = int(metadata.get("n_seed", 2)) if metadata else 2
    expected_rows: dict[int, int] = {}   # step -> summed source rows, for verify
    total_rows = 0
    step_info: dict[str, dict] = {}
    print()
    for step in sorted(by_step):
        entries = sorted(by_step[step])          # deterministic chunk order
        if len(entries) != len(part_dirs):
            have = {d for d, _ in entries}
            print(f"  WARNING: step {step} missing in "
                  f"{', '.join(sorted(set(part_dirs) - have))}")
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
        out_path = os.path.join(args.output_dir, f"splittings_{step}.npy")
        np.save(out_path, merged)
        expected_rows[step] = int(sum(a.shape[0] for a in arrays))
        total_rows += merged.shape[0]

        ids_name = None
        if have_ids and id_arrays:
            merged_ids = np.concatenate(id_arrays, axis=0)
            ids_name = f"event_ids_{step}.npy"
            np.save(os.path.join(args.output_dir, ids_name), merged_ids)

        step_info[str(step)] = {
            "snapshot_file": os.path.basename(out_path),
            "event_ids_file": ids_name,
            "shape": list(merged.shape),
        }
        print(f"  step {step:2d}: {merged.shape[0]:8d} rows  "
              f"shape={tuple(merged.shape)}"
              + ("" if have_ids else "   [event_ids skipped: missing in a chunk]"))

    # Fixed-multiplicity momenta dataset: a single step -> also write momenta_N.
    momenta_name = None
    if len(by_step) == 1:
        (step,) = by_step
        n_final = n_seed + step
        momenta_name = f"momenta_N{n_final}.npy"
        shutil.copyfile(os.path.join(args.output_dir, f"splittings_{step}.npy"),
                        os.path.join(args.output_dir, momenta_name))
        print(f"  single-step dataset: also written as {momenta_name} "
              f"(N = {n_final} partons x (E, px, py, pz))")

    if metadata is not None:
        metadata["snapshot_files"] = step_info
        if momenta_name:
            metadata["momenta_file"] = momenta_name
        metadata["merged_from_glob"] = parts_glob
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  metadata.json written ({metadata['n_events_generated']} events "
              f"generated over {metadata['n_chunks']} chunks)")

    print(f"\nMerged {total_rows} total rows into {args.output_dir}/")
    if momenta_name:
        print(f"Load the momenta with np.load('{args.output_dir}/{momenta_name}')")
    else:
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

        print("\nVerification passed. Deleting chunk directories:")
        for d in part_dirs:
            shutil.rmtree(d)
            print(f"    removed {d}")
        print("Done.")
    else:
        print("\n--keep-parts given: chunk directories left in place.")


if __name__ == "__main__":
    main()
