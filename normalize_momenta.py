#!/usr/bin/env python
"""Normalize the parton momenta in a (merged) dataset directory by the CM energy.

For every ``splittings_k.npy`` (and ``momenta_N*.npy`` / ``splittings_all.npy``
when present) in DATASET_DIR, writes a sibling file with the same name plus a
``_norm`` suffix holding p / E_cm, so that each row's four-momenta sum to
(1, 0, 0, 0) and every energy is the energy fraction z_i = E_i / E_cm.

    splittings_3.npy   ->  splittings_3_norm.npy
    momenta_N5.npy     ->  momenta_N5_norm.npy
    splittings_all.npy ->  splittings_all_norm.npy   (zero padding stays zero)

E_cm is read from ``ecm_GeV`` in metadata.json (override with --ecm).  The
normalized arrays are checked to sum to (1, 0, 0, 0) per row and the check
result is written back into metadata.json under ``normalized_files``.
Existing ``*_norm.npy`` files are only replaced with --overwrite.

Usage:
    python normalize_momenta.py DATASET_DIR [--ecm ECM] [--overwrite]
                                            [--tolerance TOL]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

SOURCE_RE = re.compile(r"^(splittings_(\d+|all)|momenta_N\d+)\.npy$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="Merged dataset directory.")
    ap.add_argument("--ecm", type=float, default=None,
                    help="CM energy in GeV (default: ecm_GeV from metadata.json).")
    ap.add_argument("--tolerance", type=float, default=1e-5,
                    help="Per-component tolerance on the normalized total "
                         "4-momentum (default 1e-5; float32 data).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing *_norm.npy files.")
    args = ap.parse_args()

    d = args.dataset_dir
    if not os.path.isdir(d):
        raise SystemExit(f"Not a directory: {d!r}")

    meta_path = os.path.join(d, "metadata.json")
    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    ecm = args.ecm
    if ecm is None:
        if metadata is None or "ecm_GeV" not in metadata:
            raise SystemExit("No --ecm given and no ecm_GeV in metadata.json.")
        ecm = float(metadata["ecm_GeV"])
    if ecm <= 0:
        raise SystemExit(f"E_cm must be positive, got {ecm}")
    print(f"Normalizing momenta in {d}/ by E_cm = {ecm} GeV")

    sources = sorted(
        f for f in os.listdir(d)
        if SOURCE_RE.match(f) and not f.endswith("_norm.npy")
    )
    if not sources:
        raise SystemExit("No splittings_k.npy / momenta_N*.npy files found.")

    expected = np.array([1.0, 0.0, 0.0, 0.0])
    results: dict[str, dict] = {}
    n_bad_total = 0
    for name in sources:
        out_name = name[:-4] + "_norm.npy"
        out_path = os.path.join(d, out_name)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  {out_name} exists; skipping (use --overwrite to replace)")
            continue

        arr = np.load(os.path.join(d, name))
        if arr.ndim != 3 or arr.shape[-1] != 4:
            print(f"  {name}: unexpected shape {arr.shape}; skipping")
            continue
        # divide in float64, store as float32 like the source
        norm = (arr.astype(np.float64) / ecm).astype(np.float32)
        np.save(out_path, norm)

        total = norm.astype(np.float64).sum(axis=1)
        dev = np.abs(total - expected)
        max_dev = float(dev.max())
        n_bad = int(np.any(dev > args.tolerance, axis=1).sum())
        n_bad_total += n_bad
        flag = "OK" if n_bad == 0 else "FAIL"
        print(f"  {name:22s} -> {out_name:27s} shape={tuple(norm.shape)}  "
              f"max |sum p/Ecm - (1,0,0,0)| = {max_dev:.2e}  "
              f"({n_bad} rows outside tol) [{flag}]")
        results[out_name] = {
            "source_file": name,
            "shape": list(norm.shape),
            "max_conservation_deviation": max_dev,
            "rows_outside_tolerance": n_bad,
        }

    if metadata is not None and results:
        metadata.setdefault("normalized_files", {}).update(results)
        metadata["normalization"] = {
            "ecm_GeV": ecm,
            "tolerance": args.tolerance,
            "note": ("*_norm.npy files hold (E, px, py, pz) / E_cm; every row "
                     "sums to (1, 0, 0, 0) and E_i is the energy fraction z_i. "
                     "Zero padding (splittings_all) stays zero."),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print("  metadata.json updated")

    if n_bad_total:
        raise SystemExit(f"Normalization check FAILED for {n_bad_total} rows.")
    print("Done; all normalized rows sum to (1, 0, 0, 0) within tolerance.")


if __name__ == "__main__":
    main()
