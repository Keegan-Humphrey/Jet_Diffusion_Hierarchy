"""e+e- -> gg / qqbar at a common CM energy, snapshotted at N final-state partons.

Both samples are produced through e+e- -> H with the Higgs pole mass set equal
to the CM energy (``25:m0 = eCM``, default 2000 GeV), so the colourless H sits
on resonance and is produced at rest: the total 4-momentum is (eCM, 0, 0, 0)
and the conservation check in ``run_generation`` applies unchanged.  The H is
then forced to decay to the requested hard partons, which shower via ordinary
FSR (hadronization off, QED showers off, ISR/MPI off).

    --process gg : force H -> g g.
    --process qq : force H -> q qbar with q in {u, d, s} (light quarks, mixed;
                   the mix follows the Yukawa-weighted branching ratios, so
                   s sbar dominates -- see ``--quark`` to pin one flavour).

The strong coupling is set via ``TimeShower:alphaSvalue`` (default 0.13).  By
default this is alpha_s(mZ) with the usual running (``TimeShower:alphaSorder``
= 1); pass ``--fixed-alphas`` to freeze alpha_s = 0.13 at every scale.

Snapshots: the pT-ordered FSR shower is walked with ``extract_fsr_snapshots``,
which records the active partons after every branching.  Only the snapshot at
exactly ``--n-final`` partons (5 or 10, i.e. k = N - 2 branchings after the two
seed partons) is kept, so every event that showers to at least N partons
contributes one (N, 4) array of (E, px, py, pz).  Events that stop short of N
partons are dropped.  Output (``splittings_{N-2}.npy``, ``event_ids_{N-2}.npy``,
``metadata.json``, optional ``event_record.npz``) uses the standard layout.

Usage:
    python generate_pythia_ee_gg_qq.py --process gg --n-final 5  [-n N] [--seed S]
    python generate_pythia_ee_gg_qq.py --process qq --n-final 10 [-n N] [--seed S]
"""

from __future__ import annotations

import json
import socket
from datetime import datetime

import pythia8

from pythia_snapshot_common import (
    COMMON_EE_SETTINGS,
    build_base_parser,
    extract_fsr_snapshots,
    find_hard_outgoing,
    resolve_seed,
    run_generation,
)

N_SEED = 2  # two hard partons (g g or q qbar) seed the shower

QUARK_CHOICES = {
    "uds": (1, 2, 3),
    "u": (2,),
    "d": (1,),
    "s": (3,),
}


def configure_pythia(seed: int, ecm: float, process: str, alphas: float,
                     fixed_alphas: bool, quarks: tuple[int, ...]):
    p = pythia8.Pythia("", False)
    settings = [
        "Random:setSeed = on",
        f"Random:seed = {seed}",
        "Beams:idA = 11",
        "Beams:idB = -11",
        f"Beams:eCM = {ecm}",
        # s-channel e+e- -> H (electron Yukawa); put H on resonance at eCM.
        "HiggsSM:ffbar2H = on",
        f"25:m0 = {ecm}",
        "25:onMode = off",
        "HadronLevel:all = off",
        # Strong coupling in the final-state shower.
        f"TimeShower:alphaSvalue = {alphas}",
        f"TimeShower:alphaSorder = {0 if fixed_alphas else 1}",
    ]
    if process == "gg":
        settings.append("25:onIfMatch = 21 21")               # H -> g g
    elif process == "qq":
        ids = " ".join(str(q) for q in quarks)
        settings.append(f"25:onIfAny = {ids}")                # H -> q qbar
    else:
        raise SystemExit(f"unknown process {process!r}")
    settings.extend(COMMON_EE_SETTINGS)
    for s in settings:
        p.readString(s)
    if not p.init():
        raise RuntimeError("Pythia.init() failed; check the configuration above.")
    return p


def make_extractor(n_final: int):
    """Return extract(event) -> [(k, snap)] keeping only the snapshot with
    exactly ``n_final`` active partons (k = n_final - N_SEED)."""
    k_target = n_final - N_SEED

    def extract(event):
        seeds = find_hard_outgoing(event, 25)
        if len(seeds) != N_SEED:
            return []
        snaps = extract_fsr_snapshots(event, seeds)
        return [(k, snap) for k, snap in snaps if k == k_target]

    return extract


def main() -> None:
    parser = build_base_parser(
        description=__doc__,
        default_output_dir="pythia_ee_gg_qq",
        default_ecm=2000.0,
    )
    parser.add_argument("--process", choices=("gg", "qq"), required=True,
                        help="Hard final state: gg (H->gg) or qq (H->qqbar).")
    parser.add_argument("--n-final", type=int, required=True,
                        help="Number of final-state partons at which to take "
                             "the snapshot (5 or 10).")
    parser.add_argument("--alphas", type=float, default=0.13,
                        help="TimeShower:alphaSvalue (default 0.13).")
    parser.add_argument("--fixed-alphas", action="store_true",
                        help="Freeze alpha_s at --alphas (TimeShower:alphaSorder "
                             "= 0) instead of running it from mZ.")
    parser.add_argument("--quark", choices=tuple(QUARK_CHOICES), default="uds",
                        help="Quark flavours allowed in H -> q qbar "
                             "(--process qq only). Default: u, d, s mixed.")
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    quarks = QUARK_CHOICES[args.quark]
    pythia = configure_pythia(args.seed, args.ecm, args.process, args.alphas,
                              args.fixed_alphas, quarks)

    if args.process == "gg":
        process = "e+e- -> H -> gg (H mass = eCM)"
    else:
        process = f"e+e- -> H -> qqbar, q in {args.quark} (H mass = eCM)"

    extra_metadata = {
        "process": process,
        "hard_final_state": args.process,
        "quark_flavours": list(quarks) if args.process == "qq" else None,
        "higgs_m0_GeV": args.ecm,
        "alphas_value": args.alphas,
        "alphas_order": 0 if args.fixed_alphas else 1,
        "n_final_partons": args.n_final,
        "snapshot_step_k": args.n_final - N_SEED,
        "note": (f"only the snapshot with exactly {args.n_final} partons "
                 f"(k = {args.n_final - N_SEED}) is kept per event; events that "
                 f"shower to fewer partons are dropped."),
    }

    t_start = datetime.now().astimezone()
    extra_metadata["generation_started"] = t_start.isoformat(timespec="seconds")
    extra_metadata["hostname"] = socket.gethostname()

    run_generation(
        pythia, make_extractor(args.n_final),
        n_events=args.n_events, ecm=args.ecm, seed=args.seed,
        tolerance=args.tolerance, output_dir=args.output_dir,
        save_event_record=args.save_event_record, extra_metadata=extra_metadata,
        n_seed=N_SEED, save_all_events=False,
    )

    # run_generation writes metadata.json itself; append the finishing time.
    t_end = datetime.now().astimezone()
    meta_path = args.output_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["generation_finished"] = t_end.isoformat(timespec="seconds")
    meta["wall_time_s"] = round((t_end - t_start).total_seconds(), 1)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wall time: {meta['wall_time_s']} s  (seed {args.seed})")


if __name__ == "__main__":
    main()
