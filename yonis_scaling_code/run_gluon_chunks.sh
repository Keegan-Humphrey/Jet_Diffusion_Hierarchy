#!/usr/bin/env bash
#
# Generate the H -> gg gluon sample at 700 GeV in memory-safe chunks.
#
# The generator accumulates all snapshots in RAM and only writes at the end,
# so a single -n 100000 run overruns an 8 GB machine (~9.5 GB projected).
# Splitting into independent chunks caps peak memory at roughly one chunk's
# worth. Events are independent, so N chunks with distinct seeds is
# numerically equivalent to one big run.
#
# After this finishes, run:  python merge_chunks.py pythia_gluon_700
#
# Usage:  bash run_gluon_chunks.sh
# Watch memory in a second shell:  watch -n 2 free -h

set -euo pipefail

# --- configuration -----------------------------------------------------------
PYTHON="python"   # env with pythia8 bindings
SCRIPT="generate_pythia_flavor.py"
ECM=700
CHUNK_SIZE=25000        # events per chunk; lower to 20000 (5 chunks) if tight
N_CHUNKS=4              # CHUNK_SIZE * N_CHUNKS = total events (default 100k)
OUT_PREFIX="pythia_gluon_700_part"
RECORD_FLAG="--no-event-record"   # keep OFF for the classification data.
# To also save the per-particle event record, use SMALLER chunks (e.g.
# CHUNK_SIZE=10000, N_CHUNKS=10) and set:  RECORD_FLAG=""
# -----------------------------------------------------------------------------

echo "Generating ${N_CHUNKS} chunks of ${CHUNK_SIZE} events (H->gg, ecm=${ECM})"
echo "Total: $(( CHUNK_SIZE * N_CHUNKS )) events"
echo

for s in $(seq 1 "${N_CHUNKS}"); do
    outdir="${OUT_PREFIX}${s}"
    echo "=== chunk ${s}/${N_CHUNKS}  seed=${s}  ->  ${outdir} ==="
    rm -rf "${outdir}"        # clear any partial output from a previous attempt
    "${PYTHON}" "${SCRIPT}" \
        --flavor gluon \
        --ecm "${ECM}" \
        -n "${CHUNK_SIZE}" \
        --seed "${s}" \
        --output-dir "${outdir}" \
        ${RECORD_FLAG}
    echo "=== chunk ${s} done ==="
    echo
done

echo "All chunks generated. Now merge them:"
echo "    ${PYTHON} merge_chunks.py pythia_gluon_700"
