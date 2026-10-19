#!/usr/bin/env bash
#
# Generate the e+e- -> gg / qqbar sample at 2 TeV, snapshotted at N partons,
# in independent memory-safe chunks with distinct seeds.
#
# Each chunk is a separate generate_pythia_ee_gg_qq.py run.  Events are
# independent, so N chunks with distinct seeds are numerically equivalent to
# one big run.  Because this generator keeps only ONE (N,4) snapshot per event,
# memory is far below the old full-shower runs, so the chunks here are mainly
# for restartability: a crash or kill loses only the current chunk, and
# already-finished chunks are skipped on re-run (unless --force).
#
# After this finishes, run:  python merge_chunks.py <OUT_PREFIX base>
#
# Usage:
#   bash run_ee_chunks.sh <gg|qq> <5|10> [--fixed-alphas] [--force] [extra args...]
#
# Examples:
#   bash run_ee_chunks.sh gg 5                    # running alpha_s(mZ)=0.13
#   bash run_ee_chunks.sh qq 10 --fixed-alphas    # alpha_s frozen at 0.13
#   bash run_ee_chunks.sh gg 10 --quark u         # extra args go to the generator
#
# Environment overrides:
#   CHUNK_SIZE (default 25000), N_CHUNKS (default 40), SEED0 (default 1),
#   PYTHON (default python), ECM (default 2000), ALPHAS (default 0.13)
#
# Watch memory in a second shell:  watch -n 2 free -h

set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

PROCESS="$1"; shift
N_FINAL="$1"; shift

case "${PROCESS}" in gg|qq) ;; *) echo "process must be gg or qq, got '${PROCESS}'" >&2; exit 1 ;; esac
case "${N_FINAL}" in 5|10) ;; *) echo "n-final must be 5 or 10, got '${N_FINAL}'" >&2; exit 1 ;; esac

# --- configuration (override via environment) --------------------------------
PYTHON="${PYTHON:-python}"          # env with pythia8 bindings
SCRIPT="${SCRIPT:-generate_pythia_ee_gg_qq.py}"
ECM="${ECM:-2000}"
ALPHAS="${ALPHAS:-0.13}"
CHUNK_SIZE="${CHUNK_SIZE:-25000}"   # events per chunk
N_CHUNKS="${N_CHUNKS:-40}"           # CHUNK_SIZE * N_CHUNKS = total events (default 1M)
SEED0="${SEED0:-1}"                  # chunk s uses seed SEED0 + s - 1
RECORD_FLAG="${RECORD_FLAG:---no-event-record}"   # set RECORD_FLAG="" to save event records
# -----------------------------------------------------------------------------

ALPHA_TAG="run"
FIXED_FLAG=""
FORCE=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fixed-alphas) FIXED_FLAG="--fixed-alphas"; ALPHA_TAG="fix"; shift ;;
        --force)        FORCE=1; shift ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

BASE="pythia_${PROCESS}_${ECM}_N${N_FINAL}_a${ALPHAS}${ALPHA_TAG}"
OUT_PREFIX="${BASE}_part"

echo "Generating ${N_CHUNKS} chunks of ${CHUNK_SIZE} events"
echo "  process=${PROCESS}  ecm=${ECM}  n-final=${N_FINAL}  alpha_s=${ALPHAS} (${ALPHA_TAG})"
echo "  extra generator args: ${EXTRA_ARGS[*]:-<none>}"
echo "  total: $(( CHUNK_SIZE * N_CHUNKS )) events  ->  ${OUT_PREFIX}{1..${N_CHUNKS}}"
echo

for s in $(seq 1 "${N_CHUNKS}"); do
    outdir="${OUT_PREFIX}${s}"
    seed=$(( SEED0 + s - 1 ))
    if [[ ${FORCE} -eq 0 && -f "${outdir}/metadata.json" ]]; then
        echo "=== chunk ${s}/${N_CHUNKS} already complete (${outdir}/metadata.json exists); skipping ==="
        continue
    fi
    echo "=== chunk ${s}/${N_CHUNKS}  seed=${seed}  ->  ${outdir} ==="
    rm -rf "${outdir}"        # clear any partial output from a previous attempt
    "${PYTHON}" "${SCRIPT}" \
        --process "${PROCESS}" \
        --n-final "${N_FINAL}" \
        --ecm "${ECM}" \
        --alphas "${ALPHAS}" \
        ${FIXED_FLAG} \
        -n "${CHUNK_SIZE}" \
        --seed "${seed}" \
        --output-dir "${outdir}" \
        ${RECORD_FLAG} \
        "${EXTRA_ARGS[@]}"
    echo "=== chunk ${s} done ==="
    echo
done

echo "All chunks generated. Now merge them:"
echo "    ${PYTHON} merge_chunks.py ${BASE}"
