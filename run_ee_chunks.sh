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
# When all chunks are done the script automatically runs
#     python merge_chunks.py <BASE>          (merge + verify + delete chunks)
#     python normalize_momenta.py <BASE>     (write *_norm.npy = p / E_cm)
# so the finished dataset is <BASE>/momenta_N{N}.npy and momenta_N{N}_norm.npy.
# Pass --no-merge to stop after generation and do those steps by hand.
#
# Usage:
#   bash run_ee_chunks.sh <gg|qq> <N>=3> [--fixed-alphas] [--no-g2qq] [--force] [--no-merge] [extra args...]
#
# Examples:
#   bash run_ee_chunks.sh gg 5                    # running alpha_s(mZ)=0.13
#   bash run_ee_chunks.sh qq 10 --fixed-alphas    # alpha_s frozen at 0.13
#   bash run_ee_chunks.sh gg 10 --quark u         # extra args go to the generator
#   bash run_ee_chunks.sh gg 30 --fixed-alphas --no-g2qq   # pure glue, fixed alpha_s
#
# --no-g2qq switches off g -> q qbar in the shower; the output directory gets
# a "_noG2qq" suffix so it never mixes with the default datasets.  For a series
# of multiplicities in one go, use run_ee_series.sh.
#
# Environment overrides:
#   CHUNK_SIZE (default 100000), N_CHUNKS (default 10), SEED0 (default: see
#   run-index logic below), PYTHON (default python), ECM (default 2000),
#   ALPHAS (default 0.13)
#
# Repeated runs of the same configuration go to CONFIG, CONFIG_run2,
# CONFIG_run3, ... with disjoint seed ranges, so nothing is overwritten and
# no events are duplicated.
#
# Watch memory in a second shell:  watch -n 2 free -h

set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

PROCESS="$1"; shift
N_FINAL="$1"; shift

case "${PROCESS}" in gg|qq) ;; *) echo "process must be gg or qq, got '${PROCESS}'" >&2; exit 1 ;; esac
#case "${N_FINAL}" in 5|10) ;; *) echo "n-final must be 5 or 10, got '${N_FINAL}'" >&2; exit 1 ;; esac
if ! [[ "${N_FINAL}" =~ ^[0-9]+$ ]] || [[ ${N_FINAL} -lt 3 ]]; then
    echo "n-final must be an integer >= 3, got '${N_FINAL}'" >&2; exit 1
fi

# --- configuration (override via environment) --------------------------------
PYTHON="${PYTHON:-python}"          # env with pythia8 bindings
SCRIPT="${SCRIPT:-generate_pythia_ee_gg_qq.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-merge_chunks.py}"
NORMALIZE_SCRIPT="${NORMALIZE_SCRIPT:-normalize_momenta.py}"
ECM="${ECM:-2000}"
ALPHAS="${ALPHAS:-0.13}"
CHUNK_SIZE="${CHUNK_SIZE:-100000}"   # events per chunk
N_CHUNKS="${N_CHUNKS:-10}"           # CHUNK_SIZE * N_CHUNKS = total events (default 1M)
# SEED0: chunk s uses seed SEED0 + s - 1.  Default: chosen per run index
#        below so that repeated runs of one configuration never reuse a seed.
RECORD_FLAG="${RECORD_FLAG:---no-event-record}"   # set RECORD_FLAG="" to save event records
# -----------------------------------------------------------------------------

ALPHA_TAG="run"
FIXED_FLAG=""
G2QQ_FLAG=""
G2QQ_TAG=""
FORCE=0
DO_MERGE=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fixed-alphas) FIXED_FLAG="--fixed-alphas"; ALPHA_TAG="fix"; shift ;;
        --no-g2qq)      G2QQ_FLAG="--no-g2qq"; G2QQ_TAG="_noG2qq"; shift ;;
        --force)        FORCE=1; shift ;;
        --no-merge)     DO_MERGE=0; shift ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

CONFIG="pythia_${PROCESS}_${ECM}_N${N_FINAL}_a${ALPHAS}${ALPHA_TAG}${G2QQ_TAG}"

# --- pick a run index so repeated runs of the same configuration never
# collide (in directory name OR in seeds).  Run r is named CONFIG (r=1) or
# CONFIG_run{r} (r>1).  A run whose merged directory already exists is taken
# as finished and we move on to the next index; a run with leftover _part
# directories but no merged directory is resumed (complete chunks are skipped).
# Unless SEED0 is given explicitly, run r uses seeds
# (r-1)*N_CHUNKS+1 ... r*N_CHUNKS, so no seed is ever reused across runs.
RUN=1
while :; do
    BASE="${CONFIG}"; [[ ${RUN} -gt 1 ]] && BASE="${CONFIG}_run${RUN}"
    if [[ -d "${BASE}" ]]; then
        RUN=$(( RUN + 1 )); continue        # merged dataset exists -> next run
    fi
    break
done
OUT_PREFIX="${BASE}_part"
if [[ -z "${SEED0:-}" ]]; then
    SEED0=$(( (RUN - 1) * N_CHUNKS + 1 ))
fi
if [[ ${RUN} -gt 1 ]]; then
    echo "NOTE: ${CONFIG} (and earlier runs) already exist; this is run ${RUN}"
    echo "      -> writing to ${BASE}, seeds ${SEED0}..$(( SEED0 + N_CHUNKS - 1 ))"
fi
if compgen -G "${OUT_PREFIX}*" > /dev/null; then
    echo "NOTE: leftover chunk directories found for ${BASE}; resuming that run"
    echo "      (completed chunks are skipped; use --force to redo them)"
fi

echo "Generating ${N_CHUNKS} chunks of ${CHUNK_SIZE} events"
echo "  process=${PROCESS}  ecm=${ECM}  n-final=${N_FINAL}  alpha_s=${ALPHAS} (${ALPHA_TAG})  g->qqbar=$([[ -n ${G2QQ_FLAG} ]] && echo off || echo on)"
echo "  extra generator args: ${EXTRA_ARGS[*]:-<none>}"
echo "  seeds: ${SEED0}..$(( SEED0 + N_CHUNKS - 1 ))"
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
        ${G2QQ_FLAG} \
        -n "${CHUNK_SIZE}" \
        --seed "${seed}" \
        --output-dir "${outdir}" \
        ${RECORD_FLAG} \
        "${EXTRA_ARGS[@]}"
    echo "=== chunk ${s} done ==="
    echo
done

if [[ ${DO_MERGE} -eq 0 ]]; then
    echo "All chunks generated (--no-merge). To finish by hand:"
    echo "    ${PYTHON} ${MERGE_SCRIPT} ${BASE}"
    echo "    ${PYTHON} ${NORMALIZE_SCRIPT} ${BASE}"
    exit 0
fi

echo "All chunks generated. Merging into ${BASE}/ ..."
"${PYTHON}" "${MERGE_SCRIPT}" "${BASE}"
echo
echo "Normalizing momenta by E_cm ..."
"${PYTHON}" "${NORMALIZE_SCRIPT}" "${BASE}"
echo
echo "Done. Dataset: ${BASE}/"
echo "    momenta_N${N_FINAL}.npy        (E, px, py, pz) in GeV"
echo "    momenta_N${N_FINAL}_norm.npy   divided by E_cm = ${ECM} GeV"
echo "    metadata.json"
