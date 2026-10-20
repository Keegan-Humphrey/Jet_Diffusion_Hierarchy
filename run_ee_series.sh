#!/usr/bin/env bash
#
# Run run_ee_chunks.sh for a series of multiplicities N, one after another,
# so a whole scan can be started once and left alone.
#
# Each N is a complete run_ee_chunks.sh call (generate -> merge -> normalize),
# logged to ${LOG_DIR}/<process>_N<N>.log.  If one N fails the series carries
# on with the next and reports the failures at the end.  Re-running the same
# series resumes any N whose chunks were left half-done (finished chunks are
# skipped), but an N whose merged dataset already exists is generated AGAIN as
# a new _run2, _run3, ... -- trim N_LIST if you only want the missing ones.
#
# Usage:
#   bash run_ee_series.sh                                   # use the block below
#   bash run_ee_series.sh gg 5 10 20 30                     # override process + N list
#   bash run_ee_series.sh gg 20 40 -- --fixed-alphas --no-g2qq --ptmin 0.5
#
# Anything after "--" replaces FLAGS below and is passed to run_ee_chunks.sh
# (its own flags --fixed-alphas, --no-g2qq, --force, --no-merge, plus any
# generator args such as --ptmin or --quark).  Environment overrides of
# run_ee_chunks.sh (CHUNK_SIZE, N_CHUNKS, PYTHON, ECM, ALPHAS, ...) apply to
# every N in the series.

set -uo pipefail     # no -e: one failing N must not stop the rest

# --- User-specific settings (edit here) -------------------------------------
PROCESS="gg"
N_LIST=(10 12 14 16 18 20)
FLAGS=(--fixed-alphas --no-g2qq)
CHUNKS_SCRIPT="${CHUNKS_SCRIPT:-run_ee_chunks.sh}"
LOG_DIR="${LOG_DIR:-logs_series_$(date +%Y%m%d_%H%M%S)}"
# -----------------------------------------------------------------------------

# --- command-line overrides ---------------------------------------------------
if [[ $# -gt 0 && "$1" != "--" ]]; then
    PROCESS="$1"; shift
    CLI_N=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do CLI_N+=("$1"); shift; done
    [[ ${#CLI_N[@]} -gt 0 ]] && N_LIST=("${CLI_N[@]}")
fi
if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
    FLAGS=("$@")
fi

case "${PROCESS}" in gg|qq) ;; *) echo "process must be gg or qq, got '${PROCESS}'" >&2; exit 1 ;; esac
for N in "${N_LIST[@]}"; do
    if ! [[ "${N}" =~ ^[0-9]+$ ]] || [[ ${N} -lt 3 ]]; then
        echo "each N must be an integer >= 3, got '${N}'" >&2; exit 1
    fi
done
if [[ ! -f "${CHUNKS_SCRIPT}" ]]; then
    echo "cannot find ${CHUNKS_SCRIPT} (run from the directory that holds it, or set CHUNKS_SCRIPT)" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
echo "Series: process=${PROCESS}  N = ${N_LIST[*]}"
echo "        flags: ${FLAGS[*]:-<none>}"
echo "        logs:  ${LOG_DIR}/"
echo

FAILED=()
T0=$(date +%s)
for N in "${N_LIST[@]}"; do
    log="${LOG_DIR}/${PROCESS}_N${N}.log"
    echo "################ N=${N}  started $(date '+%F %T')  (log: ${log})"
    t=$(date +%s)
    if bash "${CHUNKS_SCRIPT}" "${PROCESS}" "${N}" "${FLAGS[@]}" 2>&1 | tee "${log}"; then
        echo "################ N=${N}  OK  in $(( ($(date +%s) - t) / 60 )) min"
    else
        echo "################ N=${N}  FAILED (see ${log})"
        FAILED+=("${N}")
    fi
    echo
done

echo "Series finished in $(( ($(date +%s) - T0) / 60 )) min."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED: N = ${FAILED[*]}   (re-run those with: bash $0 ${PROCESS} ${FAILED[*]} -- ${FLAGS[*]})"
    exit 1
fi
echo "All N completed."
