#!/usr/bin/env bash
# Usage:
#   ./ptnt-launch.sh -c configs/figure3.yaml            # default: CPU single-thread, no overrides
#   ./ptnt-launch.sh -c configs/figure3.yaml -d cpu -t 16
#   ./ptnt-launch.sh -c configs/figure3.yaml -d cuda
#   ./ptnt-launch.sh -c configs/figure3.yaml -d cuda -t 8 -q 4 -s 5 -O auto-hq
#
# Flags:
#   -c <config.yaml>     (required) base YAML config
#   -d <device>          cpu | cuda        (default: cpu)
#   -t <threads>         integer threads for BLAS/OpenMP on host (default: 1 on CPU; 4 on CUDA)
#   -q <n_qubits>        optional override of pt.n_qubits
#   -s <n_steps>         optional override of pt.n_steps
#   -O <opt>             tensor contraction path preset (default: auto-hq)
#   -o <outdir>          optional override output dir (else YAML/default)
#   -m <mode>            training.mode override if you want (normal | X_decomp). If omitted, YAML wins.

set -euo pipefail

CONFIG=""
DEVICE="cpu"
THREADS=""
NQ=""
NSTEPS=""
OPT="auto-hq"
OUTDIR=""
MODE=""

while getopts "c:d:t:q:s:O:o:m:" opt; do
  case $opt in
    c) CONFIG="$OPTARG" ;;
    d) DEVICE="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    q) NQ="$OPTARG" ;;
    s) NSTEPS="$OPTARG" ;;
    O) OPT="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    m) MODE="$OPTARG" ;;
    *) echo "Invalid option"; exit 2 ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Error: -c <config.yaml> is required" >&2
  exit 2
fi

# ------------------------
# 1) Environment selection
# ------------------------
# Host threading defaults
if [[ -z "${THREADS}" ]]; then
  if [[ "${DEVICE}" == "cuda" ]]; then
    THREADS=4       # moderate host threads when GPU is used (planning, I/O, etc.)
  else
    THREADS=1       # deterministic-ish CPU default; raise for speed (e.g. 8, 16)
  fi
fi

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"

# JAX device choice + numerics
if [[ "${DEVICE}" == "cuda" ]]; then
  export JAX_PLATFORMS=cuda
  # Precision that balances speed & stability on GPU; 'higher' is invalid—use 'high' or 'highest'
  export JAX_DEFAULT_MATMUL_PRECISION=high
  # GPU memory behavior (optional but user-friendly):
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
else
  export JAX_PLATFORMS=cpu
  export JAX_DEFAULT_MATMUL_PRECISION=high
  # To *force* single-threaded Eigen kernels on CPU, uncomment:
  # export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
fi

# Use float64 in JAX for better numerical stability in optimization (can turn off if you prefer speed)
export JAX_ENABLE_X64=True

# --------------------------------------
# 2) Make a temp YAML with CLI overrides
# --------------------------------------
TMPCFG="$(mktemp /tmp/ptnt_cfg_XXXX.yaml)"

python - "$CONFIG" "$TMPCFG" "$NQ" "$NSTEPS" "$OUTDIR" "$MODE" <<'PY'
import sys, yaml
src, dst, nq, ns, outdir, mode = sys.argv[1:]

with open(src, "r") as fh:
    cfg = yaml.safe_load(fh)

# Safely walk and override if flags were provided
pt = cfg.setdefault("pt", {})
if nq:
    pt["n_qubits"] = int(nq)
if ns:
    pt["n_steps"] = int(ns)

if outdir:
    cfg.setdefault("output", {})["dir"] = outdir

if mode:
    cfg.setdefault("training", {})["mode"] = mode

with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(dst)
PY

# Python prints the TMPCFG path on stdout; capture it
TMPCFG_PATH="$(tail -n1 <<< "$TMPCFG")" || true
# Some shells may not capture as expected; fallback:
if [[ ! -f "$TMPCFG_PATH" ]]; then TMPCFG_PATH="$TMPCFG"; fi

# ------------------------
# 3) Run your CLI
# ------------------------
# Your CLI already accepts --device, --num-threads, --opt (as you showed in your logs).
# If your installed CLI lacks these flags, remove them here and rely solely on env vars.

CMD=(ptnt run "${TMPCFG_PATH}" --device "${DEVICE}" --num-threads "${THREADS}" --opt "${OPT}")
echo "+ ${CMD[@]}"
exec "${CMD[@]}"

