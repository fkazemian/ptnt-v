#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./ptnt-launch.sh -c <cfg.yaml> [-d cpu|cuda] [-t N] [--fp64] [--deterministic] \
                   [--opt <auto|auto-hq|greedy|...>] [-q <n_qubits>] [-s <n_steps>]

Flags:
  -c, --config         Path to YAML config (required).
  -d, --device         'cpu' (default) or 'cuda'.
  -t, --threads        Set OMP/MKL/OPENBLAS thread count for CPU-heavy parts.
  --fp64               Enable JAX 64-bit (x64). Default: off (FP32) on GPU for speed.
  --deterministic      Enable deterministic GPU ops (can be slower).
  --opt                Contraction optimizer preset for quimb/cotengra (default: auto-hq).
  -q, --nqubits        Override pt.n_qubits in YAML.
  -s, --nsteps         Override pt.n_steps in YAML.

Examples:
  # Fast GPU, FP32, all CPU threads, config as-is:
  ./ptnt-launch.sh -c configs/figure3.yaml -d cuda

  # CPU only, 16 threads:
  ./ptnt-launch.sh -c configs/figure3.yaml -d cpu -t 16

  # GPU but make runs reproducible-ish (deterministic) and small override:
  ./ptnt-launch.sh -c configs/figure3.yaml -d cuda --deterministic -q 3 -s 3

  # GPU with greedy path (may be faster for tiny problems):
  ./ptnt-launch.sh -c configs/figure3.yaml -d cuda --opt greedy
USAGE
}

# ---- parse args
CFG=""; DEVICE="cpu"; THREADS=""; FP64="0"; DET="0"; OPT="auto-hq"; NQ=""; NS=""
while (( "$#" )); do
  case "$1" in
    -c|--config)       CFG="$2"; shift 2 ;;
    -d|--device)       DEVICE="$2"; shift 2 ;;
    -t|--threads)      THREADS="$2"; shift 2 ;;
    --fp64)            FP64="1"; shift ;;
    --deterministic)   DET="1"; shift ;;
    --opt)             OPT="$2"; shift 2 ;;
    -q|--nqubits)      NQ="$2"; shift 2 ;;
    -s|--nsteps)       NS="$2"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$CFG" ]]; then usage; exit 2; fi
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "cuda" ]]; then
  echo "Invalid --device '$DEVICE' (use 'cpu' or 'cuda')" >&2; exit 2
fi

# ---- environment (fast defaults)
if [[ "$DEVICE" == "cuda" ]]; then
  export JAX_PLATFORMS=cuda
  export JAX_DEFAULT_MATMUL_PRECISION=high    # valid: default|high|highest
  [[ "$FP64" == "1" ]] && export JAX_ENABLE_X64=True || export JAX_ENABLE_X64=False
  if [[ "$DET" == "1" ]]; then
    export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
  else
    unset XLA_FLAGS || true
  fi
else
  export JAX_PLATFORMS=cpu
  # On CPU you can safely use x64 by default if you care about numerics:
  [[ "$FP64" == "1" ]] && export JAX_ENABLE_X64=True || export JAX_ENABLE_X64=True
  unset JAX_DEFAULT_MATMUL_PRECISION || true
  # Optional CPU determinism knob:
  [[ "$DET" == "1" ]] && export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" || unset XLA_FLAGS || true
fi

# Threads for CPU-side work (path search, Aer, BLAS):
if [[ -n "${THREADS}" ]]; then
  export OMP_NUM_THREADS="$THREADS"
  export MKL_NUM_THREADS="$THREADS"
  export OPENBLAS_NUM_THREADS="$THREADS"
fi

# ---- patch YAML overrides if requested
TMP_CFG="$(mktemp /tmp/ptnt_cfg_XXXX.yaml)"
python - "$CFG" "$TMP_CFG" "$NQ" "$NS" <<'PY'
import sys, yaml
src, dst, nq, ns = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src) as f:
    cfg = yaml.safe_load(f)
if nq:
    cfg.setdefault("pt", {})["n_qubits"] = int(nq)
if ns:
    cfg.setdefault("pt", {})["n_steps"] = int(ns)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f)
print(dst)
PY

# ---- show effective runtime config
python - <<'PY'
import os, jax
print("=== PTNT runtime ===")
print("devices:", jax.devices())
for k in ["JAX_PLATFORMS","JAX_ENABLE_X64","JAX_DEFAULT_MATMUL_PRECISION","XLA_FLAGS",
          "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    print(f"{k}={os.environ.get(k)}")
PY

# ---- run
set -x
ptnt run "$TMP_CFG" --device "$DEVICE" ${THREADS:+--num-threads "$THREADS"} --opt "$OPT"
set +x

