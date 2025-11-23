# ptnt/cli.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
import yaml

"""
How control flows at runtime

    User runs: ptnt ...
    We parse CLI *first*, apply environment overrides (device, threads, precision),
    THEN import the pipeline runner. This ensures JAX sees the right env.

Subcommands:
    ptnt run <config.yaml>        -> load YAML & run
    ptnt experiment <name> [--out DIR] -> load a packaged YAML (figure3|noise_pink|spillage)

Extra overrides (optional, apply to either subcommand):
    --device {cpu,cuda}           -> select JAX platform
    --num-threads N               -> OMP/MKL/OPENBLAS threading
    --opt OPT                     -> contraction path ('auto-hq' default)
    --mode {normal,X_decomp,auto} -> likelihood representation
    --q N, --steps T              -> override n_qubits / n_steps
"""


def _apply_env(device: str | None, num_threads: int | None):
    # device
    if device:
        if device not in {"cpu", "cuda"}:
            raise SystemExit(f"--device must be 'cpu' or 'cuda', got {device}")
        os.environ["JAX_PLATFORMS"] = device
        # Good defaults: float32 on GPU unless we need x64
        if device == "cuda":
            os.environ.setdefault("JAX_ENABLE_X64", "False")
            os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "high")
        else:
            os.environ.setdefault("JAX_ENABLE_X64", "True")
            os.environ.pop("JAX_DEFAULT_MATMUL_PRECISION", None)
        # don't force determinism here; user can export XLA_FLAGS outside

    # threads
    if num_threads is not None:
        nt = str(num_threads)
        os.environ["OMP_NUM_THREADS"] = nt
        os.environ["MKL_NUM_THREADS"] = nt
        os.environ["OPENBLAS_NUM_THREADS"] = nt


def _merge_overrides(
    cfg: dict,
    device: str | None,
    num_threads: int | None,
    opt: str | None,
    mode: str | None,
    n_qubits: int | None,
    n_steps: int | None,
):
    # device/basis are already in YAML; we don't force basis here
    if n_qubits is not None:
        cfg.setdefault("pt", {})["n_qubits"] = int(n_qubits)
    if n_steps is not None:
        cfg.setdefault("pt", {})["n_steps"] = int(n_steps)
    if opt is not None:
        cfg.setdefault("training", {})["opt"] = str(opt)
    if mode is not None:
        if mode not in {"normal", "X_decomp", "auto"}:
            raise SystemExit("--mode must be one of {normal,X_decomp,auto}")
        cfg.setdefault("training", {})["mode"] = mode
    # we don't store num_threads in cfg; it is applied via env before JAX import

    # for completeness, record device for report/debug (doesn't control JAX itself)
    if device is not None:
        cfg.setdefault("runtime", {})["device_cli"] = device
        cfg.setdefault("runtime", {})["num_threads_cli"] = num_threads
    return cfg


def main():
    parser = argparse.ArgumentParser(
        prog="ptnt", description="Process Tensor Network Tomography pipelines"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ptnt run <config.yaml>
    p_run = sub.add_parser("run", help="Run from a YAML config file")
    p_run.add_argument("config", type=Path, help="Path to YAML config")

    # ptnt experiment <name> [--out <dir>]
    p_exp = sub.add_parser(
        "experiment", help="Run a named experiment [figure3|noise_pink|spillage]"
    )
    p_exp.add_argument("name", choices=["figure3", "noise_pink", "spillage"])
    p_exp.add_argument(
        "--out", type=Path, default=None, help="Override output dir"
    )

    # common overrides
    for p in (p_run, p_exp):
        p.add_argument(
            "--device",
            choices=["cpu", "cuda"],
            default=None,
            help="Force JAX platform (cpu|cuda)",
        )
        p.add_argument(
            "--num-threads",
            type=int,
            default=None,
            help="BLAS threading (OMP/MKL/OPENBLAS)",
        )
        p.add_argument(
            "--opt",
            default=None,
            help="Contraction optimizer preset or name (default auto-hq)",
        )
        p.add_argument(
            "--mode",
            choices=["normal", "X_decomp", "auto"],
            default=None,
            help="Likelihood mode",
        )
        p.add_argument(
            "--q",
            type=int,
            default=None,
            help="Override number of qubits",
        )
        p.add_argument(
            "--steps",
            type=int,
            default=None,
            help="Override number of time steps",
        )

    args = parser.parse_args()

    # 1) Apply env before importing any JAX code
    _apply_env(
        device=getattr(args, "device", None),
        num_threads=getattr(args, "num_threads", None),
    )

    # 2) Now import the runner (this imports jax, quimb, etc.)
    from .io.run import run_from_config, default_config_for_experiment  # noqa: F401

    # 3) Load config, merge CLI overrides, and run
    if args.cmd == "run":
        cfg = yaml.safe_load(Path(args.config).read_text())
        cfg = _merge_overrides(
            cfg, args.device, args.num_threads, args.opt, args.mode, args.q, args.steps
        )
        run_from_config(cfg)
    else:
        cfg = default_config_for_experiment(args.name)
        if args.out is not None:
            cfg.setdefault("output", {})["dir"] = str(args.out)
        cfg = _merge_overrides(
            cfg, args.device, args.num_threads, args.opt, args.mode, args.q, args.steps
        )
        run_from_config(cfg)

