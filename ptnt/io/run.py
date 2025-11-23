# ptnt/io/run.py  (HPC: pre-built contractor, cached causality keys, bond policy helper)
from __future__ import annotations
import time
import yaml
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from tqdm import trange
from qiskit.circuit import Parameter
from qiskit_aer import Aer  # used inside make_backend

from ..circuits.noise_models import (
    make_backend,
    make_coherent_depol_noise_model,
    create_env_IA,
    pink_noise_series,
)
from ..circuits.templates import base_PT_circ_template
from ..circuits.utils import bind_ordered

from ..preprocess.shadow import (
    shadow_results_to_data_vec,
    shadow_seqs_to_op_array_rz,
    shadow_seqs_to_op_array,
    pure_measurement,
    clifford_rz_unitaries_vT,
    clifford_measurements_vT,
    clifford_unitaries_vT,
    val_rz_unitaries_vT,
    val_unitaries_vT,
    val_measurements_vT,
    clifford_param_dict,
    validation_param_dict,
)

from ..tn.optimize import TNOptimizer
from ..tn.fit import (
    compute_likelihood,
    causality_keys_to_op_arrays,
    compute_probabilities,
    make_contractor,  # pre-built cotengra/quimb contractor
)
from ..tn.pepo import create_PT_PEPO_guess, expand_initial_guess_, create_PEPO_X_decomp
from ..utilities import hellinger_fidelity
from .report import write_report


# ---------------------------------------------------------------------
# Config convenience
# ---------------------------------------------------------------------
def default_config_for_experiment(name: str) -> dict:
    here = Path(__file__).resolve().parents[2] / "configs"
    with open(here / f"{name}.yaml", "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------
# Circuit template + shadow batch generation
# ---------------------------------------------------------------------
def _shadow_template(cfg):
    nQ = cfg["pt"]["n_qubits"]
    nS = cfg["pt"]["n_steps"]
    backend = make_backend(cfg["device"]["backend"])
    basis = cfg["device"]["basis_gates"]
    env = create_env_IA(**cfg["pt"]["env_IA"])
    kwargs = {}
    if "crosstalk_IA" in cfg["pt"] and cfg["pt"]["crosstalk_IA"] is not None:
        kwargs["crosstalk_IA"] = create_env_IA(**cfg["pt"]["crosstalk_IA"])
    if cfg["pt"]["template"] == "dd_rx_error":
        kwargs["error_param"] = Parameter("err_X")
    if cfg["pt"]["template"] == "dd_spillage":
        sp = cfg["pt"].get("spillage", {})
        kwargs["spillage_rot"] = sp.get("control_rotation", 0.0)
        kwargs["init_qubits"] = sp.get("init_qubits", [0])
    circ = base_PT_circ_template(
        nQ,
        nS,
        backend,
        basis,
        cfg["pt"]["template"],
        env,
        **kwargs,
    )
    return circ


def _generate_shadow_batches(cfg, template, nS):
    """
    Create batches of (bound) circuits + their shadow index sequences.

    IMPORTANT:
      * bind **by name** to avoid any parameter-order surprises.
      * if template==dd_rx_error and data.pink_noise.enabled, bind err_X per-circuit.
    """
    nQ = cfg["pt"]["n_qubits"]
    # Which params exist in this template?
    template_param_names = {p.name for p in template.parameters}
    has_err = ("err_X" in template_param_names)

    # Pink control-noise settings (optional)
    pn_cfg = cfg.get("data", {}).get("pink_noise", {}) or {}
    pink_enabled = bool(pn_cfg.get("enabled", False)) and has_err
    pink_max = float(pn_cfg.get("max_angle", 0.0))
    base_seed = int(cfg.get("seed", 0))

    def one_batch(n, param_dict, seed_offset=0):
        seqs, circs = [], []
        # If enabled, draw a quasi-static, slowly drifting err_X series for these n circuits
        angles = None
        if pink_enabled:
            angles = pink_noise_series(
                n=n,
                max_angle=pink_max,
                random_state=base_seed + seed_offset,
            )

        for idx in range(n):
            # For each (t, q) pick an index -> U3 triple
            seq = np.random.randint(0, len(param_dict), size=(nS + 1, nQ))
            # Build a *mapping* name -> value
            mapping = {}
            for t in range(nS + 1):
                for q in range(nQ):
                    th, ph, la = param_dict[int(seq[t, q])]
                    mapping[f"t{t}_q{q}_x"] = float(th)
                    mapping[f"t{t}_q{q}_y"] = float(ph)
                    mapping[f"t{t}_q{q}_z"] = float(la)
            if has_err:
                mapping["err_X"] = 0.0 if angles is None else float(angles[idx])

            circs.append(bind_ordered(template, mapping))  # mapping-safe binder
            # NB: downstream uses (nQ, nS+1) ordering
            seqs.append(seq.T)
        return circs, seqs

    return one_batch


# ---------------------------------------------------------------------
# Bond policy helper (Fix #3)
# ---------------------------------------------------------------------
def _bond_policy(cfg: dict, nQ: int, nS: int):
    """
    Decide seed and target bond dimensions for the PEPO.

    Returns:
        seed_K_lists      : list[list[int]] shape (nQ, nS+1)
        seed_horiz_bonds  : list[int]       length nQ         (temporal bonds for create_PT_PEPO_guess)
        seed_vert_bonds   : list[list[int]] shape (nS+1, nQ-1)
        target_K_lists    : list[list[int]] shape (nQ, nS+1)
        target_horiz_bonds: list[list[int]] shape (nQ, nS+1)
        target_vert_bonds : list[list[int]] shape (nS+1, nQ-1)

    Design:
      * Seed network is minimal (all bond dims = 1) -> cheap to build.
      * Target bonds encode a simple physics-inspired heuristic:
            - Kraus rank >1 at first/last time slice (capture prep/readout memory),
            - uniform modest bond_dim (default 2) elsewhere.
      * Advanced users can override target bonds via:
            training.K_lists, training.horizontal_bonds, training.vertical_bonds
        if they supply shapes compatible with (nQ, nS+1) / (nS+1, nQ-1).
    """
    tr = cfg.get("training", {}) or {}

    # Base bond dimension (can be overridden from YAML)
    max_dim = int(tr.get("bond_dim", 2))

    nT = nS + 1            # number of time slices (0..nS)
    nV = max(nQ - 1, 0)    # vertical bonds per time slice

    # -----------------------
    # Seed: everything = 1
    # -----------------------
    seed_K = [[1] * nT for _ in range(nQ)]
    # one temporal bond dim per qubit for create_PT_PEPO_guess
    seed_horiz = [1 for _ in range(nQ)]
    # vertical bonds (not used if nQ == 1)
    seed_vert = [[1] * nV for _ in range(nT)]

    # -----------------------
    # Default target policy
    # -----------------------
    if nS >= 2:
        # Enlarge Kraus rank at first/last time step
        default_row = [max_dim] + [1] * (nS - 1) + [max_dim]
    else:
        # Short time window -> just use max_dim everywhere
        default_row = [max_dim] * nT

    target_K = [default_row[:] for _ in range(nQ)]
    target_horiz = [[max_dim] * nT for _ in range(nQ)]
    target_vert = [[max_dim] * nV for _ in range(nT)]

    # -----------------------
    # Optional YAML overrides
    # -----------------------
    # K_lists: list-of-lists [nQ][nS+1]
    user_K = tr.get("K_lists")
    if user_K is not None:
        try:
            if len(user_K) != nQ:
                raise ValueError
            for row in user_K:
                if len(row) != nT:
                    raise ValueError
            target_K = [[int(x) for x in row] for row in user_K]
        except Exception:
            print(
                f"[ptnt] Warning: training.K_lists has incompatible shape; "
                f"expected ({nQ}, {nT}), got something else. Using default bond policy."
            )

    # horizontal_bonds: can be
    #   - 2D: [nQ][nS+1] per (qubit,time)
    #   - 1D: [nS] or [nS+1] per time, broadcast over qubits
    user_h = tr.get("horizontal_bonds")
    if user_h is not None:
        try:
            if isinstance(user_h, (list, tuple)) and user_h and isinstance(
                user_h[0], (list, tuple)
            ):
                # 2D
                if len(user_h) != nQ:
                    raise ValueError
                rows = []
                for row in user_h:
                    row = list(row)
                    if len(row) == nS:
                        # pad with last entry to reach nS+1
                        row = row + [row[-1]]
                    if len(row) != nT:
                        raise ValueError
                    rows.append([int(x) for x in row])
                target_horiz = rows
            else:
                # 1D per-time list
                row = list(user_h)
                if len(row) == nS:
                    row = row + [row[-1]]
                if len(row) != nT:
                    raise ValueError
                row = [int(x) for x in row]
                target_horiz = [row[:] for _ in range(nQ)]
        except Exception:
            print(
                "[ptnt] Warning: training.horizontal_bonds has incompatible shape; "
                "using default horizontal bond policy."
            )

    # vertical_bonds: list-of-lists [nS+1][nQ-1]; only meaningful if nQ > 1
    user_v = tr.get("vertical_bonds")
    if user_v is not None and nV > 0:
        try:
            rows = list(user_v)
            if len(rows) not in {nS, nT}:
                raise ValueError
            if len(rows) == nS:
                # pad last row
                rows = rows + [rows[-1]]
            if len(rows) != nT:
                raise ValueError
            parsed = []
            for row in rows:
                row = list(row)
                if len(row) != nV:
                    raise ValueError
                parsed.append([int(x) for x in row])
            target_vert = parsed
        except Exception:
            print(
                "[ptnt] Warning: training.vertical_bonds has incompatible shape; "
                "using default vertical bond policy."
            )

    return seed_K, seed_horiz, seed_vert, target_K, target_horiz, target_vert


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def run_from_config(cfg: dict):
    # Seed NP draws for reproducibility of circuit choices
    np.random.seed(cfg.get("seed", 0))

    # Output directory
    run_dir = Path(cfg.get("output", {}).get("dir", f"runs/{int(time.time())}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build one symbolic template; pull sizes
    template = _shadow_template(cfg)
    nQ = cfg["pt"]["n_qubits"]
    nS = cfg["pt"]["n_steps"]

    # Optional noise model
    noise_model = None
    if "noise" in cfg.get("device", {}):
        nz = cfg["device"]["noise"]
        if nz and "depolarizing_p1q" in nz and "coherent_rx_angle" in nz:
            noise_model = make_coherent_depol_noise_model(
                nz["depolarizing_p1q"],
                nz["coherent_rx_angle"],
            )

    backend = make_backend(cfg["device"]["backend"])

    # -----------------------
    # Validation set
    # -----------------------
    data = cfg["data"]
    val_n = data.get("val_shadows", data.get("val_circs", 300))
    shots_val = data.get("shots_per_val", 16384)

    one_batch = _generate_shadow_batches(cfg, template, nS)
    val_circs, val_sequences = one_batch(
        val_n,
        validation_param_dict,
        seed_offset=1_000_000,
    )
    job = backend.run(val_circs, shots=shots_val, noise_model=noise_model)
    val_counts = job.result().get_counts()
    val_data, val_keys = shadow_results_to_data_vec(val_counts, shots=shots_val, nQ=nQ)

    # -----------------------
    # Training set (Cliffords)
    # -----------------------
    if "char_circs" in data:
        char_n = data["char_circs"]
        shots_char = data["shots_per_char"]
        char_circs, shadow_sequences = one_batch(
            char_n,
            clifford_param_dict,
            seed_offset=2_000_000,
        )
        jobc = backend.run(char_circs, shots=shots_char, noise_model=noise_model)
        training_counts = jobc.result().get_counts()
    else:
        jobs = data.get("n_jobs", data.get("jobs", 20))
        per = data.get("shadows_per_job", 300)
        shots_char = data.get("shots_per_char", 1024)
        training_counts = []
        shadow_sequences = []
        for j in trange(jobs, desc="characterization jobs"):
            tmp_circs, tmp_seq = one_batch(
                per,
                clifford_param_dict,
                seed_offset=10_000 + j,
            )
            tmp_job = backend.run(tmp_circs, shots=shots_char, noise_model=noise_model)
            training_counts += tmp_job.result().get_counts()
            shadow_sequences += tmp_seq

    training_data_flat, training_keys = shadow_results_to_data_vec(
        training_counts,
        shots_char,
        nQ,
    )

    # -----------------------
    # Reverse sequences (time goes right->left inside the TN)
    # -----------------------
    def reverse_seq_list(seq_list):
        out = []
        for seq in seq_list:
            tmp = []
            for T in seq:
                # reverse time only; DO NOT reverse qubit axis
                tmp.append([o for o in reversed(T)])
            out.append(tmp)
        return out

    train_seq_rev = reverse_seq_list(shadow_sequences)
    val_seq_rev = reverse_seq_list(val_sequences)

    # -----------------------
    # Build operator arrays for both “views”
    # -----------------------
    train_full = shadow_seqs_to_op_array(
        train_seq_rev,
        training_keys,
        clifford_measurements_vT,
        clifford_unitaries_vT,
    )
    val_full = shadow_seqs_to_op_array(
        val_seq_rev,
        val_keys,
        val_measurements_vT,
        val_unitaries_vT,
    )
    train_rz = shadow_seqs_to_op_array_rz(
        train_seq_rev,
        training_keys,
        pure_measurement,
        clifford_rz_unitaries_vT,
    )
    val_rz = shadow_seqs_to_op_array_rz(
        val_seq_rev,
        val_keys,
        pure_measurement,
        val_rz_unitaries_vT,
    )

    # -----------------------
    # Initial PEPO guess (small scaffold then expand)
    # -----------------------
    import quimb as qu

    (
        seed_K_lists,
        seed_horiz_bonds,
        seed_vert_bonds,
        real_K_lists,
        real_horiz_bonds,
        real_vert_bonds,
    ) = _bond_policy(cfg, nQ, nS)

    # Tiny seed PEPO with minimal bond dimensions
    initial = create_PT_PEPO_guess(
        nS,
        nQ,
        seed_horiz_bonds,
        seed_vert_bonds,
        seed_K_lists,
    )
    initial = qu.tensor.tensor_2d.TensorNetwork2DFlat.from_TN(
        initial,
        site_tag_id="q{}_I{}",
        Ly=nS + 1,
        Lx=nQ,
        y_tag_id="ROWq{}",
        x_tag_id="COL{}",
    )

    # Inflate seed PEPO to target bond dims, add small random noise
    expand_initial_guess_(
        initial,
        real_K_lists,
        real_horiz_bonds,
        real_vert_bonds,
        rand_strength=0.05,
        squeeze=False,
    )
    initial.squeeze_()
    Lx, Ly = initial.Lx, initial.Ly

    X_guess = create_PEPO_X_decomp(nS, nQ)

    # -----------------------
    # Choose mode: normal (Full-U), X_decomp (RZ), or auto
    # -----------------------
    mode_cfg = cfg["training"].get("mode", "auto")
    if mode_cfg == "auto":
        # For multi-qubit / deeper circuits, prefer the cheaper RZ/X decomposition
        mode = "X_decomp" if (nQ >= 2 and nS >= 7) else "normal"
    else:
        mode = mode_cfg

    bundles = {
        "normal": dict(
            initial_tn=initial,
            train=train_full,
            val=val_full,
            const=False,
        ),
        "X_decomp": dict(
            initial_tn=(initial & X_guess),
            train=train_rz,
            val=val_rz,
            const=True,
        ),
    }

    # -----------------------
    # Empirical entropy baselines
    # -----------------------
    td = np.array(training_data_flat, dtype=float)
    td[td < 1e-12] = 1e-12
    vd = np.array(val_data, dtype=float)
    vd[vd < 1e-12] = 1e-12
    data_entropy = float(-(1 / len(td)) * td @ np.log(td + 1e-18))
    v_data_entropy = float(-(1 / len(vd)) * vd @ np.log(vd + 1e-18))

    # -----------------------
    # Training hyperparameters
    # -----------------------
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    iterations = int(2 * epochs * len(training_data_flat) / batch_size)

    # Contraction preset from YAML/CLI; keep 'auto-hq' as robust default
    opt_preset = cfg["training"].get("opt", "auto-hq")
    contractor = make_contractor(opt_preset)

    # Optional cadence for refreshing Monte-Carlo causality keys
    refresh_cadence = cfg["training"].get("refresh_causality_every", None)

    optmzr = TNOptimizer(
        bundles[mode]["initial_tn"],
        loss_fn=compute_likelihood,
        causality_fn=causality_keys_to_op_arrays,
        causality_key_size=cfg["training"]["causality_key_size"],
        training_data=td,
        training_sequences=bundles[mode]["train"],
        Lx=Lx,
        Ly=Ly,
        validation_data=list(vd),
        validation_sequences=bundles[mode]["val"],
        batch_size=batch_size,
        loss_constants={},
        loss_kwargs={
            "kappa": cfg["training"]["kappa"],
            "opt": contractor,              # pass contractor into JAX loss
            "X_decomp": bundles[mode]["const"],
        },
        autodiff_backend=cfg["training"]["autodiff"],
        optimizer=cfg["training"]["optimizer"],
        progbar=True,
        refresh_causality_every=refresh_cadence,  # controls re-drawing cArrays
    )
    pepo_opt = optmzr.optimize(iterations)
    best_val_mpo = optmzr.best_val_mpo

    # -----------------------
    # Validation predictions & metrics
    # -----------------------
    v_pred = compute_probabilities(
        best_val_mpo,
        bundles[mode]["val"],
        X_decomp=bundles[mode]["const"],
        opt=contractor,  # reuse same contraction strategy for validation
    )
    v_pred = sum(vd) * v_pred / sum(v_pred)

    # Per-circuit fidelities (Hellinger on each 2^nQ-sized marginal)
    fids = []
    b = 2**nQ
    for i in range(len(vd) // b):
        p = np.array(v_pred[b * i : b * (i + 1)])
        p = p / max(p.sum(), 1e-16)
        a = np.array(vd[b * i : b * (i + 1)])
        fids.append(hellinger_fidelity(p, a))

    # One loss per epoch (last batch each epoch)
    nB = optmzr._nBatches
    epoch_losses = [
        float(optmzr.losses[nB - 1 + nB * i])
        for i in range(int(optmzr._n / nB))
    ]
    epoch_val_losses = [
        float(optmzr.val_losses[nB - 1 + nB * i])
        for i in range(int(optmzr._n / nB))
    ]

    # Report
    metrics = dict(
        title=f"ptnt ({mode})",
        data_entropy=data_entropy,
        v_data_entropy=v_data_entropy,
        epoch_losses=epoch_losses,
        epoch_val_losses=epoch_val_losses,
        fidelities=[float(x) for x in fids],
    )
    write_report(Path(cfg.get("output", {}).get("dir", ".")), metrics, prefix="ptnt")
    return metrics

