# ptnt/preprocess/shadow.py
'''
tables + converters to operator arrays 

functions to preprocess the data and create relevant objects, with
robust, file-free fallbacks for the small 'Clifford tables' we need.

If ptnt/ptnt/data/{clifford_params.npy, validation_params.pickle} are missing,
we synthesize them on first import and write them to ptnt/ptnt/data/ when
possible (silently skipping the write if the directory is not writable)



On import, ensures the Clifford and validation tables exist (either loaded from ptnt/data/ or synthesized and cached)
Imports & utilities (U3 conversions, Choi vectors):
    _unitary_to_u3(U) computes Qiskit’s U3(θ,φ,λ) params from a 2×2 SU(2) unitary (remove global phase, extract angles from entries)
    _Rx, _Rz build explicit 2×2 rotation matrices
    _canonical_phase_distance checks if two unitaries are equal up to global phase



Noise composed on sx gives us realistic error channels where real devices actually err.
Clifford vs U3 tables guarantee broad basis coverage (training) and robust validation.
Endianness fix guarantees our qubit-to-bit mapping is consistent (no silent swaps).
Operator arrays provide a regular, batched tensor interface so our likelihood is just a series of efficient contractions (and autodiff works).
Two views (Full-U / RZ) let us pick the representation that contracts fastest/most stably for your optimizer.


High‑level:
    At import time, make sure two small tables exist:
        24 one‑qubit Clifford unitaries -> stored as U3 triples
        100 validation unitaries -> random U3 triples
    If files are missing, synthesize them, and cache to ptnt/data/ (best‑effort)
    Build vectorized tensors (Choi‑vector form) for those unitaries and for measurement operators, ready for tensor‑network contractions
    Provide converters from “lists of sampled Clifford/U3 indices+measurement outcomes” to batched operator arrays with fixed shapes that the optimizer consumes

    
this file:
Prepares the tables and vectorized operator tensors (unitaries & measurements), and
Converts sampled sequences into batch operator arrays (and can pack a sample’s operators into a 2D TN for contraction).
The initial PEPO guess (the model you optimize) is created elsewhere :
    In ptnt/tn/pepo.py: create_PT_PEPO_guess(...), expand_initial_guess_(...), create_PEPO_X_decomp(...)
    Wired up in ptnt/io/run.py: they call those functions to build and expand the initial process‑tensor network before trainig


a. Parameter tables (load or synthesize + cache)
    _ensure_parameter_tables()
        If ptnt/data/clifford_params.npy exists -> np.load it. Else ->  _generate_clifford_u3_params() -> np.save best‑effort
        If ptnt/data/validation_params.pickle exists ->  pickle.load. Else ->  _generate_validation_params(n=100, seed=11) -> pickle.dump best‑effort.

    On import:
        clifford_params, validation_param_dict = _ensure_parameter_tables()
        So by the time anything uses this module, those tables exist in memory (and likely on disk)
b. Make unitary matrices and decompositions from U3
    make_unitary(*C) turns each U3 triple into a 2×2 matrix (JAX array).
    u3_to_rz_params(...) gives the three RZ angles per U3 triple; rz_unitary(...) turns each into a 2×2 Rz matrix.

c. Vectorized blocks (“Choi vectors”) for contraction
    For Full‑U view: clifford_unitaries_vT / val_unitaries_vT map index -> (1,2,2) tensor.
    For RZ view: clifford_rz_unitaries_vT / val_rz_unitaries_vT map index -> list of three (1,2,2) tensors (reversed for contraction order).
    Measurement tensors:
        Full‑U: clifford_measurements_vT, val_measurements_vT (pre‑rotated by U†)
        RZ‑view: pure_measurement (computational basis)

d. Converters used by your pipeline
    shadow_results_to_data_vec(...): counts -> probabilities (with a bitstring reversal to fix endianness in a consistent way)
    shadow_seqs_to_op_array(...): sequences -> Full‑U batched operator arrays with shape (samples, nQ, nsteps+1, 2, 2)
    shadow_seqs_to_op_array_rz(...): sequences -> RZ‑decomp batched arrays with shape (samples, nQ, 3*(nsteps+1)+1, 2, 2)
    The two op_arrays_to_single_vector_TN_padded* helpers can pack one sample’s operator array into a 2D tensor network (rows=time, cols=qubits) for contraction—but these build the observation TN, not the model PEPO.

    

“RZ view” is just a representation choice for each single‑qubit local gate in your shadow sequences. 
Instead of carrying a whole U3(θ,φ,λ) block per time step (“Full‑U” view), 
we factor that gate into three Z‑axis rotations (RZs), and you push the SX toggles into a fixed network that is handled elsewhere. 
This pays off for tensor‑network likelihoods: diagonal Z‑rotations are simpler objects to contract and keep the per‑sample operator stream very regular.

'''

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import jax.numpy as jnp
import quimb.tensor as qtn

# Import only what we actually use
# Utilities we used:
#   make_unitary(θ,φ,λ) -> 2×2 from U3 angles.
#   u3_to_rz_params(...) converts a U3 triple into three RZ angles (the U3 -> Rz–Sx–Rz–Sx–Rz structure used in our templates)
#   rz_unitary(θ) returns diag(1, e^{iθ})
#   TN_choi_vec(U) returns the vectorized operator (Choi‑style “ket”) for a 1‑qubit unitary.
#   zero_vec, one_vec are |0⟩, |1⟩ column vectors.
from ptnt.utilities import (
    make_unitary,
    u3_to_rz_params,
    rz_unitary,
    TN_choi_vec,
    zero_vec,
    one_vec,
)

# ---------------------------------------------------------------------
# Paths & small helpers
# ---------------------------------------------------------------------

# Compute ptnt/preprocess directory and a sibling ptnt/data dir to cache parameter tables. Create it if missing
# Two target files:
#   clifford_params.npy -> (24, 3) float array of U3 triples
#   validation_params.pickle -> dict {i: np.array([θ,φ,λ])} for 100 random unitaries

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[0]
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

val_file_path = data_dir / "validation_params.pickle"
cliff_file_path = data_dir / "clifford_params.npy"


# Phase handling and U3 decomposition tools
# SU(2) has determinant 1; arbitrary U(2) has a global phase
# This normalizes U so det(U)=1 (divide by √det). If det≈0 (degenerate numeric corner), return as is
def _remove_global_phase(U: np.ndarray) -> np.ndarray:
    # Return U with determinant set to 1 (remove global phase)
    det = np.linalg.det(U)
    if np.isclose(det, 0.0):
        return U
    return U / np.sqrt(det)


def _unitary_to_u3(U: np.ndarray) -> Tuple[float, float, float]:
    '''
    Decompose a 2x2 SU(2) unitary into Qiskit's U3(θ, φ, λ) parameters.

    U3(θ, φ, λ) = [[cos(θ/2), -e^{iλ} sin(θ/2)],
                   [e^{iφ} sin(θ/2), e^{i(φ+λ)} cos(θ/2)]]
    
    Ensure complex array & phase‑normalize. Name entries for clarity:
    U= [a	b
        c	d]

    '''
    U = np.asarray(U, dtype=complex)
    U = _remove_global_phase(U)
    a, b = U[0, 0], U[0, 1]
    c, d = U[1, 0], U[1, 1]

    # θ from amplitudes
    # cos(θ/2) = |a| = |d|, sin(θ/2) = |b| = |c|
    # Extract θ directly from magnitudes consistent with the U3 form
    cos_th_2 = np.clip(np.abs(a), 0.0, 1.0)
    sin_th_2 = np.clip(np.abs(b), 0.0, 1.0)
    theta = 2.0 * np.arctan2(sin_th_2, cos_th_2)

    # Edge case sin(θ/2)≈0 (near identity): phases in b,c are ill‑defined; choose a consistent convention: set ϕ=0, λ=arg(d)
    eps = 1e-12
    if sin_th_2 < eps:
        # θ ≈ 0 -> a≈1, b≈c≈0, d≈e^{i(φ+λ)}; set φ=0, λ=arg(d)
        phi = 0.0
        lam = float(np.angle(d))
        return float(theta), float(phi), float(lam)

    # General case
    # c = e^{iφ} sin(θ/2) -> φ = arg(c)
    # b = -e^{iλ} sin(θ/2) -> λ = arg(-b)
    # For general position: read off phases from c and -b exactly as the U3 formula states
    phi = float(np.angle(c))
    lam = float(np.angle(-b))

    # Optional consistency check; can be relaxed in noisy case
    # d_pred = np.exp(1j * (phi + lam)) * cos_th_2
    # if not np.allclose(d / d_pred, 1.0, atol=1e-6):
    #     pass  # Accept small numeric deviations

    return float(theta), float(phi), float(lam)


# Explicit R_x(θ)
def _Rx(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
            [-1j * np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=complex,
    )


def _Rz(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * phi / 2.0), 0], [0, np.exp(1j * phi / 2.0)]],
        dtype=complex,
    )


# Computes Frobenius‑norm distance mod global phase by aligning B to A via phase from inner product
def _canonical_phase_distance(A: np.ndarray, B: np.ndarray) -> float:
    # Distance between A and B modulo global phase
    # phase that best aligns B to A in Frobenius sense
    phase = np.angle(np.trace(B.conj().T @ A))
    return np.linalg.norm(A - np.exp(1j * phase) * B)


# Generate the 24 single-qubit Cliffords
# Builds the 24 one-qubit Cliffords by closing over the four quarter-turn generators, 
#   keeping only new elements mod global phase, then sorting deterministically. 
#   Converts each to U3 parameters, returns a (24,3) array


def _generate_clifford_u3_params() -> np.ndarray:
    # Generate the 24 single-qubit Clifford unitaries via closure of Rx(π/2), Rz(π/2),
    #   then convert to U3(θ, φ, λ). Deterministic and order-stable.
    # Start with the identity and the standard quarter‑turn generators that generate the single‑qubit Clifford group

    I = np.eye(2, dtype=complex)
    gens = [
        _Rx(np.pi / 2),
        _Rx(-np.pi / 2),
        _Rz(np.pi / 2),
        _Rz(-np.pi / 2),
    ]

    # BFS‑like closure under left multiplication by generators. Add V if it’s new modulo global phase. Stop when you have 24
    unitaries: List[np.ndarray] = [I]
    # simple closure up to short words is sufficient to reach 24
    # expand until we collect 24 unique (mod global phase)
    queue = [I]
    tol = 1e-10
    while len(unitaries) < 24:
        U = queue.pop(0)
        for G in gens:
            V = G @ U
            # check uniqueness modulo global phase
            if all(_canonical_phase_distance(V, W) > tol for W in unitaries):
                unitaries.append(V)
                queue.append(V)
        # Safety: if left‑multiplication BFS stalls early (numerics), also try right multiplication to complete the set
        if not queue and len(unitaries) < 24:
            # as a fallback, also try right-multiplication
            for W in list(unitaries):
                for G in gens:
                    V = W @ G
                    if all(_canonical_phase_distance(V, Z) > tol for Z in unitaries):
                        unitaries.append(V)
                        queue.append(V)

    # Sort deterministically by a simple key (rounded real/imag of flattened entries)
    # Make ordering stable across runs (important for reproducibility and indexing)
    def _key(U):
        r = np.round(np.real(U).flatten(), 6)
        i = np.round(np.imag(U).flatten(), 6)
        return tuple(np.concatenate([r, i]).tolist())

    unitaries.sort(key=_key)

    # Convert to U3 params (θ, φ, λ)
    # Convert to U3(θ,φ,λ) triples and return as a (24,3) array
    params = [np.array(_unitary_to_u3(U), dtype=float) for U in unitaries]
    return np.vstack(params)  # (24, 3)


# Generate validation U3 table (random)
# Makes 100 seeded random triples for validation
# Make n IID triples in  [0,2π). Return a dict {i: [θ,φ,λ]}.
def _generate_validation_params(n: int = 100, seed: int = 7) -> Dict[int, np.ndarray]:
    # Generate n random U3(θ, φ, λ) parameter triples
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phis = rng.uniform(0.0, 2.0 * np.pi, size=n)
    lams = rng.uniform(0.0, 2.0 * np.pi, size=n)
    arr = np.vstack([thetas, phis, lams]).T
    return {i: arr[i] for i in range(n)}


# Load / synthesize tables and cache
def _ensure_parameter_tables() -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    # Load clifford & validation parameter tables, or generate + cache them
    clifford_params: np.ndarray
    validation_param_dict: Dict[int, np.ndarray]

    # Try load; if missing, generate
    #   Try to load the Clifford table; if missing, generate and best‑effort save
    if cliff_file_path.exists():
        clifford_params = np.load(cliff_file_path)
    else:
        clifford_params = _generate_clifford_u3_params()
        try:
            np.save(cliff_file_path, clifford_params)
        except Exception:
            pass  # non-fatal if directory is not writable

    # Same for validation dict; returns both
    if val_file_path.exists():
        with open(val_file_path, "rb") as f:
            validation_param_dict = pickle.load(f)
    else:
        validation_param_dict = _generate_validation_params(100, seed=11)
        try:
            with open(val_file_path, "wb") as f:
                pickle.dump(validation_param_dict, f)
        except Exception:
            pass

    return clifford_params, validation_param_dict


# ---------------------------------------------------------------------
# Build the tables (load or synthesize)
# ---------------------------------------------------------------------

# Load or synthesize + cache
# Loads .npy/.pickle if present, else creates and writes them
clifford_params, validation_param_dict = _ensure_parameter_tables()

# Indices we use
v_params = [validation_param_dict[i] for i in range(100)]
# Build unitary/measurement tensors (vectorized/Choi form)
# TN_choi_vec(U) returns a vectorized representation; scaled by 2 then reshaped to (1,2,2) for consistent contractions
# Dicts map an integer index -> that tensor block
unique_cliffords = jnp.array([jnp.array(make_unitary(*C)) for C in clifford_params])
clifford_dict = {i: unique_cliffords[i] for i in range(24)}
clifford_param_dict = {i: clifford_params[i] for i in range(24)}

# Convert U3 to RZ-only decomposition (3 angles), make unitary matrices per RZ angle, 
# vectorize and reverse the order so time runs “measure -> … -> U0” in the contraction. Same for the validation set
clifford_rz_params = [u3_to_rz_params(clifford_param_dict[i]) for i in range(24)]
val_rz_params = [u3_to_rz_params(validation_param_dict[i]) for i in range(100)]
clifford_rz_unitaries = [[rz_unitary(P) for P in clifford] for clifford in clifford_rz_params]
val_rz_unitaries = [[rz_unitary(P) for P in val] for val in val_rz_params]
unique_v_unitaries = jnp.array([jnp.array(make_unitary(*C)) for C in v_params])

# Measurement vectors
# For Full-U view, 
#   pre-rotate computational basis measurement vectors by each unitary; 
#   store two lists (index 0/1 for the observed bit)
#   Pack them as (1,2,2) blocks with correct scaling 
#   Validation set is analogous.
# For RZ-decomp we’ll use pure_measurement: 
#   projectors onto |0⟩/|1⟩ (also packed (1,2,2)). 
clifford_meas_vec = jnp.array(
    [
        jnp.array([u.conj().T @ zero_vec for u in unique_cliffords]),
        jnp.array([u.conj().T @ one_vec for u in unique_cliffords]),
    ]
)
validation_meas_vec = jnp.array(
    [
        jnp.array([u.conj().T @ zero_vec for u in unique_v_unitaries]),
        jnp.array([u.conj().T @ one_vec for u in unique_v_unitaries]),
    ]
)

# Choi vector tensors
clifford_unitaries_vT = {
    i: 2 * TN_choi_vec(unique_cliffords[i]).conj().reshape(1, 2, 2) for i in range(24)
}
val_unitaries_vT = {
    i: 2 * TN_choi_vec(unique_v_unitaries[i]).conj().reshape(1, 2, 2)
    for i in range(100)
}

clifford_rz_unitaries_vT = {
    i: [2 * TN_choi_vec(RZ).conj().reshape(1, 2, 2) for RZ in reversed(clifford)]
    for i, clifford in enumerate(clifford_rz_unitaries)
}
val_rz_unitaries_vT = {
    i: [2 * TN_choi_vec(RZ).conj().reshape(1, 2, 2) for RZ in reversed(val)]
    for i, val in enumerate(val_rz_unitaries)
}

clifford_measurements_vT = {
    0: [
        jnp.kron(2 * zero_vec, clifford_meas_vec[0][i]).reshape(1, 2, 2)
        for i in range(24)
    ],
    1: [
        jnp.kron(2 * zero_vec, clifford_meas_vec[1][i]).reshape(1, 2, 2)
        for i in range(24)
    ],
}
val_measurements_vT = {
    0: [
        jnp.kron(2 * zero_vec, validation_meas_vec[0][i]).reshape(1, 2, 2)
        for i in range(100)
    ],
    1: [
        jnp.kron(2 * zero_vec, validation_meas_vec[1][i]).reshape(1, 2, 2)
        for i in range(100)
    ],
}

pure_measurement = {
    0: jnp.kron(2 * zero_vec, zero_vec).reshape(1, 2, 2),
    1: jnp.kron(2 * one_vec, one_vec).reshape(1, 2, 2),
}

# ---------------------------------------------------------------------
# Pipeline helper functions (unchanged logic, with one bug fix noted)
# ---------------------------------------------------------------------
# Counts -> probabilities (endianness fixed)
# Produces a long probability vector across all circuits and a parallel list of keys (strings) that were present. Reverse fixes qubit order conventions
def shadow_results_to_data_vec(results, shots, nQ):
    # Precompute the binary keys once to avoid repeated string allocations in the
    # inner loop, and cache the normalization factor.
    binary_keys = [np.binary_repr(i, nQ)[::-1] for i in range(2**nQ)]
    inv_shots = 1.0 / shots

    data_vec = []
    data_keys = []

    for res in results:
        tmp = []
        for key in binary_keys:
            p = res.get(key)
            if p is not None:
                data_vec.append(p * inv_shots)
                tmp.append(key)
        data_keys.append(tmp)
    return data_vec, data_keys


# Sequences -> operator arrays (Full-U)
# Builds a uniform 5-D array: 
#   (samples, qubit, time_slot, 2, 2), where time_slot = 1 (measurement) + nsteps unitaries
#   This is the “Full-U” representation
def shadow_seqs_to_op_array(sequences, keys, measurements, unitaries):
    nsteps = len(sequences[0][0]) - 1
    nQ = len(sequences[0])
    nUnique = sum([len(K) for K in keys])

    # Convert once to numpy arrays without the leading singleton dimension so we
    # can fill a preallocated buffer.
    meas_lookup = {i: [np.asarray(m)[0] for m in meas_list] for i, meas_list in measurements.items()}
    unitary_lookup = {i: np.asarray(u)[0] for i, u in unitaries.items()}
    dtype = np.asarray(next(iter(unitary_lookup.values()))).dtype

    out = np.empty((nUnique, nQ, nsteps + 1, 2, 2), dtype=dtype)

    sample_idx = 0
    for seq, seq_keys in zip(sequences, keys):
        for key in seq_keys:
            for q_idx, ops in enumerate(seq):
                out[sample_idx, q_idx, 0] = meas_lookup[int(key[q_idx])][ops[0]]
                for t_idx, gate_idx in enumerate(ops[1:], start=1):
                    out[sample_idx, q_idx, t_idx] = unitary_lookup[gate_idx]
            sample_idx += 1

    return jnp.array(out)


# Sequences -> operator arrays (RZ-decomp)
# Same idea, but three RZ blocks per time step (+1 measurement). This feeds the “X_decomp” likelihood path
'''
(For TN assembly): 
The two op_arrays_to_single_vector_TN_padded* helpers turn a per-sample operator array into a 2D tensor network (rows=time, cols=qubits). 
Not used directly in run.py, but used downstream in TN building. 
Note the corrected row_tag_id/col_tag_id usage for Quimb in the Full-U variant
'''
def shadow_seqs_to_op_array_rz(sequences, keys, measurements, unitaries):
    nsteps = len(sequences[0][0]) - 1
    nQ = len(sequences[0])
    nUnique = sum([len(K) for K in keys])

    meas_lookup = {i: np.asarray(m)[0] for i, m in measurements.items()}
    unitary_lookup = {i: [np.asarray(u)[0] for u in us] for i, us in unitaries.items()}
    sample_unitaries = next(iter(unitary_lookup.values()))
    dtype = np.asarray(sample_unitaries[0]).dtype

    out = np.empty((nUnique, nQ, 3 * (nsteps + 1) + 1, 2, 2), dtype=dtype)

    sample_idx = 0
    for seq, seq_keys in zip(sequences, keys):
        for key in seq_keys:
            for q_idx, ops in enumerate(seq):
                out[sample_idx, q_idx, 0] = meas_lookup[int(key[q_idx])]
                pos = 1
                for gate_idx in ops:
                    gates = unitary_lookup[gate_idx]
                    out[sample_idx, q_idx, pos : pos + 3] = gates
                    pos += 3
            sample_idx += 1

    return jnp.array(out)


def op_arrays_to_single_vector_TN_padded(op_seq):
    k = len(op_seq[0]) - 1
    nQ = op_seq.shape[0]
    # input in order: measure, U_{k-1}, ..., U_0

    TN_list = []

    for i in range(nQ):
        initial = qtn.Tensor(
            op_seq[i][0],
            inds=(f"kP_q{i}", f"ko{k}_q{i}"),
            tags=["U3", f"q{i}_U{k}", f"ROW{i}", f"COL{k}"],
        )
        for j, O in enumerate(op_seq[i][1:]):
            initial = initial & qtn.Tensor(
                O,
                inds=(f"ki{k-j}_q{i}", f"ko{k-j-1}_q{i}"),
                tags=["U3", f"q{i}_U{k-j-1}", f"ROW{i}", f"COL{k-j-1}"],
            )
        TN_list.append(initial)

    TN_list = qtn.TensorNetwork(TN_list)

    # Robustly construct a 2D grid regardless of quimb version:
    # newer quimb -> row_tag_id/col_tag_id ; older quimb -> y_tag_id/x_tag_id.
    common = dict(site_tag_id="q{}_U{}", Ly=k + 1, Lx=nQ)
    try:
        OTN_ket = qtn.tensor_2d.TensorNetwork2DFlat.from_TN(
            TN_list, row_tag_id="ROW{}", col_tag_id="COL{}", **common
        )
    except Exception:
        # FIX: fall back on any exception, not just TypeError, to support quimb versions that
        # raise ValueError when row/col keyword is unknown. This path uses the older API.
        OTN_ket = qtn.tensor_2d.TensorNetwork2DFlat.from_TN(
            TN_list, y_tag_id="ROW{}", x_tag_id="COL{}", **common
        )

    OTN_bra = OTN_ket.H.copy()
    OTN_bra.reindex_({f"ko{i}_q{j}": f"bo{i}_q{j}" for i in range(k + 1) for j in range(nQ)})
    OTN_bra.reindex_({f"ki{i}_q{j}": f"bi{i}_q{j}" for i in range(1, k + 1) for j in range(nQ)})

    OTN_ket.add_tag("OP KET")
    OTN_bra.add_tag("OP BRA")
    return OTN_ket & OTN_bra


def op_arrays_to_single_vector_TN_padded_X_decomp(op_seq):
    k = len(op_seq[0]) - 1
    tsteps = int(k / 3)
    nQ = op_seq.shape[0]
    # input in order | measure -- U_k-1 -- ... -- U_0 |

    TN_list = []

    for i in range(nQ):
        cStep = tsteps
        initial = qtn.Tensor(op_seq[i][0], inds=(f"kP_q{i}", f"ko{tsteps}_q{i}"), tags=["MEAS"])
        for j, O in enumerate(op_seq[i][1:]):
            if j % 3 == 0:
                initial = initial & qtn.Tensor(
                    O, inds=(f"ki{cStep}_q{i}", f"kXmo{cStep-1}_q{i}"), tags=["RZ"]
                )
            if j % 3 == 1:
                initial = initial & qtn.Tensor(
                    O, inds=(f"kXmi{cStep-1}_q{i}", f"kXpo{cStep-1}_q{i}"), tags=["RZ"]
                )
            if j % 3 == 2:
                cStep -= 1
                initial = initial & qtn.Tensor(
                    O, inds=(f"kXpi{cStep}_q{i}", f"ko{cStep}_q{i}"), tags=["RZ"]
                )

        TN_list.append(initial)

    OTN_ket = qtn.TensorNetwork(TN_list)
    OTN_bra = OTN_ket.H.copy()
    OTN_bra.reindex_({f"ko{i}_q{j}": f"bo{i}_q{j}" for i in range(k + 1) for j in range(nQ)})
    OTN_bra.reindex_({f"ki{i}_q{j}": f"bi{i}_q{j}" for i in range(1, k + 1) for j in range(nQ)})
    OTN_bra.reindex_({f"kXpo{i}_q{j}": f"bXpo{i}_q{j}" for i in range(k + 1) for j in range(nQ)})
    OTN_bra.reindex_({f"kXpi{i}_q{j}": f"bXpi{i}_q{j}" for i in range(k + 1) for j in range(nQ)})
    OTN_bra.reindex_({f"kXmo{i}_q{j}": f"bXmo{i}_q{j}" for i in range(k + 1) for j in range(nQ)})
    OTN_bra.reindex_({f"kXmi{i}_q{j}": f"bXmi{i}_q{j}" for i in range(k + 1) for j in range(nQ)})

    OTN_ket.add_tag("OP KET")
    OTN_bra.add_tag("OP BRA")
    return OTN_ket & OTN_bra

