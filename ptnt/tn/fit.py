# ptnt/tn/fit.py
# Helper functions for performing tensor network estimation of non-Markovian processes
# Negative log‑likelihood (MLE), causality regularization, forward probabilities

from __future__ import annotations

import importlib
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import quimb.tensor as qtn

# robust contractor factory depends on cotengra when available
try:
    import cotengra as ctg
except Exception:  # pragma: no cover
    ctg = None

from ptnt.preprocess.shadow import (
    op_arrays_to_single_vector_TN_padded,
    op_arrays_to_single_vector_TN_padded_X_decomp,
)
from ptnt.tn.pepo import produce_LPDO
from ptnt.utilities import qubit_pauli_set


# ---------------------------------------------------------------------------------------
# Contractor factory (robust: avoid optuna version pitfalls unless explicitly enabled)
# ---------------------------------------------------------------------------------------
def make_contractor(preset: str | object = "auto-hq"):
    """
    Return an object you can pass to .contract(optimize=...).

    - If `preset` is already a contractor object, return it.
    - If `preset` is a string and cotengra is available, build a robust HyperOptimizer.
    - Else, return the string and let quimb choose.

    IMPORTANT: we deliberately avoid 'greedy-compressed' here, because
    compressed trees require `TensorNetwork.contract_compressed`, while
    this code consistently uses `.contract(...)`.
    """
    # Already a contractor-like object (HyperOptimizer etc.)
    if hasattr(preset, "search"):
        return preset

    # No cotengra -> just give quimb the string
    if ctg is None:
        return str(preset or "auto-hq")

    name = (preset or "auto-hq").lower().strip()

    # Let quimb pick simple built‑in schemes if the user explicitly asks
    if name in {"auto", "greedy", "random-greedy"}:
        return name

    if name in {"auto-hq", "hq"}:
        # **Key change**: only non‑compressed methods here
        methods = ["greedy", "random-greedy"]
        if importlib.util.find_spec("kahypar") is not None:
            methods.append("kahypar")
        if importlib.util.find_spec("pyflowcutter") is not None:
            methods.append("flowcutter")

        # Optuna is powerful but fragile across versions; default to 'random'.
        # Enable optuna only if the user explicitly asks for it:
        use_optuna = os.environ.get("PTNT_USE_OPTUNA", "0") == "1"
        optlib = (
            "optuna"
            if (use_optuna and importlib.util.find_spec("optuna") is not None)
            else "random"
        )

        try:
            # Keep things bounded and reproducible; don't pass any kwargs that older optuna can't handle.
            return ctg.HyperOptimizer(
                methods=methods,
                max_repeats=64,
                parallel=False,
                progbar=False,
                optlib=optlib,
                # Optuna-safe extras (only if actually using optuna):
                **({"direction": "minimize", "seed": 1} if optlib == "optuna" else {}),
            )
        except TypeError:
            # Fall back to the fully robust random search if the constructor chokes
            return ctg.HyperOptimizer(
                methods=methods,
                max_repeats=64,
                parallel=False,
                progbar=False,
                optlib="random",
            )

    # Unknown preset -> just return as string for quimb
    return name


# ---------------------------------------------------------------------------------------
# Pauli dictionaries and random causality keys
# ---------------------------------------------------------------------------------------

two_qubit_pauli_dict = {
    (i, j): jnp.kron(qubit_pauli_set[i], qubit_pauli_set[j]).reshape(2, 2, 2, 2)
    for i in range(4)
    for j in range(4)
}


def generate_random_causality_keys(nSteps: int, nQ: int, nSamples: int):
    """
    Generate random Pauli keys for the causality regulariser.

    Each key is a tuple of length (2*nSteps + 1), specifying which Pauli
    (I,X,Y,Z) to place on each input/output leg in the process tensor picture.
    """
    nqubits = 2 * nSteps + 1
    rand_keys = []
    for _ in range(nSamples):
        tmp = []
        for _n in range(nQ):
            system = 2 * np.random.randint(nSteps) + 1
            n_left_qubits = system
            n_right_qubits = nqubits - n_left_qubits - 1
            lk = [0 for _ in range(n_left_qubits)]
            sk = [np.random.randint(1, 4)]
            rk = list(np.random.randint(4, size=n_right_qubits))
            tmp.append(tuple(lk + sk + rk))
        rand_keys.append(tmp)

    return rand_keys


def causality_key_to_pauli_tn(cKey):
    """
    Turn a single causality key (one sample) into a Pauli‑operator TN with
    the same index structure as the process tensor PEPO.
    """
    k = int((len(cKey[0]) - 1) / 2)
    nQ = len(cKey)
    TN_list = []
    for i in range(nQ):
        initial_key = (0, cKey[i][0])
        initial = qtn.Tensor(
            two_qubit_pauli_dict[initial_key],
            inds=(f"kP_q{i}", f"ko{k}_q{i}", f"kP_q{i}", f"bo{k}_q{i}"),
            tags=["U3", f"q{i}_U{k}", f"ROW{i}", f"COL{k}"],
        )
        for j in range(k):
            current_key = cKey[i][2 * j + 1 : 2 * j + 3]
            initial = initial & qtn.Tensor(
                two_qubit_pauli_dict[current_key],
                inds=(
                    f"ki{k-j}_q{i}",
                    f"ko{k-j-1}_q{i}",
                    f"bi{k-j}_q{i}",
                    f"bo{k-j-1}_q{i}",
                ),
                tags=["U3", f"q{i}_U{k-j-1}", f"ROW{i}", f"COL{k-j-1}"],
            )
        TN_list.append(initial)
    return qtn.TensorNetwork(TN_list)


def causality_keys_to_op_arrays(cKeys):
    """
    Convert a list of causality keys into a JAX array of Pauli operators

        shape: (nSamples, nQ, nSteps+1, 2, 2, 2, 2)

    which can then be fed into `randomised_causality_regularisation`.
    """
    nsteps = int((len(cKeys[0][0]) - 1) / 2)
    nQ = len(cKeys[0])

    seq_of_seqs = []
    for key in cKeys:
        tmp_nQ_seq = []
        for j in range(nQ):
            tmp_seq = []
            initial_key = (0, key[j][0])
            tmp_seq.append(two_qubit_pauli_dict[initial_key])
            for k in range(nsteps):
                current_key = key[j][2 * k + 1 : 2 * k + 3]
                tmp_seq.append(two_qubit_pauli_dict[current_key])
            tmp_seq = np.concatenate(tmp_seq)
            tmp_nQ_seq.append(tmp_seq)
        tmp_nQ_seq = np.vstack(tmp_nQ_seq)
        seq_of_seqs.append(tmp_nQ_seq)
    final_shape = (len(cKeys), nQ, nsteps + 1, 2, 2, 2, 2)
    return jnp.array(np.vstack(seq_of_seqs).reshape(*final_shape))


def causality_ops_to_pauli_tn(op_seq):
    """
    Take a single sample of op_arrays (nQ, nSteps+1, 2,2,2,2) and turn it
    into a Pauli TN with the same index pattern as the process tensor.
    """
    k = len(op_seq[0]) - 1
    nQ = op_seq.shape[0]
    TN_list = []
    for i in range(nQ):
        initial = qtn.Tensor(
            op_seq[i][0],
            inds=(f"kP_q{i}", f"ko{k}_q{i}", f"kP_q{i}", f"bo{k}_q{i}"),
            tags=["U3", f"q{i}_U{k}", f"ROW{i}", f"COL{k}"],
        )
        for j, O in enumerate(op_seq[i][1:]):
            initial = initial & qtn.Tensor(
                O,
                inds=(
                    f"ki{k-j}_q{i}",
                    f"ko{k-j-1}_q{i}",
                    f"bi{k-j}_q{i}",
                    f"bo{k-j-1}_q{i}",
                ),
                tags=["U3", f"q{i}_U{k-j-1}", f"ROW{i}", f"COL{k-j-1}"],
            )
        TN_list.append(initial)
    return qtn.TensorNetwork(TN_list)


# ---------------------------------------------------------------------------------------
# Causality regularisation (deterministic & randomised / Monte‑Carlo)
# ---------------------------------------------------------------------------------------


def causality_term_k(mpo_traced, k, nQ):
    # Single‑time‑step causality term at layer k
    trace_op_i = qtn.TensorNetwork(
        [
            qtn.Tensor(
                0.5 * jnp.eye(2),
                inds=(f"bi{k}_q{j}", f"ki{k}_q{j}"),
                tags=[f"q{j}_I{k}"],
            )
            for j in range(nQ)
        ]
    )
    mpo_traced_double = ((2**nQ) * trace_op_i & mpo_traced).contract_tags(
        [f"q{j}_I{k}" for j in range(nQ)]
    )
    c1 = mpo_traced
    c2 = trace_op_i & mpo_traced_double
    v1 = jnp.real((c1 & c2.H).contract(optimize="greedy"))
    v2 = jnp.real((c1 & c1.H).contract(optimize="greedy"))

    return jnp.abs(1 - v1 / v2), mpo_traced_double


def causality_regularisation(mpo_half):
    """
    Deterministic version of the causality regulariser (sums over all
    time slices). Mostly useful for small toy systems / diagnostics.
    """
    ntimes = mpo_half.Ly
    nQ = mpo_half.Lx
    mpo_traced = produce_LPDO(mpo_half).copy()

    summed_causalities = []
    for k in reversed(range(1, ntimes)):
        trace_op_o = qtn.TensorNetwork(
            [
                qtn.Tensor(
                    jnp.eye(2),
                    inds=(f"bo{k}_q{j}", f"ko{k}_q{j}"),
                    tags=[f"q{j}_I{k+1}"],
                )
                for j in range(nQ)
            ]
        )
        tmp_tags = [
            f"q{j}_I{i}" for i in reversed(range(k, ntimes + 1)) for j in range(nQ)
        ]
        mpo_traced = (trace_op_o & mpo_traced).contract_tags(tmp_tags)
        c_term, mpo_traced = causality_term_k(mpo_traced, k, nQ)
        summed_causalities.append(c_term)
    return sum(summed_causalities)


def randomised_causality_regularisation(
    mpo_half,
    op_arrays,
    T_decomp: bool = False,
    opt="auto-hq",
):
    """
    Monte‑Carlo causal penalty: sample random Pauli strings (in op_arrays)
    and contract them with the current LPDO model.

    `opt` can be a string or a pre‑built contractor (HyperOptimizer).
    """
    if T_decomp:
        first_half = mpo_half.select("PT")
        second_half = mpo_half.select("Tester")
        mpo_model = produce_LPDO(first_half & second_half)
    else:
        mpo_model = produce_LPDO(mpo_half.select("PT"))

    def evaluate_single_expectation(op):
        pauli_tn = causality_ops_to_pauli_tn(op)
        return (mpo_model & pauli_tn).contract(optimize=opt)

    # vmap over samples; contractions themselves stay on CPU/GPU via quimb+cotengra
    p_list = jax.vmap(evaluate_single_expectation)(op_arrays)
    p_list = jnp.abs(jnp.real(p_list))

    return sum(p_list)


def trace_PT(mpo_half):
    """
    Evaluate the trace of the PT (needed for enforcing trace preservation
    in some scenarios).
    """
    trace_keys = [[tuple([0 for _ in range(7)]) for _ in range(2)]]
    trace_arrays = causality_keys_to_op_arrays(trace_keys)
    return randomised_causality_regularisation(mpo_half, trace_arrays)


def compare_POVMs_to_identity(mpo_half):
    """
    Diagnostic: compare the two‑qubit POVM marginals to identity.
    """
    POVM0 = mpo_half.select("POVM_q0")
    POVM1 = mpo_half.select("POVM_q1")

    POVM0b = POVM0.H
    POVM0b.reindex_({"ko4_q0": "bo4_q0", "ki4_q0": "bi4_q0"})
    POVM1b = POVM1.H
    POVM1b.reindex_({"ko4_q1": "bo4_q1", "ki4_q1": "bi4_q1"})

    povm0_q0 = (POVM0 & POVM0b).contract().data[:, 0, :, 0]
    povm1_q0 = (POVM0 & POVM0b).contract().data[:, 1, :, 1]

    povm0_q1 = (POVM1 & POVM1b).contract().data[:, 0, :, 0]
    povm1_q1 = (POVM1 & POVM1b).contract().data[:, 1, :, 1]

    return jnp.linalg.norm(jnp.eye(2) - (povm0_q0 + povm1_q0)) + jnp.linalg.norm(
        jnp.eye(2) - (povm0_q1 + povm1_q1)
    )


# ---------------------------------------------------------------------------------------
# Likelihood & forward probabilities
# ---------------------------------------------------------------------------------------


def compute_likelihood(
    mpo_half,
    sequence_list,
    data,
    kappa,
    cArrays,
    X_decomp: bool = False,
    T_decomp: bool = False,
    opt="auto-hq",
):
    """
    Negative log‑likelihood + causality penalty.

    `opt` is either a string ('auto-hq', 'greedy', ...) or a pre‑built
    contractor object created by `make_contractor`.
    """
    nD = len(data)
    data_sum = sum(data)
    mpo_model = produce_LPDO(mpo_half)

    if X_decomp:

        def evaluate_single_prob(sequence):
            sequence_TN = op_arrays_to_single_vector_TN_padded_X_decomp(sequence)
            return (mpo_model & sequence_TN).contract(optimize=opt)

    else:

        def evaluate_single_prob(sequence):
            sequence_TN = op_arrays_to_single_vector_TN_padded(sequence)
            return (mpo_model & sequence_TN).contract(optimize=opt)

    # vmap over all unique (shadow circuit, measurement) combos
    p_list = jax.vmap(evaluate_single_prob)(sequence_list)
    p_list = jnp.real(p_list)

    # normalise, avoid log(0)
    p_list = jnp.abs(p_list) + 1e-10
    p_list = p_list / sum(p_list)
    p_list = data_sum * p_list
    p_list = jnp.log(p_list)
    data = jnp.array(data)

    # data fit term
    c1 = -(1 / nD) * sum(data * p_list)

    # causal regulariser (Monte‑Carlo estimate) – share the same contractor
    c2 = kappa * randomised_causality_regularisation(
        mpo_half, cArrays, T_decomp=T_decomp, opt=opt
    )

    return c1 + c2


def compute_probabilities(
    mpo_half,
    op_seqs,
    X_decomp: bool = False,
    T_decomp: bool = False,
    opt="auto-hq",
):
    """
    Forward probabilities for a fixed set of operator sequences (no penalty).
    Used at validation time.
    """
    mpo_model = produce_LPDO(mpo_half)

    if X_decomp:
        p_list = [
            (op_arrays_to_single_vector_TN_padded_X_decomp(S) & mpo_model).contract(
                optimize=opt
            )
            for S in op_seqs
        ]
    else:
        p_list = [
            (op_arrays_to_single_vector_TN_padded(S) & mpo_model).contract(optimize=opt)
            for S in op_seqs
        ]
    p_list = jnp.array(p_list)
    p_list = 0.5 * jnp.real(p_list)
    return p_list


def randomly_check_causality(mpo_half, nSamples: int, opt="auto-hq"):
    """
    Quick diagnostic: randomly sample causality Pauli strings and evaluate
    their expectation w.r.t. the current PT model. Should be near zero.
    """
    mpo_model = produce_LPDO(mpo_half.select("PT"))
    nQ = mpo_half.Lx
    nSteps = mpo_half.Ly - 1
    cKeys = generate_random_causality_keys(nSteps, nQ, nSamples)

    expectation_list = [
        (causality_key_to_pauli_tn(C) & mpo_model).contract(optimize=opt) for C in cKeys
    ]

    return sum(jnp.abs(jnp.array(expectation_list)))

