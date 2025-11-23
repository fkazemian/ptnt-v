# ptnt/tn/pepo.py
# Creation and manipulation of tensor network code for process tensors
# Build the model network (PEPO), X‑decomposition, and the LPDO wrapper

import jax.numpy as jnp
import numpy as np
import quimb.tensor as qtn
import scipy

from ptnt.utilities import *

# One “column” (one qubit) along time with Kraus legs
# time‑ordered strip; Kraus rank per slice driven by K_list; 
# labels use q_str (e.g., "q0"). The pos hint ('u', 'd', 'ud') tells how many extra legs to add so we can later connect strips vertically 
# in a staggered fashion (up/down)


def create_PT_MPO_guess_full_kraus(nsteps, bond_dim, K_list, q_str, pos):
    # creates one half of a PT MPO 'ring'
    # Each site is a rank 4 tensor (except \rho_0, which is 3): middle ones have two bond and two site
    # Final step has two site, one bond, one kraus (K1)
    # initial state has one site, one bond, one kraus (K0)
    # q_str will be an addition to each bond to label it for each qubit
    # pos will be 'u', 'd', or 'ud' to indicate the position of each qubit

    # Extra legs (one per character in pos) are pre‑declared: 
    # they’ll be the connectors used later to join neighbor‑strips. 
    # Their shapes start as 1 so the strip is initially independent. 
    # Labels like k u_t{t}_q0 or k d_t{t}_q0 get stamped at each time slice.
    extra_shape = tuple(1 for i in range(len(pos)))
    extra_label = tuple(f"k{P}" + "_t{}" + f"_{q_str}" for P in pos)

    # Final time slice tensor: 
    #   random complex entries with Kraus rank K_list[-1], 
    #   two site legs (the Choi “in/out” 2‑levels), 
    #   one temporal bond (capped by bond_dim and a heuristic min(4, bond_dim) bound), 
    #   plus any extra legs. The 0.999 just avoids exact +/- 1 values.
    final_shape = (K_list[-1], 2, 2, min(4, bond_dim)) + extra_shape
    final_step = 0.999 * (2 * np.random.rand(*final_shape) - 1) + 0.999 * (
        1.0j * (2 * np.random.rand(*final_shape) - 1)
    )

    # Wraps as a Quimb tensor with named indices:
    #   K{t} = Kraus leg; ki{t}/ko{t} = input/output site legs at time t; bond-* = temporal bond; extra legs get time‑stamped
    #   Tags: PT (this belongs to the Process Tensor), and ROW/ COL to make 2D‑grid assembly easy later
    final_step = qtn.Tensor(
        jnp.array(final_step),
        inds=(
            f"K{nsteps}_{q_str}",
            f"ko{nsteps}_{q_str}",
            f"ki{nsteps}_{q_str}",
            f"bond-{nsteps}_{q_str}",
        )
        + tuple(label.format(nsteps) for label in extra_label),
        tags=[f"{q_str}_I{nsteps}", "PT", f"ROW{q_str}", f"COL{nsteps}"],
    )

    # Initial slice tensor at time 0: has a left temporal bond, an output site leg, and Kraus leg K0. No ki0 because there’s no earlier input at t=0.
    initial_shape = (min(2, bond_dim), 2, K_list[0]) + extra_shape
    initial_state = 0.999 * (2 * np.random.rand(*initial_shape) - 1) + 0.999 * 1.0j * (
        2 * np.random.rand(*initial_shape) - 1
    )
    initial_state = qtn.Tensor(
        jnp.array(initial_state, dtype=complex),
        inds=(f"bond-0_{q_str}", f"ko0_{q_str}", f"K{0}_{q_str}")
        + tuple(label.format(0) for label in extra_label),
        tags=[f"{q_str}_I0", "PT", f"ROW{q_str}", f"COL{0}"],
    )

    # Heuristic bond‑dimension caps for internal slices: 
    # the powers of 4 reflect the worst‑case growth of Choi‑space DOFs to the left/right of each cut;
    #  *2 at boundaries accounts for a single site leg. Capping by bond_dim keeps the network tractable
    middle_sites = []

    for i in reversed(range(1, nsteps)):
        dim_left_L = 4 ** (nsteps - i)
        dim_right_L = 4 ** (i) * 2
        dim_left_R = 4 ** (nsteps - i + 1)
        dim_right_R = 4 ** (i - 1) * 2

        bond_size_L = min(dim_left_L, dim_right_L, bond_dim)
        bond_size_R = min(dim_left_R, dim_right_R, bond_dim)

        # Middle slice tensors: 
        #   5‑leg objects (left/right temporal bonds, two site legs, and Kraus leg), plus any extra connectivity legs
        tmp_shape = (bond_size_L, 2, 2, bond_size_R, K_list[i]) + extra_shape
        tmp_middle = 0.999 * (2 * np.random.rand(*tmp_shape) - 1) + 0.999 * (
            1.0j * (2 * np.random.rand(*tmp_shape) - 1)
        )
        tmp_middle = qtn.Tensor(
            jnp.array(tmp_middle),
            inds=(
                f"bond-{i}-L_{q_str}",
                f"ko{i}_{q_str}",
                f"ki{i}_{q_str}",
                f"bond-{i}-R_{q_str}",
                f"K{i}_{q_str}",
            )
            + tuple(label.format(i) for label in extra_label),
            tags=[f"{q_str}_I{i}", "PT", f"ROW{q_str}", f"COL{i}"],
        )

        middle_sites.append(tmp_middle)

    # Connect each time slice to the next along the temporal bonds (index positions 3 and 0 here), and return a 1D chain TN for this qubit
    total_tensors = [final_step] + middle_sites + [initial_state]

    for i in range(nsteps):
        qtn.tensor_core.connect(total_tensors[i], total_tensors[i + 1], 3, 0)

    return qtn.TensorNetwork(total_tensors)


# Stack single‑qubit strips into a 2D PEPO (qubits × time)
# Build one chain per qubit using the function above, then expand & connect vertical bonds to obtain a full 2D grid; wrap it as TensorNetwork2DFlat for convenient addressing.


def create_PT_PEPO_guess(nsteps, nqubits, bond_dims_t, bond_dims_q, K_lists):
    # First will create nqubits worth of just temporal PT MPOs
    # Each bond collection will be a list of lists (with |bond_dims_t| = nqubits and |bond_dims_q| = nqubits-1)
    # Will create these lines with their own function first and then expand and connect the bonds
    assert len(K_lists) == nqubits, "Mismatch of bond allocations!"  # Sanity check: one Kraus rank list per qubit

    # Build columns: top qubit configured with 'u', middle with 'ud', bottom with 'd' so the extra legs line up for vertical bonds. 
    # Temporal bond sizes per column can differ
    individual_PTs = [
        create_PT_MPO_guess_full_kraus(nsteps, bond_dims_t[0], K_lists[0], f"q{0}", "u")
    ]
    individual_PTs += [
        create_PT_MPO_guess_full_kraus(
            nsteps, bond_dims_t[i + 1], K_lists[i + 1], f"q{i+1}", "ud"
        )
        for i, K in enumerate(K_lists[1:-1])
    ]
    if nqubits > 1:
        individual_PTs += [
            create_PT_MPO_guess_full_kraus(
                nsteps, bond_dims_t[-1], K_lists[-1], f"q{nqubits-1}", "d"
            )
        ]

    # Internal detail: which index position is the vertical connector at each time slice (depends on whether it’s first/last/middle slice)
    connect_dict = {j: 5 for j in range(nsteps)}
    connect_dict[0] = 4
    connect_dict[nsteps] = 3

    # For each time slice i and vertical neighbor pair (j, j+1), grow the intended vertical index to bond_dims_q[i][j] and connect. This builds the 2D lattice
    for i in range(nsteps + 1):
        for j in range(nqubits - 1):
            connect_0 = connect_dict[i]
            connect_1 = connect_0 + 1
            if j == nqubits - 2:
                connect_1 = connect_0

            T1 = individual_PTs[j].tensors[i]
            T2 = individual_PTs[j + 1].tensors[i]

            # expand the matching indices to desired vertical bond dim
            T1.expand_ind(T1.inds[connect_0], bond_dims_q[i][j])
            T2.expand_ind(T2.inds[connect_1], bond_dims_q[i][j])

            # connect the two tensors along those positions
            qtn.tensor_core.connect(
                individual_PTs[j].tensors[i],
                individual_PTs[j + 1].tensors[i],
                connect_0,
                connect_1,
            )

    # Wrap as a flat 2D grid with Ly (rows=time), Lx (cols=qubits). The tag conventions match how we later index sites as [i, j] via tn[i, j]
    ptntN = qtn.TensorNetwork(individual_PTs)
    ptntN = qtn.tensor_2d.TensorNetwork2DFlat.from_TN(
        ptntN,
        site_tag_id="q{}_I{}",
        Ly=nsteps + 1,  # rows = time steps
        Lx=nqubits,  # cols = qubits (we use Lx as nQ later)
        y_tag_id="ROWq{}",  # was row_tag_id
        x_tag_id="COL{}",  # was col_tag_id
    )
    return ptntN

    return ptntN


# ------------ create_X_decomp, create_PEPO_X_decomp : Local “sqrt‑X + POVM” scaffolding
# These build the X‑decomposition side network used in the “RZ view” likelihood.



# Compute X^ (1/2) (matrix square‑root), vectorize to Choi form, conjugate to match bra/ket usage, and add a tiny random second “rank‑1” component (so the decomp isn’t singular)
# ptnt/tn/pepo.py  (drop-in, tag tester so it can be frozen; positivity on PT handled in loss)
import jax.numpy as jnp
import numpy as np
import quimb.tensor as qtn
import scipy

from ptnt.utilities import *

def create_X_decomp(nsteps, q_str, rand_strength=0.01):
    sqrtX = scipy.linalg.sqrtm(X)
    tmp_data = TN_choi_vec(sqrtX).reshape(2, 2).conj()
    tmp_data = jnp.array(
        [
            tmp_data,
            rand_strength * np.random.rand(2, 2)
            + rand_strength * 1.0j * np.random.rand(2, 2),
        ]
    )

    meas_data_0 = jnp.array(
        [
            np.array([1.0, 0.0]),
            rand_strength * np.random.rand(2)
            + 1.0j * rand_strength * np.random.rand(2),
        ]
    )
    meas_data_1 = jnp.array(
        [
            np.array([0.0, 1.0]),
            rand_strength * np.random.rand(2)
            + 1.0j * rand_strength * np.random.rand(2),
        ]
    )
    measure_data = jnp.array([meas_data_0, meas_data_1])

    X_TN_k = []
    for i in range(nsteps + 1):
        X_TN_k.append(
            qtn.Tensor(
                tmp_data,
                inds=(
                    f"KXp{i}_" + q_str,
                    f"kXpo{i}_" + q_str,
                    f"kXpi{i}_" + q_str,
                ),
                tags=["sqrtX", "Tester"],  # tag to allow freezing
            )
        )
        X_TN_k.append(
            qtn.Tensor(
                tmp_data,
                inds=(
                    f"KXm{i}_" + q_str,
                    f"kXmo{i}_" + q_str,
                    f"kXmi{i}_" + q_str,
                ),
                tags=["sqrtX", "Tester"],  
            )
        )

    X_TN_k.append(
        qtn.Tensor(
            measure_data,
            inds=(f"ko{nsteps+1}_" + q_str, "KM_" + q_str, f"ki{nsteps+1}_" + q_str),
            tags=[f"POVM_{q_str}", "Tester"],  
        )
    )
    return qtn.TensorNetwork(X_TN_k)


def create_PEPO_X_decomp(nsteps, nQ, rand_strength=0.01):
    individual_decomps = [
        create_X_decomp(nsteps, f"q{i}", rand_strength) for i in range(nQ)
    ]
    return qtn.TensorNetwork(individual_decomps)


def produce_LPDO(first_half):
    """Positive LPDO by doubling the supplied half."""
    bra_tn = first_half.copy().H
    bra_tn.reindex_(
        {
            ind: "b" + ind[1:]
            for ind in first_half.outer_inds()
            if ind[0] == "k" and ind[1] != "M"
        }
    )
    return first_half & bra_tn


# Inflate bond dimensions, seed data, add noise, optionally compress
# Replace site data: left boundary with |0⟩, internal with a “Bell‑mid” seed
# Expand Kraus legs, temporal bonds, and vertical bonds to the requested sizes (kraus_bond_list, horizontal_bond_list, vertical_bond_list)
# Add small complex noise to avoid symmetric saddles; squeeze_() removes trivial 1‑bonds
# This makes your initial PEPO a valid but learnable model


def expand_initial_guess_(
    guess,
    kraus_bond_list,
    horizontal_bond_list,
    vertical_bond_list,
    rand_strength=0.05,
    squeeze=True,
):
    nS = guess.Ly - 1
    nQ = guess.Lx
    # replace data
    for i in range(nQ):
        tmp_shape = guess[i, 0].shape
        tmp_zero = zero_vec.reshape(*tmp_shape)
        guess[i, 0].modify(data=tmp_zero)

    for i in range(nQ):
        for j in range(1, nS + 1):
            tmp_shape = guess[i, j].shape
            tmp_bell = bell_mid.reshape(*tmp_shape)
            guess[i, j].modify(data=tmp_bell)
    # expand Kraus indices
    for i in range(nQ):
        for j in range(nS + 1):
            T = guess[i, j]
            if "K{}_q{}".format(j, i) in T.inds:
                T.expand_ind("K{}_q{}".format(j, i), kraus_bond_list[i][j])

    for i in range(nQ):
        for j in range(nS + 1):
            T = guess[i, j]
            if j == 0:
                T.expand_ind(T.inds[0], horizontal_bond_list[i][j])
            if j > 0 and j < nS:
                T.expand_ind(T.inds[0], horizontal_bond_list[i][j])
                T.expand_ind(T.inds[3], horizontal_bond_list[i][j - 1])
            if j == nS:
                T.expand_ind(T.inds[3], horizontal_bond_list[i][j - 1])

    for i in range(nQ - 1):
        for j in range(nS + 1):
            T1 = guess[i, j]
            T2 = guess[i + 1, j]
            tmp_bonds = T1.filter_bonds(T2)[0][0]
            T1.expand_ind(tmp_bonds, vertical_bond_list[j][i])
            T2.expand_ind(tmp_bonds, vertical_bond_list[j][i])

    for T in guess.tensors:
        tmp = T.data.copy()
        tmp += rand_strength * (
            np.random.rand(*tmp.shape) + 1.0j * np.random.rand(*tmp.shape)
        )
        T.modify(data=tmp)
    if squeeze:
        guess.squeeze_()
    return None


def create_single_X(nsteps, q_str, rank=2, rand_strength=0.01):
    tmp_data = TN_choi_vec(sqrtX).reshape(2, 2).conj()
    X_data = [tmp_data]
    for i in range(rank - 1):
        X_data.append(
            rand_strength * np.random.rand(2, 2)
            + rand_strength * 1.0j * np.random.rand(2, 2)
        )

    X_data = jnp.array(X_data)

    meas_data_0 = jnp.array(
        [
            jnp.array(
                [
                    1.0,
                    rand_strength * np.random.rand()
                    + 1.0j * rand_strength * np.random.rand(),
                ]
            ),
            rand_strength * np.random.rand(2)
            + 1.0j * rand_strength * np.random.rand(2),
        ]
    )
    # meas_data_1 = np.array([np.array([0.0,1.0]), rand_strength * np.random.rand(2) + 1.j*rand_strength*np.random.rand(2)])
    meas_data_1 = jnp.array(
        [
            rand_strength * np.random.rand(2)
            + 1.0j * rand_strength * np.random.rand(2),
            jnp.array(
                [
                    rand_strength * np.random.rand()
                    + 1.0j * rand_strength * np.random.rand(),
                    1.0,
                ]
            ),
        ]
    )
    meas_data_2 = jnp.array(
        rand_strength * np.random.rand(2, 2)
        + 1.0j * rand_strength * np.random.rand(2, 2)
    )
    meas_data_3 = jnp.array(
        rand_strength * np.random.rand(2, 2)
        + 1.0j * rand_strength * np.random.rand(2, 2)
    )

    measure_data = jnp.array([meas_data_0, meas_data_1, meas_data_2, meas_data_3])

    X_TN = [
        qtn.Tensor(
            X_data,
            inds=(f"KXp{0}_" + q_str, f"kXpo{0}_" + q_str, f"kXpi{0}_" + q_str),
            tags=["sqrtX", "decomp"],
        )
    ]

    X_TN.append(
        qtn.Tensor(
            measure_data,
            inds=("KM_" + q_str, f"ko{nsteps+1}_" + q_str, f"ki{nsteps+1}_" + q_str),
            tags=["POVM_" + q_str, "decomp"],
        )
    )

    return qtn.TensorNetwork(X_TN)


def create_X_set(nsteps, nQ, rank=2, rand_strength=0.01):
    individual_decomps = [
        create_single_X(nsteps, f"q{i}", rank, rand_strength) for i in range(nQ)
    ]

    return qtn.TensorNetwork(individual_decomps)

