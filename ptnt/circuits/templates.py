# ptnt/circuits/templates.py
from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit import transpile
from .utils import sanitize_basis

'''
building template for quantum circuits that run in time steps:

    Prepare an extra environment qubit (ancilla) on wire 0
    For each time step:
        a) do local moves on the real “system” qubits (wires 1..nQ),
        b) then let them bump into the environment (a 2-qubit unitary on [env, system])
    After the last step, measure the system qubits.

These templates create the right kind of data to learn time-correlated noise (process tensor stuff). 

Different templates = different experiments:
    dd_clifford: neutral baseline (random local gates each step)
    dd_rx_error: same, but inject a small shared X-rotation error around each √X pulse (models slow control drift per circuit)
    dd_spillage: same, but add CRX pulses around each √X (models “spillage”/leakage of control onto other lines, e.g., the environment)
'''

# ---------------------------------------------------------------------
# small internal helper: build the per-(qubit, timestep) U(x,y,z) params
# ---------------------------------------------------------------------
def _make_param_dict(nQ: int, n_steps: int):
    """
    Create U3-style parameter triplets for each (qubit, timestep),
    including the final local layer at t = n_steps:
        (t{t}_q{q}_x, t{t}_q{q}_y, t{t}_q{q}_z)
    The names and creation order are stable, so bind_ordered(...) is deterministic.
    """
    return {
        (q, t): [
            Parameter(f"t{t}_q{q}_x"),
            Parameter(f"t{t}_q{q}_y"),
            Parameter(f"t{t}_q{q}_z"),
        ]
        for q in range(nQ)
        for t in range(n_steps + 1)
    }


# Appends the Figure‑3‑style randomized‑compiling shell onto circ
'''
1) append_dd_clifford(...) : baseline shell (Figure 3 style)

per time step (for each system qubit j = 1..nQ):
    Apply a local single-qubit gate with three parameters (x,y,z). In code it’s Qiskit’s u(x,y,z)
    Couple env (0) to that system qubit with env_IA
    (Optional) Couple neighboring system qubits with crosstalk_IA
    At the very end: another local layer, then measure the system
this is a neutral, randomized scaffolding that excites the right temporal correlations to learn from.

Example:

q0: |env> -- Ry(pi/4) -- U(x0,y0,z0) --[env_IA with q1]-- U(x1,y1,z1) -- measure? (env not measured here)
q1:        (sys)       U(x0,y0,z0) --[env_IA with q0]-- U(x1,y1,z1) -- MEASURE
'''
def append_dd_clifford(
    circ: QuantumCircuit,
    nQ: int,
    n_steps: int,
    env_IA,
    crosstalk_IA=None,
) -> None:
    """Baseline shell: local U(x,y,z) layers interleaved with env/system couplings."""
    params = _make_param_dict(nQ, n_steps)

    # time-layered structure
    for t in range(n_steps):
        # local single-qubit layer on system qubits 1..nQ
        for q in range(nQ):
            circ.u(*params[(q, t)], q + 1)

        # environment-system interactions; optional nearest-neighbor crosstalk
        for q in range(nQ):
            circ.append(env_IA, [0, q + 1])
            if crosstalk_IA is not None and q < nQ - 1:
                circ.append(crosstalk_IA, [q + 1, q + 2])

    # final local layer (post-interaction)
    for q in range(nQ):
        circ.u(*params[(q, n_steps)], q + 1)

    # measure system qubits into classical bits (qubit j+1 -> classical bit j)
    for q in range(nQ):
        circ.measure(q + 1, q)


# Appends the control‑noise (“pink”) variant that injects a shared err_X between sx pulses
'''
Same skeleton, but each local layer is not a single u(x,y,z). Instead, for each system qubit you do:
Rz(z) – sx – Rx(err_X) – Rz(x + π) – sx – Rx(err_X) – Rz(y + 3π)

err_X is a single shared Parameter for the whole circuit.
we bind it to a small angle per circuit (e.g., from our pink_noise_series).
That models quasi-static control drift: within one circuit, the extra rotation is the same across all steps; across circuits, it drifts slowly.

Then we do the same env coupling as before. Final layer uses the same pattern, then we measure.
we are doing this to separate “the bath’s memory” from “the controller’s slow coherent drift”.

Tiny example (1 system qubit, 1 step):
q0: env
q1: Rz(z0) – sx – Rx(err_X) – Rz(x0+π) – sx – Rx(err_X) – Rz(y0+3π)
    -- [env_IA with q0] --
    Rz(z1) – sx – Rx(err_X) – Rz(x1+π) – sx – Rx(err_X) – Rz(y1+3π)
    MEASURE
'''
def append_dd_rx_error(
    circ: QuantumCircuit,
    nQ: int,
    n_steps: int,
    env_IA,
    error_param: Parameter,
    crosstalk_IA=None,
) -> None:
    """Pink/control-noise shell: insert a shared Rx(err_X) around each sx pulse."""
    params = _make_param_dict(nQ, n_steps)

    for t in range(n_steps):
        for q in range(nQ):
            p = params[(q, t)]
            circ.rz(p[2], q + 1); circ.sx(q + 1); circ.rx(error_param, q + 1)
            circ.rz(p[0] + np.pi, q + 1); circ.sx(q + 1); circ.rx(error_param, q + 1)
            circ.rz(p[1] + 3 * np.pi, q + 1)

        for q in range(nQ):
            circ.append(env_IA, [0, q + 1])
            if crosstalk_IA is not None and q < nQ - 1:
                circ.append(crosstalk_IA, [q + 1, q + 2])

    # final layer, same pink structure
    for q in range(nQ):
        p = params[(q, n_steps)]
        circ.rz(p[2], q + 1); circ.sx(q + 1); circ.rx(error_param, q + 1)
        circ.rz(p[0] + np.pi, q + 1); circ.sx(q + 1); circ.rx(error_param, q + 1)
        circ.rz(p[1] + 3 * np.pi, q + 1)

    for q in range(nQ):
        circ.measure(q + 1, q)


# Appends the spillage variant: CRX pulses around SX to emulate superprocess effects
'''
Same skeleton, but around every sx pulse we insert CRX gates to other qubits (usually the environment):
... – sx – (for each init_qubit: CRX(spillage_rot, control=system, target=init)) – ...

Barriers are added just so it’s visually/structurally clear in the transpiled graph.
this intentionally “leaks” the action of the system’s drive into neighbors (e.g., the environment). 
That’s a simple way to simulate superprocess/spillage effects (the control itself alters the environment going forward).

Tiny example (1 system qubit, 1 step, spillage onto env):
q0: env                       <--- target of CRX
q1: Rz(z0) – sx – CRX(q1→q0) – Rz(x0+π) – sx – CRX(q1→q0) – Rz(y0+3π)
    -- [env_IA with q0] --
    (final layer similarly)
    MEASURE
'''
def append_dd_spillage(
    circ: QuantumCircuit,
    nQ: int,
    n_steps: int,
    env_IA,
    spillage_rot: float,
    init_qubits,
    crosstalk_IA=None,
) -> None:
    """Spillage shell: insert CRX(control=system, target=init) around each sx pulse."""
    params = _make_param_dict(nQ, n_steps)

    def _insert_spillage(circ: QuantumCircuit, rot: float, control_qubit: int, neighbor_qubits):
        # purely visual barriers (optional) for DAG readability after transpile
        circ.barrier()
        for iq in neighbor_qubits:
            # Qiskit signature: crx(theta, control, target)
            circ.crx(rot, control_qubit, iq)
        circ.barrier()

    for t in range(n_steps):
        for q in range(nQ):
            p = params[(q, t)]
            circ.rz(p[2], q + 1); circ.sx(q + 1); _insert_spillage(circ, spillage_rot, q + 1, init_qubits)
            circ.rz(p[0] + np.pi, q + 1); circ.sx(q + 1); _insert_spillage(circ, spillage_rot, q + 1, init_qubits)
            circ.rz(p[1] + 3 * np.pi, q + 1)

        for q in range(nQ):
            circ.append(env_IA, [0, q + 1])
            if crosstalk_IA is not None and q < nQ - 1:
                circ.append(crosstalk_IA, [q + 1, q + 2])

    # final local layer, also with spillage around sx
    for q in range(nQ):
        p = params[(q, n_steps)]
        circ.rz(p[2], q + 1); circ.sx(q + 1); _insert_spillage(circ, spillage_rot, q + 1, init_qubits)
        circ.rz(p[0] + np.pi, q + 1); circ.sx(q + 1); _insert_spillage(circ, spillage_rot, q + 1, init_qubits)
        circ.rz(p[1] + 3 * np.pi, q + 1)

    for q in range(nQ):
        circ.measure(q + 1, q)


# Public factory that builds and transpiles one circuit from the requested template plus options

'''
This is the one function we actually call
    Makes a circuit with nQ+1 qubits, prepares the env qubit (0) with Ry(pi/4) (just puts it in a superposition to “be present”)
    Picks which builder to use (dd_clifford, dd_rx_error, or dd_spillage)
    For pink/control, if we didn’t pass an error_param, it creates one named "err_X"
    Transpiles the whole circuit to match the backend’s basis (and removes junk basis entries like "unitary")
    Returns the ready-to-simulate circuit.
'''
def base_PT_circ_template(
    n_qubits: int,
    n_steps: int,
    backend,
    basis_gates,
    template: str,
    env_IA,
    **kwargs,
):
    # Allocate a circuit with env ancilla at qubit 0 and n_qubits system qubits; 
    # prepare env ancilla in ∣+⟩ via Ry(π/4) 
    #   (note: for exact |+⟩ you’d use H; ry(π/2) also works—here π/4 is a design choice that produces a superposition).
    circ = QuantumCircuit(n_qubits + 1, n_qubits)
    circ.ry(np.pi / 4, 0)

    # Dispatches to the appropriate appender:
    #   dd_clifford (Figure 3 shell)
    #   dd_rx_error (pink/control) with a shared err_X parameter (creates it if not provided)
    #   dd_spillage with spillage_rot and init_qubits options
    #   Any other string raises a clear error.
    name = str(template).strip()

    if name == "dd_clifford":
        append_dd_clifford(circ, n_qubits, n_steps, env_IA, kwargs.get("crosstalk_IA"))

    elif name == "dd_rx_error":
        err = kwargs.get("error_param")
        if err is None:
            err = Parameter("err_X")
        append_dd_rx_error(circ, n_qubits, n_steps, env_IA, err, kwargs.get("crosstalk_IA"))

    elif name == "dd_spillage":
        append_dd_spillage(
            circ,
            n_qubits,
            n_steps,
            env_IA,
            kwargs.get("spillage_rot", 0.0),
            kwargs.get("init_qubits", [0]),
            kwargs.get("crosstalk_IA"),
        )
    else:
        raise ValueError(f"Unknown template: {template}")

    circ.global_phase = 0  # normalize global phase (useful for deterministic printing)

    # Transpiles the circuit for the given backend and basis (filtering out non‑physical placeholders like "unitary"), 
    #   at a light optimization level 1
    return transpile(
        circ,
        backend=backend,
        basis_gates=sanitize_basis(basis_gates),
        optimization_level=1,
    )

