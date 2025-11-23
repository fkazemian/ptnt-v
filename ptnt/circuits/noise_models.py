# ptnt/circuits/noise_models.py

from __future__ import annotations

from typing import Optional
import numpy as np
import scipy.linalg

# Aer 0.17+ lives under top-level `qiskit_aer`
from qiskit_aer import Aer, AerSimulator, noise      # Pulls the Aer provider, its simulator class, and the noise module from qiskit‑aer
from qiskit.quantum_info import Operator             # Operator is Qiskit’s linear‑algebra wrapper we’ll return for 2‑qubit unitaries


# Docstring explains accepted inputs (either a registered backend ID or a simulation method)
def make_backend(name: str):
    '''
    Return an Aer backend. Accepts either a backend name like
    'aer_simulator' / 'aer_simulator_statevector' or a simulator
    method like 'statevector', 'density_matrix', 'stabilizer', 'automatic'.

    We try provider lookup first, then fall back to AerSimulator(method=name).
    First tries Aer.get_backend(name); 
        if that fails, assumes name is a method (e.g., "statevector") and constructs an AerSimulator for it. 
    Default to AerSimulator() if name is empty or an auto alias.

    Example: 
    make_backend("aer_simulator")                  # registered backend
    make_backend("aer_simulator_statevector")      # registered backend
    make_backend("statevector")                    # AerSimulator(method="statevector")
    make_backend("density_matrix")                 # AerSimulator(method="density_matrix")
    make_backend("")                               # AerSimulator() defaults to automatic

    '''
    # Try provider lookup (registered backends such as 'aer_simulator')
    # it tries two routes:
    #   Provider lookup: 
    #       if name is the id of an installed Aer backend (e.g. "aer_simulator", "aer_simulator_statevector"), Aer.get_backend(name) returns it.
    #   Method fallback: 
    #       if lookup fails, the code assumes we gave a simulation method (like "statevector", "density_matrix", "stabilizer", "automatic") and builds an AerSimulator(method=name). 
    #       If name is empty or one of the typical auto aliases, it returns the default AerSimulator() (which chooses an “automatic” method).
    try:
        return Aer.get_backend(name)
    except Exception:
        # Treat the given name as a simulation method
        if not name or name in {"aer_simulator", "automatic", "AerSimulator"}:
            return AerSimulator()
        return AerSimulator(method=name)


# Function to build a device‑style noise model for Figure‑3 experiments: depolarizing + coherent X‑rotation on sx
# 1q = one‑qubit (single‑qubit) operation or channel. Here, the depolarizing error acts on one qubit

# Depolarizing channel (1‑qubit, probability p): E_dep​(ρ)=(1−p)ρ+p/3​(XρX+YρY+ZρZ):
#   with probability p, a random Pauli error X,Y,Z is applied; with 1−p, nothing happens
#   This models fully mixed, basis‑agnostic noise that “shrinks” the Bloch vector: In code: depolarizing_error(p1q, 1)

# Coherent Rx over‑rotation: we want to implement some gate sequence, but the hardware systematically over‑rotates around the X axis by a small angle ε
# Mathematically that extra unitary is U_coh​(ε)=e^(−iεX),
#   which is exactly what we build with expm(-i*rx_angle*X)
#   This is a deterministic (unitary) error—no randomness—so as a channel it can be represented by a single Kraus operator, {U}
#   That’s why we wrap it as kraus_error([errU])

# sx gate: Qiskit’s X^(1/2) - a π/2 rotation about the X axis (up to a global phase). 
#   Two sx in a row equal an X (again up to phase). 
#   Many superconducting devices natively implement sx, so attaching noise to sx is a realistic stress test. 
#   In our circuit templates we indeed use sx as the native pulse inside the local layers


# This helper will return 
#   a NoiseModel that composes (in order) the 1‑qubit depolarizing and 
#   the coherent unitary channel onto every sx on every qubit of any circuit we later simulate with Aer. 
# This reproduces the “coherent + stochastic” stress 


'''
Example:
backend = make_backend("aer_simulator")
nz = make_coherent_depol_noise_model(p1q=1e-3, rx_angle=0.02)
job = backend.run([circ1, circ2], shots=8192, noise_model=nz)
'''
def make_coherent_depol_noise_model(p1q: float, rx_angle: float) -> noise.NoiseModel:
    '''
    Build a NoiseModel that composes 1q depolarizing noise with a small
    coherent RX over-rotation, and attaches it to the 'sx' (sqrt-X) gate.

    Parameters:
    p1q : float
        Depolarizing probability for 1-qubit gates (0 <= p1q <= 1).
    rx_angle : float
        Extra rotation angle (radians) about X to model coherent error.
    '''
    nm = noise.NoiseModel()       # Instantiates a fresh NoiseModel

    # 1-qubit depolarizing error (helper still exists in Aer 0.17.x)
    dep_err = noise.errors.depolarizing_error(p1q, 1)      # Creates a single‑qubit depolarizing channel with probability p1q

    # Coherent error as a deterministic Kraus channel with a single unitary
    # Builds a coherent error: a unitary  e^(−i rx_angle X), wrapped as a single‑Kraus channel.
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    errU = scipy.linalg.expm(-1.0j * rx_angle * X)
    coh_err = noise.errors.kraus_error([errU])

    # Attach both errors to the 'sx' gate for all qubits.
    # Keeping them as two calls avoids any dependency on removed helpers
    # like `unitary_error` or version-specific import paths.
    # Adds both errors to the sx instruction across all qubits, then returns the combined model
    nm.add_all_qubit_quantum_error(dep_err, "sx")
    nm.add_all_qubit_quantum_error(coh_err, "sx")
    return nm


# Builds the environment–system interaction U=exp(−iH) with XYZ couplings

# This is an anisotropic Heisenberg‑type two‑qubit Hamiltonian between the environment ancilla and a system qubit:
#   H=−1/2​(r_xx​X⊗X+r_yy​Y⊗Y+r_zz​Z⊗Z)
# The real coefficients r_xx,r_yy,r_zz stand in for “coupling × interaction time”. 
# we then take the time‑evolution unitary U=exp⁡(−iH)
#   and append that as a two‑qubit unitary acting on wires [0,q+1] (env with system‑qubit q)
#   Physically, this seeds non‑Markovian correlations across time by letting information flow into—and back from—the environment between local control layers

'''
Example:
our templates apply this interaction after each local layer, to every system qubit, at every time step:

envU = create_env_IA(rxx=0.10, ryy=0.00, rzz=0.20)
circ.append(envU, [0, 1])   # couple env (0) with system qubit 1 at this time slice
'''
def create_env_IA(rxx: float, ryy: float, rzz: float) -> Operator:
    '''
    Return a 2-qubit environment unitary U = exp(-i H) as a qiskit Operator,
    where H = -1/2 * (rxx X⊗X + ryy Y⊗Y + rzz Z⊗Z)
    '''
    # Pauli matrices on one qubit
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    # Two‑qubit Hamiltonian from Pauli‑tensor terms, exponentiated to unitary; wrapped as a Qiskit Operator
    H = -0.5 * (rxx * np.kron(X, X) + ryy * np.kron(Y, Y) + rzz * np.kron(Z, Z))
    U = scipy.linalg.expm(-1.0j * H)
    return Operator(U)  # dims inferred as (4,)



# Generates a slowly varying per‑circuit angle series (quasi‑static control noise)

# give each circuit in a batch a quasi‑static coherent control offset (e.g., a shared err_X), with slow drifts across circuits in time.
# a random walk (cumulative sum of white noise) has a low‑frequency‑dominated spectrum S(ω)∝1/(ω^2)
#   That is not exactly 1/f, but it is a lightweight, long‑correlation proxy that produces slowly varying angles. 
#   Normalizing by its maximum magnitude then scaling to max_angle just keeps the offsets inside a practical bound

'''
Example:
angles = pink_noise_series(n=100, max_angle=0.04, random_state=7)
# For each circuit k in our batch, bind err_X = angles[k]


In our pink/control template (append_dd_rx_error), err_X is the shared coherent over‑rotation inserted after each sx at every time step within that circuit. 
This matches the experimental picture where low‑frequency drift is constant over a short circuit duration, but drifts between circuits.
'''

# Quasi‑static control noise: low‑frequency dominated noise that appears “constant” over the duration of one circuit but drifts between circuits; 
#   our pink_noise_series makes a simple proxy by integrating white noise (random walk) and amplitude‑capping the result. 
def pink_noise_series(n: int, max_angle: float = 0.25, random_state: Optional[int] = None) -> np.ndarray:
    '''
    Lightweight 1/f-like series for "coherent control noise" studies.
    Uses a cumulative sum of white noise (~1/f^2), normalized to max 1,
    then scaled by `max_angle`.
    '''
    # Produces a random walk (cumsum of white), normalizes by its max amplitude, and scales to max_angle
    rng = np.random.default_rng(random_state)
    white = rng.standard_normal(n)
    series = np.cumsum(white)
    peak = np.max(np.abs(series))
    if peak > 0:
        series = series / peak
    return max_angle * series

