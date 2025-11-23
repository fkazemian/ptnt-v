"""Misc utility functions (with faster Hellinger)."""

import numpy as np
import scipy

# Pauli matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.0j], [1.0j, 0]])
Z = np.array([[1, 0], [0, -1]])

sqrtX = scipy.linalg.sqrtm(X)
qubit_pauli_set = [I, X, Y, Z]

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
S = np.array([[1, 0], [0, np.exp(np.pi * 1.0j / 2)]])

# single qubit Choi basis
choi_bases = []
for i in range(2):
    tmp = []
    for j in range(2):
        v1 = np.eye(1, 2, i)
        v2 = np.eye(1, 2, j)
        temp_mat = np.dot(v1.T, v2)
        tmp.append(temp_mat)
    choi_bases.append(tmp)
choi_bases = np.array(choi_bases)

bell_vec = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=complex).reshape(2, 2)
bell_v = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1], dtype=complex)
plus = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)

bell_end = np.reshape(bell_v, [1, 2, 2, 1])
bell_mid = np.reshape(bell_v, [1, 2, 2, 1, 1])
plus_end = np.reshape(plus, [1, 2, 1])

zero_vec = np.array([1.0, 0.0], dtype=complex)
one_vec = np.array([0.0, 1.0], dtype=complex)

def TN_choi_mat(unitary):
    left_subsystem = np.einsum("ij,lmjk->lmik", unitary, choi_bases)
    left_subsystem = np.einsum("ijkl,lm", left_subsystem, unitary.conj().T)
    combined_choi = np.einsum("ijkl,ijmn->kmln", left_subsystem, choi_bases)
    combined_choi = np.reshape(combined_choi, (4, 4))
    return 0.5 * combined_choi

def TN_choi_vec(unitary):
    tmp_act = np.einsum("ij,jk->ik", unitary, bell_vec)
    return np.reshape(tmp_act, (4, 1))

def multi_kron(op_list):
    initial = 1
    for op in op_list:
        initial = np.kron(initial, op)
    return initial

def make_unitary(theta, phi, lamb):
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1.0j * lamb) * np.sin(theta / 2)],
            [np.exp(1.0j * phi) * np.sin(theta / 2), np.exp(1.0j * (lamb + phi)) * np.cos(theta / 2)],
        ]
    )

def make_rz_unitary(theta):
    return np.array([[1, 0], [0, np.exp(1.0j * theta)]])

def find_params(u):
    if np.abs(u[0, 0]) > 1e-6:
        u = (np.conj(u[0, 0]) / np.abs(u[0, 0])) * u
    theta = 2 * np.arccos(u[0, 0])
    if np.abs(theta) > 1e-6:
        temp = -u[0, 1] / (np.sin(theta / 2))
        lamb = np.angle(temp)
        temp = u[1, 0] / (np.sin(theta / 2))
        phi = np.angle(temp)
    else:
        phi = 0
        lamb = np.angle(u[1, 1])
    return list(np.real([theta, phi, lamb]))

def u3_to_rz_params(u3_params):
    rz1 = u3_params[2]
    rz2 = u3_params[0] + np.pi
    rz3 = u3_params[1] + 3 * np.pi
    return [rz1, rz2, rz3]

def rz_to_u3_params(rz_params):
    u3_0 = rz_params[1] - np.pi
    u3_1 = rz_params[2] - 3 * np.pi
    u3_2 = rz_params[0]
    return [u3_0, u3_1, u3_2]

def rz_unitary(theta):
    return np.array([[1, 0], [0, np.exp(1.0j * theta)]])

# --- Metrics ---
def state_fidelity(rho1, rho2):
    sqrho1 = scipy.linalg.sqrtm(rho1)
    fid = np.linalg.multi_dot([sqrho1, rho2, sqrho1])
    fid = np.trace(scipy.linalg.sqrtm(fid))
    fid = np.abs(fid) ** 2
    return fid

def hellinger_fidelity(dist_1, dist_2):
    """Fast classical Hellinger fidelity: (sum_i sqrt(p_i q_i))^2."""
    p = np.asarray(dist_1, float)
    q = np.asarray(dist_2, float)
    s = np.sqrt(np.maximum(p, 0.0)) @ np.sqrt(np.maximum(q, 0.0))
    return float(s * s)

def find_closest_density_matrix(matrix):
    matrix = matrix / matrix.trace()
    eigenValues, eigenVectors = np.linalg.eig(matrix)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    newEigenValues = np.zeros(len(eigenValues), dtype=np.complex_)
    i = len(eigenValues)
    a = 0
    while ((eigenValues[i - 1] + a / i) < 0) and (i > 0):
        newEigenValues[i - 1] = 0
        a += eigenValues[i - 1]
        i -= 1
    for j in range(i):
        newEigenValues[j] = eigenValues[j] + a / i
    states = []
    for j in range(len(eigenValues)):
        states.append(newEigenValues[j] * np.outer(eigenVectors[:, j], eigenVectors[:, j].conjugate()))
    densityMatrix = np.zeros((len(eigenValues), len(eigenValues)), dtype=np.complex_)
    for state in states:
        densityMatrix += state
    return densityMatrix

