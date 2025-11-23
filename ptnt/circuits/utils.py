# ptnt/circuits/utils.py
from __future__ import annotations

from typing import Mapping, Sequence, Union
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter

# Declares the public API of this module (what from ... import * would expose)
__all__ = ["sanitize_basis", "bind_ordered"]


# Function docstring: remove non‑physical placeholders and deduplicate while preserving order.
'''
a tiny pre‑filter for the list of gate names you hand to Qiskit’s transpile
sometimes people put placeholders like "unitary" in basis_gates. Real backends don’t support a generic "unitary" instruction, so compilation can blow up or explode in size.

What it does:
    Drops None and "unitary"
    Deduplicates gate names but keeps the original order
    If nothing remains, returns None (which means “use the backend default basis”).
'''
def sanitize_basis(basis):
    # Return basis without 'unitary', preserving order and removing duplicates
    # If the result is empty, return None so callers can treat it as 'no override'.
    if basis is None:    # If there’s no override, return None (callers can pass that straight to transpile)
        return None
    # Iterates the requested basis, normalizes each name to lowercase, skips unitary (placeholder), 
    #   dedupes via the seen set, and returns either the cleaned list or None if it became empty
    seen = set()
    out = []
    for g in basis:
        if g is None:
            continue
        gl = str(g).lower()
        if gl == "unitary":
            continue
        if gl in seen:
            continue
        seen.add(gl)
        out.append(g)
    return out or None


# Type aliases for values accepted by bind_ordered: 
#   a sequence (positional binding) or a mapping (by parameter object or by name)

ParamVal = Union[float, int]
ValLike = Union[
    Sequence[ParamVal],                 # list/tuple/np.array in parameter order
    np.ndarray,
    Mapping[Union[Parameter, str], ParamVal],  # dict keyed by Parameter or name
]


# Helper to deterministically order parameters (name, then UUID), so mapping -> sequence conversion is stable
def _param_sort_key(p: Parameter):
    # Stable deterministic ordering for mapping sequences -> parameters
    name = getattr(p, "name", str(p))
    return (name, getattr(p, "uuid", 0))


# Public binder: assigns numbers to every Parameter in a circuit—either via a positional sequence or a mapping
'''
a safe way to fill in all Parameters in a circuit with numbers
It supports two ways to provide the values:
    a. Mapping form (dict) : by parameter name or by the Parameter object:
        circ.parameters  # e.g. {Parameter('t0_q0_x'), Parameter('t0_q0_y'), ...}
        bound = bind_ordered(circ, {
            "t0_q0_x": 0.12, "t0_q0_y": 1.0, "t0_q0_z": 0.0,
            # ... include every Parameter name ...
        })
        If you forget any, it raises a clear error listing missing names
    b. Sequence form (list/array) : same length as circ.parameters:
        vals = np.random.uniform(0, 2*np.pi, size=len(circ.parameters))
        bound = bind_ordered(circ, vals)
        Parameters are sorted deterministically (by name then UUID) so you get a stable order
        If the count doesn’t match, it raises a clear error

Return: a new circuit with all parameters assigned (the original isn’t modified)
In our pipeline, parameter names look like t{t}_q{q}_{x|y|z}. In the pink template, we’ll also see a single shared Parameter("err_X").

'''
def bind_ordered(template: QuantumCircuit, values: ValLike) -> QuantumCircuit:
    '''
    Bind numeric values to all Parameters in `template`.

    Accepts either:
      - a sequence/ndarray ordered to match the circuit's parameter order (deterministic), or
      - a dict mapping Parameter objects or parameter names -> numeric values.

    Raises a *clear* error listing the expected parameter names if anything is missing.
    '''
    params = sorted(list(template.parameters), key=_param_sort_key) # Collects all Parameters and sorts them deterministically (by _param_sort_key)

    # Mapping form: fill by Parameter object or by name
    # Mapping path: 
    #   supports keys as Parameter objects or their .name. If any parameters are unbound, raises a helpful error listing exactly which names are missing. 
    #   Returns a new circuit with parameters assigned (original left intact)

    if isinstance(values, Mapping):
        mapping = {}
        missing = []
        for p in params:
            if p in values:
                mapping[p] = float(values[p])
            elif p.name in values:
                mapping[p] = float(values[p.name])
            else:
                missing.append(p.name)
        if missing:
            raise ValueError(
                "Missing values for parameters: "
                + ", ".join(missing)
                + f". Template expects {len(params)} params: {[q.name for q in params]}"
            )
        return template.assign_parameters(mapping, inplace=False)

    # Sequence / ndarray form
    # Sequence path: 
    #   requires the number of provided values to match exactly the number of parameters. 
    #   Zips in order and returns a new assigned circuit.

    vals = np.asarray(values, dtype=float).ravel()
    if len(params) != vals.size:
        raise ValueError(
            f"Parameter count mismatch: circuit expects {len(params)} values "
            f"({[p.name for p in params]}), got {vals.size}."
        )
    mapping = dict(zip(params, vals.tolist()))
    return template.assign_parameters(mapping, inplace=False)

