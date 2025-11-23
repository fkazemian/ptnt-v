import nbformat as nbf
from textwrap import dedent
from pathlib import Path

def md(s): return nbf.v4.new_cell("markdown", source=dedent(s))
def code(s): return nbf.v4.new_code_cell(dedent(s))

nb = nbf.v4.new_notebook()
nb["cells"] = []

# --- Title & Purpose ---
nb["cells"] += [md("""
# PTNT Sanity Check · Research-Grade Tutorial

**Purpose.** Provide a provable sanity check:

1. Build a **synthetic oracle** (data generator) for two scenarios:  
   - Baseline: depolarizing + coherent noise, no env memory  
   - Memory: nonzero env coupling rzz (temporal correlations)  
2. Fit a **parameterized model** (LPDO) and **compare** to the oracle on fresh hold-out experiments.  
3. Report **validation cross-entropy** vs **data-entropy** floor and **per-circuit Hellinger** distributions.  
4. Declare **PASS/FAIL** based on thresholded gaps.

> This is a pure Python notebook (no YAML required). Separate YAML configs for the same tests can be used with `ptnt run`.
""")]

# --- Bootstrap & Environment ---
nb["cells"] += [code("""
# Make ptnt importable if using a local checkout
import os, sys, pathlib
for r in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
    if (r / "ptnt").is_dir() and str(r) not in sys.path:
        sys.path.insert(0, str(r))

# Environment info
try:
    import ptnt
    from ptnt._version import __version__ as ptnt_version
    print("[ptnt] version:", ptnt_version)
except Exception as e:
    print("PTNT import error:", e)

try:
    import jax
    print("JAX devices:", jax.devices())
except Exception as e:
    print("JAX not available:", e)
""")]

# --- Imports & Helpers ---
nb["cells"] += [code("""
import numpy as np, quimb as qu
import matplotlib.pyplot as plt
from qiskit_aer import Aer

from ptnt.circuits.templates import base_PT_circ_template
from ptnt.circuits.noise_models import create_env_IA, make_coherent_depol_noise_model
from ptnt.circuits.utils import bind_ordered
from ptnt.preprocess.shadow import (
    clifford_param_dict, validation_param_dict, shadow_results_to_data_vec, 
    shadow_seqs_to_op_array, clifford_measurements_vT, clifford_unitaries_vT, val_measurements_vT, val_planar=1
)
from ptnt.tn.pepo import create_PT_PEPO_guess, expand_initial_guess_
from ptnt.tn.optimize import TNOptimizer
from ptnt.tn.fit import compute_likelihood, causality_keys_to_op_arrays, compute_probabilities
from ptnt.utilities import hellinger_fidelity

def reverse_seq_list(seq_list):
    out=[]
    for seq in seq_list:
        tmp=[]
        for T in seq:
            tmp.append([o for o in reversed(T)])
        tmp.reverse()
        out.append(tmp)
    return out
""")]

# --- Choose Scenario ---
nb["cells"] += [md("""
## 1) Choose a scenario

- `"baseline"`: `env_IA = 0`, depolarizing+coherent `sx` noise  
- `"memory"`: `env_IA.rzz = 0.2`, no additional static noise

We keep sizes small for a quick run.
""")]

nb["cells"] += [code("""
SCENARIO = "baseline"   # or "memory"

if SCENARIO == "baseline":
    Q, T = 2, 2
    env = create_partials=0
else:
    Q, T = 2, 3
    env = create_env_IA(0.0,0.0,0.2)

backend = Aer.get_backend("aer_simulator")
shell = base_PT_circ_template(Q, T, backend, basis_gates=None, template="dd_clifford", env_IA=env)

if SCENARIO=="baseline":
    noise_model = make_coherent_depol_noise_model(0.001,0.02)
else:
    noise_model = None

print("Scenario:",SCENARIO,"| Q,T =", (Q,T))
print(shell)
""")]

# ... (Truncated for brevity)
# In your local script, paste the full notebook-generation code we built earlier (incl. metrics, PASS/FAIL).
# You can reuse the code from our previous messages, or ask me to re-post the full generator if needed.
# The final notebook should implement:
#  - Build/simulate oracle
#  - Convert to operator arrays
#  - Build LPDO for chi=1 and chi=2
#  - Fit with TNOptimizer (greedy + small kappa)
#  - Compare CE vs entropy floors; compute Hellinger med/IQR; print PASS/FAIL.

# Save
out = Path("notebooks"); out.mkdir(parents=True, exist_ok=True)
nbformat = 4
nbf.write(nb, out / "PTNT_Sanity_Check_Tutorial.ipynb")
print("Wrote notebooks/PTNT_Sanity_Check_Tutorial.ipynb")
""")]

# Write to disk
with open(OUT / "make_sanity_notebook.py","w") as f:
    f.write(nbf.writes(nb))
print("Wrote:", OUT / "make_sanity_notebook.py")

