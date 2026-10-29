"""Does swapping a rebuilt AMG hierarchy into a compiled training step recompile?

Two jit-boundary designs, both under eqx.filter_jit as in ilg-toolkit training:
  (1) ilg-toolkit today: AMJaxCGSolverState (closure-converted
      lx.FunctionLinearOperator) is passed as an argument.
  (2) the raw amjax.MultilevelSolver pytree is passed and the preconditioner
      operator/state is built inside the traced function.
Uses JAXScape ResistanceDistance + AMJaxCGSolver end to end, forward and
value_and_grad.
"""

from __future__ import annotations

import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import common as cm
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, AMJaxCGSolverState
from jaxscape.solvers.amjaxcgsolver import amjax_preconditioner_operator, build_amjax_solver

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
solver = AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=20000, coarse_solver="pinv")
rd = ResistanceDistance(solver=solver)
nodes = jnp.asarray(cm.terminal_nodes(n))
TR = {"state": 0, "ml": 0}


def loss_state(surface, state):
    TR["state"] += 1
    g = GridGraph(surface, fun=cm._mean_conductance)
    return rd.nodes_to_nodes_distance(g, nodes, state=state).sum()


def loss_ml(surface, ml):
    TR["ml"] += 1
    struct = jax.ShapeDtypeStruct((surface.size - 1,), surface.dtype)
    state = AMJaxCGSolverState(
        cg_state=None, preconditioner=amjax_preconditioner_operator(ml, struct)
    )
    g = GridGraph(surface, fun=cm._mean_conductance)
    return rd.nodes_to_nodes_distance(g, nodes, state=state).sum()


uni = jnp.ones((n, n), jnp.float64)
het = jnp.asarray(cm.make_surface("grf4", n))
L_uni = cm.reduced_laplacian(uni)
L_het = cm.reduced_laplacian(het)

results = []


def call(tag, fn, key, *args):
    before = TR[key]
    t = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    dt = time.perf_counter() - t
    rec = {"design": key, "call": tag, "time_s": dt, "retraced": TR[key] != before}
    results.append(rec)
    print(rec, flush=True)
    return out


for mode in ("fwd", "grad"):
    f_state = eqx.filter_jit(loss_state if mode == "fwd" else
                             eqx.filter_value_and_grad(loss_state))
    f_ml = eqx.filter_jit(loss_ml if mode == "fwd" else eqx.filter_value_and_grad(loss_ml))

    s_B1 = rd.init_preconditioner(GridGraph(uni, fun=cm._mean_conductance))
    s_B2 = rd.init_preconditioner(GridGraph(uni, fun=cm._mean_conductance))
    s_C = rd.init_preconditioner(GridGraph(het, fun=cm._mean_conductance))
    call(f"{mode}: state_B1 first", f_state, "state", het, s_B1)
    call(f"{mode}: state_B1 again", f_state, "state", het, s_B1)
    call(f"{mode}: state_B2 (identical values, rebuilt)", f_state, "state", het, s_B2)
    call(f"{mode}: state_C (rebuilt from current)", f_state, "state", het, s_C)

    ml_B1 = build_amjax_solver(L_uni, coarse_solver="pinv")
    ml_B2 = build_amjax_solver(L_uni, coarse_solver="pinv")
    ml_C = build_amjax_solver(L_het, coarse_solver="pinv")
    print("same leaf signature B vs C:", cm.same_compiled_signature(ml_B1, ml_C))
    call(f"{mode}: ml_B1 first", f_ml, "ml", het, ml_B1)
    call(f"{mode}: ml_B1 again", f_ml, "ml", het, ml_B1)
    call(f"{mode}: ml_B2 (identical values, rebuilt)", f_ml, "ml", het, ml_B2)
    call(f"{mode}: ml_C (rebuilt from current)", f_ml, "ml", het, ml_C)

with open(os.path.join(os.path.dirname(__file__), "results", f"recompile_{n}.json"), "w") as fh:
    json.dump(results, fh, indent=1)
