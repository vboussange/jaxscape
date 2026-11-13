"""`AbstractSparseIterativeSolver`, with the Jacobi preconditioner of the solver
guide."""

from typing import final

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import (
    AbstractSparseIterativeSolver,
    BCOOLinearOperator,
    csr_matvec,
)


@final
class JacobiCGSolver(AbstractSparseIterativeSolver):
    rtol: float = 1e-5
    max_steps: int = 1_000

    def preconditioner(self, pattern, values):
        return None

    def precondition(self, preconditioner, pattern, values):
        on_diagonal = jnp.where(pattern.rows == pattern.columns, values, 0)
        diagonal = jnp.zeros(pattern.n, values.dtype).at[pattern.rows].add(on_diagonal)
        return lambda residuals: residuals / diagonal[:, None]

    def matvec(self, preconditioner, pattern, values):
        return csr_matvec(pattern, values)

    def vector_shape(self, preconditioner, pattern):
        return (pattern.n,)


def grid_of(conductance):
    return GridGraph(conductance, fun=lambda x, y: (x + y) / 2)


def conductance():
    rng = np.random.default_rng(0)
    return jnp.asarray(10 ** rng.uniform(-2, 0, (9, 11)))


def test_a_backend_needs_only_a_preconditioner():
    distance = ResistanceDistance(solver=JacobiCGSolver(rtol=1e-12))
    state = distance.init(grid_of(jnp.ones((9, 11))))
    nodes = jnp.array([0, 50, 98])

    def objective(conductance, distance, state=None):
        return jnp.sum(distance(grid_of(conductance), nodes=nodes, state=state) ** 2)

    actual = jax.jit(jax.value_and_grad(lambda c: objective(c, distance, state)))(
        conductance()
    )
    expected = jax.value_and_grad(objective)(conductance(), ResistanceDistance(None))
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-8)


def test_an_unconverged_solve_fails_with_nan():
    solver = JacobiCGSolver(max_steps=2)
    matrix = ResistanceDistance().system_matrix(grid_of(conductance()))
    operator = BCOOLinearOperator(matrix)
    state = solver.init(operator, {})
    b = jnp.ones(matrix.shape[0])

    solution = lx.linear_solve(operator, b, solver, state=state, throw=False)

    assert solution.result == lx.RESULTS.max_steps_reached
    assert jnp.all(jnp.isnan(solution.value))


@pytest.mark.parametrize("settings", [{"rtol": -1.0}, {"max_steps": 0}])
def test_invalid_settings_raise(settings):
    with pytest.raises(ValueError):
        JacobiCGSolver(**settings)
