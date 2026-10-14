"""JAX-AMG availability and numerical behavior at public solver boundaries."""

import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import (
    batched_linear_solve,
    BCOOLinearOperator,
    JAXAMGCGSolver,
    linear_solve,
)


def test_missing_dependency_keeps_imports_working_and_reports_installation():
    code = """
import builtins
original_import = builtins.__import__
def without_jaxamg(name, *args, **kwargs):
    if name == "jaxamg" or name.startswith("jaxamg."):
        raise ImportError("JAX-AMG intentionally unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_jaxamg
import jaxscape
from jaxscape.solvers import JAXAMGCGSolver
try:
    JAXAMGCGSolver()
except ImportError as error:
    assert "jaxscape[jaxamg]" in str(error)
    assert "AmgX" in str(error)
else:
    raise AssertionError("Missing optional dependency must be reported")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["cpu", "rocm"])
def test_non_cuda_backend_is_rejected(monkeypatch, backend):
    # Replace only the external dependency and backend discovery boundary.
    monkeypatch.setitem(sys.modules, "jaxamg", types.ModuleType("jaxamg"))
    monkeypatch.setattr(
        jax, "default_backend", lambda: "gpu" if backend == "rocm" else "cpu"
    )
    monkeypatch.setattr(
        jax,
        "devices",
        lambda: [
            types.SimpleNamespace(
                client=types.SimpleNamespace(platform_version=backend)
            )
        ],
    )
    with pytest.raises(RuntimeError, match="NVIDIA.*CUDA"):
        JAXAMGCGSolver()


def test_lineax_contract_and_cg_controls(monkeypatch):
    monkeypatch.setitem(sys.modules, "jaxamg", types.ModuleType("jaxamg"))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        jax,
        "devices",
        lambda: [
            types.SimpleNamespace(client=types.SimpleNamespace(platform_version="CUDA"))
        ],
    )
    solver = JAXAMGCGSolver(rtol=1e-5, atol=1e-7, max_steps=200, stabilise_every=5)
    assert isinstance(solver, lx.AbstractLinearSolver)
    for kwargs in ({"rtol": -1}, {"atol": -1}, {"max_steps": 0}):
        with pytest.raises(ValueError):
            JAXAMGCGSolver(**kwargs)


@pytest.fixture
def native_jaxamg():
    pytest.importorskip("jaxamg")
    if jax.default_backend() != "gpu":
        pytest.skip("JAX-AMG numerical tests require an NVIDIA GPU and CUDA JAX")
    pytest.importorskip("jaxamg._amgx", reason="JAX-AMG native AmgX backend missing")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("shape", [(1, 2), (2, 3)])
def test_exact_resistance_eager_and_jit(native_jaxamg, dtype, shape):
    with jax.enable_x64():
        permeability = jnp.linspace(0.8, 1.5, np.prod(shape), dtype=dtype).reshape(
            shape
        )
        solver = JAXAMGCGSolver(
            rtol=1e-6 if dtype == jnp.float32 else 1e-11,
            atol=1e-7 if dtype == jnp.float32 else 1e-12,
            max_steps=200,
        )
        distance = ResistanceDistance(solver=solver)
        nodes = jnp.array([0, np.prod(shape) - 1], dtype=jnp.int32)

        def distances(values):
            graph = GridGraph(values, fun=lambda x, y: (x + y) / 2)
            return distance(graph, nodes=nodes)

        graph = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        expected = ResistanceDistance()(graph, nodes=nodes)
        tolerance = 2e-5 if dtype == jnp.float32 else 1e-9
        for actual in (distances(permeability), jax.jit(distances)(permeability)):
            assert actual.dtype == dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_sparse_values_residuals_and_implicit_gradients(native_jaxamg, dtype, batched):
    """Duplicates and excess capacity remain correct across compiled calls."""
    with jax.enable_x64():
        n = 17
        dense = (
            jnp.diag(jnp.linspace(3.0, 4.0, n, dtype=dtype))
            - jnp.eye(n, k=1, dtype=dtype)
            - jnp.eye(n, k=-1, dtype=dtype)
        )
        base = BCOO.fromdense(dense)
        indices = jnp.concatenate(
            [base.indices, base.indices, jnp.full((7, 2), n, dtype=jnp.int32)]
        )
        values = jnp.concatenate([base.data / 2, base.data / 2, jnp.zeros(7, dtype)])
        rhs = jnp.linspace(-0.5, 1.5, n, dtype=dtype)
        if batched:
            rhs = jnp.stack([rhs, jnp.ones_like(rhs)], axis=1)
        solver = JAXAMGCGSolver(
            rtol=1e-6 if dtype == jnp.float32 else 1e-11,
            atol=1e-7 if dtype == jnp.float32 else 1e-12,
            max_steps=300,
        )
        solve_impl = batched_linear_solve if batched else linear_solve

        def solve(data, rhs):
            return solve_impl(BCOO((data, indices), shape=(n, n)), rhs, solver)

        def dense_solve(data, rhs):
            matrix = BCOO((data, indices), shape=(n, n)).todense()
            return jnp.linalg.solve(matrix, rhs)

        compiled = jax.jit(solve)
        differentiated = jax.jit(
            jax.value_and_grad(lambda data, b: jnp.sum(solve(data, b) ** 2), (0, 1))
        )
        reference_grad = jax.value_and_grad(
            lambda data, b: jnp.sum(dense_solve(data, b) ** 2), (0, 1)
        )
        tolerance = 5e-5 if dtype == jnp.float32 else 2e-9
        for scale in (1.0, 1.3):
            data = values * scale
            actual = compiled(data, rhs)
            expected = dense_solve(data, rhs)
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            residual = BCOO((data, indices), shape=(n, n)) @ actual - rhs
            np.testing.assert_allclose(residual, 0, atol=tolerance)
            value, gradients = differentiated(data, rhs)
            expected_value, expected_gradients = reference_grad(data, rhs)
            np.testing.assert_allclose(value, expected_value, rtol=tolerance)
            for gradient, expected_gradient in zip(gradients, expected_gradients):
                assert jnp.all(jnp.isfinite(gradient))
                np.testing.assert_allclose(
                    gradient, expected_gradient, rtol=tolerance, atol=tolerance
                )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_direct_lineax_state_and_initial_guess(native_jaxamg, dtype):
    with jax.enable_x64():
        matrix = BCOO.fromdense(jnp.array([[4.0, -1.0], [-1.0, 3.0]], dtype))
        operator = BCOOLinearOperator(matrix)
        rhs = jnp.array([1.0, 2.0], dtype)
        expected = jnp.linalg.solve(matrix.todense(), rhs)
        solver = JAXAMGCGSolver(rtol=1e-6, atol=1e-7, max_steps=1)
        state = solver.init(operator, {})
        # An exact warm start converges even with a one-step iteration budget.
        solution = lx.linear_solve(
            operator, rhs, solver, state=state, options={"y0": expected}
        )
        np.testing.assert_allclose(solution.value, expected, rtol=1e-6, atol=1e-7)
        assert solution.result == lx.RESULTS.successful
        with pytest.raises(ValueError, match="own CG-compatible preconditioner"):
            linear_solve(
                matrix,
                rhs,
                solver,
                options={
                    "preconditioner": lx.IdentityLinearOperator(operator.in_structure())
                },
            )
