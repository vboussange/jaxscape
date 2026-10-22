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
from jaxscape import Graph, GridGraph, ResistanceDistance
from jaxscape.solvers import (
    BCOOLinearOperator,
    JAXAMGCGSolver,
)
from jaxscape.solvers.operator import _batched_linear_solve, _linear_solve


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
def test_sparse_values_residuals_and_resistance_gradients(
    native_jaxamg, dtype, batched
):
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
        solve_impl = _batched_linear_solve if batched else _linear_solve

        def solve(data, rhs):
            return solve_impl(BCOO((data, indices), shape=(n, n)), rhs, solver)

        def dense_solve(data, rhs):
            matrix = BCOO((data, indices), shape=(n, n)).todense()
            return jnp.linalg.solve(matrix, rhs)

        compiled = jax.jit(solve)
        # Tie reciprocal entries to keep the undirected graph contract while
        # preserving duplicate storage and out-of-bounds capacity in gradients.
        edge_nodes = jnp.arange(n - 1, dtype=jnp.int32)
        edges = jnp.concatenate([
            jnp.stack([edge_nodes, edge_nodes + 1], axis=1),
            jnp.stack([edge_nodes, jnp.full_like(edge_nodes, n - 1)], axis=1),
        ])
        graph_indices = jnp.concatenate(
            [
                edges, edges[:, ::-1], edges, edges[:, ::-1],
                jnp.full((7, 2), n, dtype=jnp.int32),
            ]
        )
        coefficients = jnp.concatenate(
            [jnp.linspace(0.8, 1.3, edges.shape[0], dtype=dtype), jnp.zeros(7, dtype)]
        )
        distance = ResistanceDistance(solver=solver)
        nodes = jnp.array([0, 5, n - 1, 5], dtype=jnp.int32)
        query = {"nodes": nodes} if batched else {}

        def objective(coefficients, distance):
            weights = coefficients[:-7] / 2
            data = jnp.concatenate(
                [weights, weights, weights, weights, coefficients[-7:]]
            )
            graph = Graph(BCOO((data, graph_indices), shape=(n, n)))
            return jnp.sum(distance(graph, **query) ** 2)

        differentiated = jax.jit(
            jax.value_and_grad(lambda coefficients: objective(coefficients, distance))
        )
        reference_grad = jax.value_and_grad(
            lambda coefficients: objective(coefficients, ResistanceDistance())
        )
        tolerance = 5e-5 if dtype == jnp.float32 else 2e-9
        for scale in (1.0, 1.3):
            data = values * scale
            actual = compiled(data, rhs)
            expected = dense_solve(data, rhs)
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            residual = BCOO((data, indices), shape=(n, n)) @ actual - rhs
            np.testing.assert_allclose(residual, 0, atol=tolerance)
            value, gradient = differentiated(coefficients * scale)
            expected_value, expected_gradient = reference_grad(coefficients * scale)
            np.testing.assert_allclose(value, expected_value, rtol=tolerance)
            assert jnp.all(jnp.isfinite(gradient))
            np.testing.assert_allclose(
                gradient, expected_gradient, rtol=tolerance, atol=tolerance
            )
            np.testing.assert_array_equal(gradient[-7:], 0)


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
            _linear_solve(
                matrix,
                rhs,
                solver,
                options={
                    "preconditioner": lx.IdentityLinearOperator(operator.in_structure())
                },
            )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_compiled_inverse_objective_uses_current_landscape_and_optimizes(
    native_jaxamg, dtype
):
    """Fit changing land-cover permeabilities through sampled-site distances."""
    features = jax.nn.one_hot(
        jnp.array([[0, 0, 1, 1], [0, 2, 2, 1], [2, 2, 1, 0]]),
        num_classes=3,
        dtype=dtype,
    )
    nodes = jnp.array([0, 2, 5, 8, 11], dtype=jnp.int32)
    pairs = jnp.triu_indices(nodes.size, k=1)
    distance = ResistanceDistance(
        solver=JAXAMGCGSolver(
            rtol=1e-6 if dtype == jnp.float32 else 1e-11,
            atol=1e-7 if dtype == jnp.float32 else 1e-12,
            max_steps=200,
        )
    )
    reference = ResistanceDistance()  # Independent dense pseudoinverse path.

    def predict(params, nodes, metric):
        permeability = jnp.exp(features @ params)
        graph = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        # No matrix-bound state is retained across changing landscapes.
        return metric(graph, nodes=nodes)

    target_params = jnp.array([0.3, -0.35, 0.2], dtype=dtype)
    target = predict(target_params, nodes, reference)[pairs]

    def objective(params, nodes, metric):
        predicted = predict(params, nodes, metric)
        loss = jnp.mean((predicted[pairs] - target) ** 2)
        return loss, predicted

    evaluate = jax.jit(
        jax.value_and_grad(
            lambda params, nodes: objective(params, nodes, distance), has_aux=True
        )
    )
    reference_evaluate = jax.jit(
        jax.value_and_grad(
            lambda params, nodes: objective(params, nodes, reference), has_aux=True
        )
    )
    tolerance = 2e-4 if dtype == jnp.float32 else 2e-8
    initial_params = jnp.array([-0.2, 0.25, -0.1], dtype=dtype)
    for params in (initial_params, target_params / 2, initial_params):
        (loss, predicted), gradient = evaluate(params, nodes)
        (expected_loss, expected), expected_gradient = reference_evaluate(
            params, nodes
        )
        assert predicted.dtype == dtype
        assert gradient.dtype == dtype
        assert jnp.all(jnp.isfinite(gradient))
        np.testing.assert_allclose(predicted, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(loss, expected_loss, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=tolerance, atol=tolerance
        )

    @jax.jit
    def optimize(params, nodes):
        def step(params, _):
            (_, _), gradient = evaluate(params, nodes)
            return params - 0.5 * gradient, None

        return jax.lax.scan(step, params, xs=None, length=100)[0]

    fitted = optimize(initial_params, nodes)
    (initial_loss, _), _ = evaluate(initial_params, nodes)
    (final_loss, predicted), gradient = evaluate(fitted, nodes)
    assert jnp.all(jnp.isfinite(fitted))
    assert jnp.all(jnp.isfinite(gradient))
    assert final_loss < initial_loss * 0.01
    np.testing.assert_allclose(fitted, target_params, atol=0.05)
    np.testing.assert_allclose(predicted[pairs], target, rtol=0.02, atol=0.01)
