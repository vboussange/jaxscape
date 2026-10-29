"""Free-energy behavior at the public distance API, using an independent oracle."""

from importlib.util import find_spec

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax import Array
from jax.experimental.sparse import BCOO
from jaxscape import FreeEnergyDistance, Graph, GridGraph
from jaxscape.solvers import AMJaxCGSolver, CholmodSolver, PyAMGSolver


EDGES = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
WEIGHTS = np.array([0.8, 1.4, 0.6, 1.8, 1.1])
RASTER = np.array([[0.8, 1.2], [1.4, 0.9]])
COEFFICIENTS = np.array([[0.1, -0.2, 0.3], [1.1, 0.4, -0.7], [0.5, 0.9, -0.3]])


def adjacency(weights: Array) -> Array:
    result = jnp.zeros((4, 4), dtype=weights.dtype)
    result = result.at[EDGES[:, 0], EDGES[:, 1]].set(weights)
    return result + result.T


def reference(matrix: Array, theta: float | Array) -> Array:
    """Dense reference walk/fundamental matrix, independent of the SPD operator."""
    present = matrix > 0
    costs = jnp.where(present, 1 / jnp.where(present, matrix, 1), 0)
    walk = matrix / matrix.sum(axis=1, keepdims=True)
    weighted_walk = walk * jnp.exp(-theta * costs)
    fundamental = jnp.linalg.inv(jnp.eye(matrix.shape[0]) - weighted_walk)
    hitting = fundamental / jnp.diag(fundamental)[None, :]
    potential = -jnp.log(hitting) / theta
    return (potential + potential.T) / 2


def graph(weights: Array) -> Graph:
    indices = jnp.asarray(np.concatenate([EDGES, EDGES[:, ::-1]]))
    return Graph(BCOO((jnp.tile(weights, 2), indices), shape=(4, 4)))


def make_grid(values):
    return GridGraph(values, fun=lambda x, y: (x + y) / 2)


def test_rejects_invalid_inverse_temperature():
    for theta in [0.0, -1.0, np.nan, np.inf, jnp.array([0.5, 0.6])]:
        with pytest.raises((ValueError, RuntimeError), match="theta"):
            FreeEnergyDistance(theta=theta)(graph(jnp.asarray(WEIGHTS)))


def test_joint_gradients_match_independent_autodiff_and_finite_differences():
    weights = jnp.asarray(WEIGHTS)
    theta = jnp.asarray(0.7)
    sources, targets = jnp.array([3, 0, 3]), jnp.array([1, 2])
    coefficients = jnp.array([[0.4, -1.2], [-0.7, 2.1], [0.3, 0.8]])

    def objective(weights, theta):
        values = FreeEnergyDistance(theta)(
            graph(weights), sources=sources, targets=targets
        )
        return jnp.sum(coefficients * values)

    def oracle(weights, theta):
        values = reference(adjacency(weights), theta)
        return jnp.sum(coefficients * values[sources[:, None], targets[None, :]])

    expected = jax.grad(oracle, argnums=(0, 1))(weights, theta)
    actual = jax.jit(jax.grad(objective, argnums=(0, 1)))(weights, theta)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-11)
    step = 1e-5
    finite_weights = np.array(
        [
            (
                objective(weights.at[i].add(step), theta)
                - objective(weights.at[i].add(-step), theta)
            )
            / (2 * step)
            for i in range(weights.size)
        ]
    )
    finite_theta = (
        objective(weights, theta + step) - objective(weights, theta - step)
    ) / (2 * step)
    np.testing.assert_allclose(actual[0], finite_weights, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual[1], finite_theta, rtol=1e-8, atol=1e-9)


def test_numerically_unresolved_extremes_fail_explicitly():
    weights = jnp.ones(5, dtype=jnp.float32)
    for theta in [1e-12, 1000.0]:
        with pytest.raises((ValueError, RuntimeError), match="resolv|finite"):
            FreeEnergyDistance(theta)(graph(weights)).block_until_ready()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("theta", [0.1, 0.7, 2.0])
@pytest.mark.parametrize(
    "edge_weights",
    [WEIGHTS, np.array([1.0, 1.0, 0.0, 1.0, 1.0])],
    ids=["unequal", "tied-four-cycle"],
)
def test_precision_metric_and_selections(dtype, theta, edge_weights):
    weights = jnp.asarray(edge_weights, dtype=dtype)
    distance = FreeEnergyDistance(theta)
    expected = np.asarray(reference(adjacency(weights.astype(jnp.float64)), theta))
    actual = distance(graph(weights))
    assert actual.dtype == dtype
    tolerance = 2e-5 if dtype == jnp.float32 else 1e-11
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(jnp.diag(actual), 0)
    np.testing.assert_allclose(actual, actual.T, atol=tolerance)
    assert jnp.all(actual + jnp.eye(4) > 0)
    for k in range(4):
        assert jnp.all(actual <= actual[:, k, None] + actual[None, k, :] + tolerance)
    nodes = jnp.array([3, 1, 3])
    np.testing.assert_allclose(
        distance(graph(weights), nodes=nodes),
        expected[nodes[:, None], nodes[None, :]],
        rtol=tolerance,
        atol=tolerance,
    )
    sources, targets = jnp.array([3, 0, 3]), jnp.array([2, 0])
    np.testing.assert_allclose(
        distance(graph(weights), sources=sources, targets=targets),
        expected[sources[:, None], targets[None, :]],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        distance(graph(weights), sources=sources),
        expected[sources],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        distance(graph(weights), targets=targets),
        expected[:, targets],
        rtol=tolerance,
        atol=tolerance,
    )


def test_grid_coordinates_batching_and_self_gradients():
    raster = jnp.asarray(RASTER)
    nodes = jnp.array([[1, 1], [0, 0], [1, 1]])
    distance = FreeEnergyDistance(0.5)
    expected = reference(make_grid(raster).get_adjacency_matrix().todense(), 0.5)
    np.testing.assert_allclose(
        distance(make_grid(raster), nodes=nodes),
        expected[jnp.array([3, 0, 3])[:, None], jnp.array([3, 0, 3])[None, :]],
        rtol=1e-11,
        atol=1e-12,
    )
    matrices = jax.jit(jax.vmap(lambda values: distance(make_grid(values))))(
        jnp.stack([raster, 1.2 * raster])
    )
    np.testing.assert_allclose(matrices[0], expected, rtol=1e-11, atol=1e-12)

    def self_loss(weights, theta):
        return FreeEnergyDistance(theta)(
            graph(weights), sources=jnp.array([2, 2]), targets=jnp.array([2])
        ).sum()

    gradient = jax.jit(jax.grad(self_loss, argnums=(0, 1)))(jnp.asarray(WEIGHTS), 0.5)
    np.testing.assert_array_equal(gradient[0], 0)
    assert gradient[1] == 0


def test_rejects_nonfinite_negative_or_asymmetric_conductances():
    for value in [-0.2, np.nan, np.inf]:
        with pytest.raises((ValueError, RuntimeError), match="conductance|finite"):
            FreeEnergyDistance(0.7)(
                graph(jnp.asarray(WEIGHTS).at[0].set(value))
            ).block_until_ready()
    matrix = adjacency(jnp.asarray(WEIGHTS)).at[0, 1].multiply(2)
    with pytest.raises((ValueError, RuntimeError), match="symmetric"):
        FreeEnergyDistance(0.7)(Graph(BCOO.fromdense(matrix))).block_until_ready()


SOLVERS = ["cg"]
if find_spec("pyamg") is not None:
    SOLVERS.append("pyamg")
if find_spec("cholespy") is not None:
    SOLVERS.append("cholespy")
if find_spec("amjax") is not None and find_spec("pyamg") is not None:
    SOLVERS.append("amjax")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("solver_name", SOLVERS)
def test_sparse_solvers_state_and_joint_gradients(solver_name, dtype):
    rtol, atol = (1e-6, 1e-7) if dtype == jnp.float32 else (1e-12, 1e-12)
    solver: lx.AbstractLinearSolver
    if solver_name == "cg":
        solver = lx.CG(rtol=rtol, atol=atol, max_steps=100)
    elif solver_name == "pyamg":
        solver = PyAMGSolver(rtol=rtol, atol=atol, maxiter=100)
    elif solver_name == "cholespy":
        solver = CholmodSolver()
    else:
        solver = AMJaxCGSolver(rtol=rtol, atol=atol, max_steps=100)
    tolerance = 3e-4 if dtype == jnp.float32 else 1e-10
    weights = jnp.asarray(WEIGHTS, dtype=dtype)
    theta_value = jnp.asarray(0.7, dtype=dtype)
    nodes = jnp.array([3, 1, 3])
    coefficients = jnp.asarray(COEFFICIENTS, dtype=dtype)
    distance = FreeEnergyDistance(theta_value, solver=solver)
    state = distance.init(graph(weights))
    initialized = distance(graph(weights), nodes=nodes, state=state)
    assert initialized.dtype == dtype
    np.testing.assert_allclose(
        initialized,
        distance(graph(weights), nodes=nodes),
        rtol=tolerance,
        atol=tolerance,
    )
    for updated_weights, theta in [
        (weights * 1.1, theta_value),
        (weights, jnp.asarray(0.9, dtype=dtype)),
    ]:
        updated = FreeEnergyDistance(theta, solver=solver)
        refreshed = updated.init(graph(updated_weights))
        expected = reference(
            adjacency(updated_weights.astype(jnp.float64)),
            theta.astype(jnp.float64),
        )[nodes[:, None], nodes[None, :]]
        np.testing.assert_allclose(
            updated(graph(updated_weights), nodes=nodes, state=refreshed),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    differentiable_state = (
        distance.init_preconditioner(graph(weights))
        if isinstance(solver, AMJaxCGSolver)
        else None
    )

    def objective(weights, theta):
        return jnp.sum(
            coefficients
            * FreeEnergyDistance(theta, solver=solver)(
                graph(weights), nodes=nodes, state=differentiable_state
            )
        )

    def oracle(weights, theta):
        return jnp.sum(
            coefficients.astype(jnp.float64)
            * reference(adjacency(weights), theta)[nodes[:, None], nodes[None, :]]
        )

    actual = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        weights, theta_value
    )
    expected = jax.value_and_grad(oracle, argnums=(0, 1))(
        weights.astype(jnp.float64), theta_value.astype(jnp.float64)
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=tolerance, atol=tolerance)
    for a, e in zip(actual[1], expected[1]):
        assert a.dtype == dtype
        np.testing.assert_allclose(a, e, rtol=tolerance, atol=tolerance)


def test_amjax_preconditioner_only_state_reuses_current_operator():
    pytest.importorskip("amjax")
    pytest.importorskip("pyamg")
    weights = jnp.asarray(WEIGHTS)
    solver = AMJaxCGSolver(rtol=1e-12, atol=1e-12, max_steps=100)
    state = FreeEnergyDistance(0.7, solver=solver).init_preconditioner(graph(weights))
    current_weights = weights * 1.15
    nodes = jnp.array([0, 3])
    actual = jax.jit(
        lambda weights, theta: FreeEnergyDistance(theta, solver=solver)(
            graph(weights), nodes=nodes, state=state
        )
    )(current_weights, jnp.array(0.9))
    expected = reference(adjacency(current_weights), 0.9)[
        nodes[:, None], nodes[None, :]
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_sparse_duplicates_and_padding_preserve_matrix_semantics_and_gradients():
    weights = jnp.asarray(WEIGHTS)
    indices = jnp.asarray(np.concatenate([EDGES, EDGES[:, ::-1]]))
    duplicate_indices = jnp.concatenate([indices, indices, jnp.array([[4, 4], [0, 0]])])

    def objective(weights):
        values = jnp.concatenate(
            [jnp.tile(weights / 3, 2), jnp.tile(2 * weights / 3, 2), jnp.zeros(2)]
        )
        return FreeEnergyDistance(0.7)(
            Graph(BCOO((values, duplicate_indices), shape=(4, 4))),
            nodes=jnp.array([3, 0]),
        ).sum()

    def oracle(weights):
        return reference(adjacency(weights), 0.7)[
            jnp.array([3, 0])[:, None], jnp.array([3, 0])[None, :]
        ].sum()

    actual = jax.jit(jax.value_and_grad(objective))(weights)
    expected = jax.value_and_grad(oracle)(weights)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-11)


def test_limits_converge_to_reciprocal_lcp_and_scaled_resistance():
    from scipy.sparse.csgraph import shortest_path

    weights = jnp.asarray(WEIGHTS)
    matrix = np.asarray(adjacency(weights))
    costs = np.zeros_like(matrix)
    np.divide(1, matrix, out=costs, where=matrix > 0)
    lcp = shortest_path(costs, directed=False)
    laplacian = np.diag(matrix.sum(axis=1)) - matrix
    inverse = np.linalg.pinv(laplacian)
    resistance = inverse.diagonal()[:, None] + inverse.diagonal()[None, :] - 2 * inverse
    low = [
        np.asarray(FreeEnergyDistance(theta)(graph(weights)))
        for theta in [0.1, 0.01, 0.001]
    ]
    high = [
        np.asarray(FreeEnergyDistance(theta)(graph(weights)))
        for theta in [1.0, 4.0, 10.0]
    ]
    low_errors = [np.linalg.norm(value - len(EDGES) * resistance) for value in low]
    high_errors = [np.linalg.norm(value - lcp) for value in high]
    assert low_errors[0] > low_errors[1] > low_errors[2]
    assert high_errors[0] > high_errors[1] > high_errors[2]
    assert low_errors[-1] / np.linalg.norm(len(EDGES) * resistance) < 0.01
    assert high_errors[-1] / np.linalg.norm(lcp) < 0.1


def test_solver_convergence_does_not_imply_resolved_query_entries():
    values = jnp.ones((1, 12), dtype=jnp.float32)
    grid = make_grid(values)
    solver = lx.CG(rtol=0.05, atol=0.0, max_steps=100)
    with pytest.raises((ValueError, RuntimeError), match="resolve"):
        FreeEnergyDistance(0.7, solver=solver)(
            grid, nodes=jnp.array([0, 11])
        ).block_until_ready()


def test_accurately_resolved_distant_entries_and_gradients_are_accepted():
    """A long path has tiny hitting probabilities despite accurate CG solves."""
    raster = jnp.linspace(0.8, 1.2, 40)[None, :]
    nodes = jnp.array([0, 20, 39])
    coefficients = jnp.asarray(COEFFICIENTS)
    solver = lx.CG(rtol=1e-10, atol=1e-90, max_steps=10_000, stabilise_every=None)

    def values(raster, theta):
        grid = make_grid(raster)
        return FreeEnergyDistance(theta, solver=solver)(grid, nodes=nodes)

    def oracle(raster, theta):
        matrix = jnp.diag((raster[0, :-1] + raster[0, 1:]) / 2, 1)
        return reference(matrix + matrix.T, theta)[nodes[:, None], nodes[None, :]]

    actual = jax.jit(values)(raster, 0.7)
    np.testing.assert_allclose(actual, oracle(raster, 0.7), rtol=1e-10, atol=1e-11)
    actual_gradient = jax.jit(
        jax.grad(lambda x, t: (coefficients * values(x, t)).sum(), argnums=(0, 1))
    )(raster, 0.7)
    expected_gradient = jax.grad(
        lambda x, t: (coefficients * oracle(x, t)).sum(), argnums=(0, 1)
    )(raster, 0.7)
    for actual, expected in zip(actual_gradient, expected_gradient):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_ordinary_cg_tolerances_still_reject_unresolved_distant_entries():
    raster = jnp.linspace(0.8, 1.2, 40)[None, :]
    grid = make_grid(raster)
    solver = lx.CG(rtol=1e-10, atol=1e-14, max_steps=10_000, stabilise_every=10)
    with pytest.raises((ValueError, RuntimeError), match="resolve"):
        FreeEnergyDistance(0.7, solver=solver)(
            grid, nodes=jnp.array([0, 20, 39])
        ).block_until_ready()


@pytest.mark.parametrize(
    "dtype,theta",
    [
        (jnp.float32, 0.1),
        (jnp.float32, 0.7),
        (jnp.float32, 2.0),
        (jnp.float64, 0.001),
        (jnp.float64, 0.7),
        (jnp.float64, 4.0),
    ],
)
def test_gradient_accuracy_across_accepted_parameter_range(dtype, theta):
    weights = jnp.asarray(WEIGHTS, dtype=dtype)
    coefficients = jnp.array(
        [
            [0.2, -0.3, 0.7, 0.9],
            [-1.2, 0.1, 0.2, -0.5],
            [0.5, -0.6, 0.3, -1.1],
            [0.4, 0.8, -0.2, 0.6],
        ],
        dtype=dtype,
    )

    def objective(weights, theta):
        return jnp.sum(coefficients * FreeEnergyDistance(theta)(graph(weights)))

    def oracle(weights, theta):
        return jnp.sum(coefficients * reference(adjacency(weights), theta))

    actual = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        weights, jnp.asarray(theta, dtype=dtype)
    )
    expected = jax.grad(oracle, argnums=(0, 1))(
        weights.astype(jnp.float64), jnp.asarray(theta, dtype=jnp.float64)
    )
    tolerance = 3e-4 if dtype == jnp.float32 else 3e-7
    for a, e in zip(actual, expected):
        assert jnp.all(jnp.isfinite(a))
        np.testing.assert_allclose(a, e, rtol=tolerance, atol=tolerance)


def test_theta_derivative_matches_variational_identity():
    weights = jnp.asarray(WEIGHTS)
    theta = 0.7
    matrix = adjacency(weights)
    present = matrix > 0
    costs = jnp.where(present, 1 / jnp.where(present, matrix, 1), 0)
    walk = matrix / matrix.sum(axis=1, keepdims=True)
    weighted = walk * jnp.exp(-theta * costs)
    fundamental = jnp.linalg.inv(jnp.eye(4) - weighted)
    total_cost = (fundamental @ (costs * weighted) @ fundamental) / fundamental
    expected_cost = total_cost - jnp.diag(total_cost)[None, :]
    expected_cost = (expected_cost + expected_cost.T) / 2
    expected = (expected_cost - reference(matrix, theta)) / theta
    actual = jax.jacrev(lambda theta: FreeEnergyDistance(theta)(graph(weights)))(
        jnp.asarray(theta)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    assert jnp.all(actual <= 1e-12)


def test_grid_rectangular_coordinates_and_batched_gradients():
    raster = jnp.asarray(RASTER)
    sources = jnp.array([[1, 1], [0, 0], [1, 1]])
    targets = jnp.array([[0, 1], [1, 0]])

    def objective(values):
        grid = make_grid(values)
        return FreeEnergyDistance(0.7)(grid, sources=sources, targets=targets).sum()

    def oracle(values):
        grid = make_grid(values)
        matrix = reference(grid.get_adjacency_matrix().todense(), 0.7)
        return matrix[jnp.array([3, 0, 3])[:, None], jnp.array([1, 2])[None, :]].sum()

    batch = jnp.stack([raster, raster * 1.1])
    actual = jax.jit(jax.vmap(jax.value_and_grad(objective)))(batch)
    expected = jax.vmap(jax.value_and_grad(oracle))(batch)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-11)


def test_removed_rsp_distance_import_names_replacement():
    with pytest.raises(ImportError, match="FreeEnergyDistance"):
        from jaxscape import RSPDistance  # noqa: F401
