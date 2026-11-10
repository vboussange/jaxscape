from importlib.util import find_spec

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import networkx as nx
import numpy as np
import pytest
from equinox import filter_grad, filter_jit
from jax.experimental.sparse import BCOO
from jaxscape import (
    Graph,
    GridGraph,
    LCPDistance,
    resistance_distance,
    SpielmanApproximation,
)
from jaxscape.resistance_distance import (
    ExactResistance,
    p_inv_resistance_distance,
    ResistanceDistance,
)
from jaxscape.solvers import (
    AMJaxCGSolver,
    CholmodSolver,
    CuDSSSolver,
    PyAMGSolver,
)


# Check availability of optional solvers
PYAMG_AVAILABLE = find_spec("pyamg") is not None
CHOLMOD_AVAILABLE = find_spec("cholespy") is not None
AMJAX_AVAILABLE = PYAMG_AVAILABLE and find_spec("amjax") is not None

# Build list of available solvers
available_solvers = []
if PYAMG_AVAILABLE:
    available_solvers.append(PyAMGSolver())
if CHOLMOD_AVAILABLE:
    available_solvers.append(CholmodSolver())
if AMJAX_AVAILABLE:
    available_solvers.append(AMJaxCGSolver(rtol=1e-5, atol=1e-5, max_steps=500))

# Spielman projections are randomized; this is the expected absolute error for
# the deterministic seed and epsilon used in the test below.
EXPECTED_APPROXIMATION_ERROR = 8e-2
EXPECTED_DIAGONAL_ERROR = 5e-4
EXPECTED_GRADIENT_ERROR = 5e-3


def avg(x, y):
    return (x + y) / 2


def build_nx_resistance_distance_matrix(G):
    Rnx_dict = nx.resistance_distance(G, weight="weight", invert_weight=False)
    Rnx = jnp.zeros((G.number_of_nodes(), G.number_of_nodes()))
    node_list = list(G)
    for n, rd in Rnx_dict.items():
        i = node_list.index(n)
        for m, r in rd.items():
            j = node_list.index(m)
            Rnx = Rnx.at[i, j].set(r)
    return Rnx


def test_p_inv_resistance_distance():
    G = nx.grid_2d_graph(2, 3)
    # for u, v in G.edges():
    #     G[u][v]['weight'] = 1

    # simple graph
    A = nx.adjacency_matrix(G)
    Ajx = BCOO.from_scipy_sparse(A)
    Rjaxscape = p_inv_resistance_distance(Ajx)
    Rnx = build_nx_resistance_distance_matrix(G)
    assert jnp.allclose(Rjaxscape, Rnx)

    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.uniform(1, 10)  # Random weight between 1 and 10

    A = nx.adjacency_matrix(G)
    Ajx = BCOO.from_scipy_sparse(A)
    Rjaxscape = p_inv_resistance_distance(Ajx)
    Rnx = build_nx_resistance_distance_matrix(G)
    assert jnp.allclose(Rjaxscape, Rnx)


@pytest.mark.skipif(len(available_solvers) == 0, reason="No solvers available")
@pytest.mark.parametrize("solver", available_solvers)
def test_lineax_solver_resistance_distance(solver):
    """
    Tests that the lineax solver implementation of resistance distance
    produces the same result as the pseudo-inverse method.
    """
    key = jr.PRNGKey(42)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.1  # avoid zero permeability
    grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)

    # nodes to nodes
    dist_pinv = ResistanceDistance(solver=None)(grid)
    distance = ResistanceDistance(solver=solver)
    state = distance.init(grid)
    dist_lineax = distance(grid, state=state)
    assert jnp.allclose(dist_pinv, dist_lineax, rtol=1e-4)


@pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="Cholmod not available")
def test_cholmod_nodes_to_nodes_distance_differentiability_without_explicit_state():
    nodes = jnp.array([0, 1, 3, 4], dtype=jnp.int32)
    distance = ResistanceDistance(solver=CholmodSolver())

    def objective(permeability):
        graph = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(distance(graph, nodes=nodes))

    permeability = jnp.ones((3, 3), dtype=jnp.float32)
    gradient = jax.grad(objective)(permeability)

    assert isinstance(gradient, jax.Array)
    assert gradient.shape == permeability.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_approximate_resistance_distance():
    """
    Tests that the Spielman approximation is jittable and close to the
    pseudo-inverse method.
    """
    key = jr.PRNGKey(0)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.5
    grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)

    distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.05))
    dist_approx = filter_jit(distance)(grid)
    dist_pinv = ResistanceDistance(solver=None)(grid)

    assert dist_approx.shape == dist_pinv.shape
    assert jnp.allclose(dist_approx, dist_approx.T, atol=1e-5)
    assert jnp.allclose(jnp.diag(dist_approx), 0, atol=EXPECTED_DIAGONAL_ERROR)
    assert jnp.allclose(dist_approx, dist_pinv, atol=EXPECTED_APPROXIMATION_ERROR)

    nodes = jnp.array([0, 2])
    sources = jnp.array([0, 1])
    targets = jnp.array([2, 3])
    assert jnp.allclose(
        filter_jit(distance)(grid, nodes=nodes),
        dist_approx[nodes[:, None], nodes[None, :]],
        atol=1e-3,
    )
    assert jnp.allclose(
        filter_jit(distance)(grid, sources=sources, targets=targets),
        dist_approx[sources[:, None], targets[None, :]],
        atol=1e-3,
    )


def test_approximate_resistance_distance_differentiability():
    """
    Tests that the custom VJP keeps the approximate method compatible with
    jax.grad.
    """
    key = jr.PRNGKey(0)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.5
    distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.1))

    def objective(permeability_raster):
        grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(distance(grid))

    gradient = filter_jit(filter_grad(objective))(permeability_raster)

    step = 1e-2
    perturbation = jnp.zeros_like(permeability_raster).at[0, 0].set(step)
    finite_difference = (
        objective(permeability_raster + perturbation)
        - objective(permeability_raster - perturbation)
    ) / (2 * step)

    assert isinstance(gradient, jax.Array)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient[0, 0], finite_difference, atol=EXPECTED_GRADIENT_ERROR)


def test_approximate_nodes_to_nodes_resistance_distance_differentiability():
    """
    Tests that the selected-node Spielman path remains compatible with jax.grad.
    """
    key = jr.PRNGKey(0)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.5
    nodes = jnp.array([0, 2])
    distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.1))

    def objective(permeability_raster):
        grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(distance(grid, nodes=nodes))

    gradient = filter_jit(filter_grad(objective))(permeability_raster)

    assert isinstance(gradient, jax.Array)
    assert gradient.shape == permeability_raster.shape
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="Cholmod not available")
def test_approximate_cholmod_float64_nodes_to_nodes_distance():
    key = jr.PRNGKey(0)
    nodes = jnp.array([0, 2], dtype=jnp.int32)
    distance = ResistanceDistance(
        solver=CholmodSolver(),
        method=SpielmanApproximation(epsilon=0.05),
    )

    with jax.enable_x64():
        permeability_raster = jr.uniform(key, (2, 2), dtype=jnp.float64) + 0.5
        grid = GridGraph(permeability_raster, fun=lambda x, y: (x + y) / 2)
        approx = filter_jit(distance)(grid, nodes=nodes)

    assert approx.dtype == jnp.float64
    assert approx.shape == (nodes.shape[0], nodes.shape[0])
    assert jnp.all(jnp.isfinite(approx))


@pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="Cholmod not available")
def test_approximate_cholmod_float64_nodes_to_nodes_distance_large_graph_auto_state():
    distance = ResistanceDistance(
        solver=CholmodSolver(),
        method=SpielmanApproximation(epsilon=0.05),
    )
    coords = jnp.stack(
        [jnp.arange(20, dtype=jnp.int32) // 10, jnp.arange(20, dtype=jnp.int32) % 10],
        axis=1,
    )

    with jax.enable_x64():
        key = jr.PRNGKey(0)
        permeability_raster = jr.uniform(key, (10, 10), dtype=jnp.float64) + 0.5
        grid = GridGraph(permeability_raster, fun=lambda x, y: (x + y) / 2)
        approx = filter_jit(distance)(grid, nodes=coords)

    assert approx.dtype == jnp.float64
    assert approx.shape == (coords.shape[0], coords.shape[0])
    assert jnp.all(jnp.isfinite(approx))


def solver_tolerances(solver):
    if isinstance(solver, CholmodSolver):
        return {
            "solve_rtol": 1e-5,
            "solve_atol": 1e-6,
            "grad_rtol": 1e-4,
            "grad_atol": 1e-6,
        }
    return {
        "solve_rtol": 2e-3,
        "solve_atol": 1e-4,
        "grad_rtol": 5e-2,
        "grad_atol": 5e-4,
    }


@pytest.mark.skipif(len(available_solvers) == 0, reason="No solvers available")
@pytest.mark.parametrize("solver", available_solvers)
def test_solver_resistance_differentiability(solver):
    """Sparse solver losses and gradients agree with the dense resistance API."""
    permeability = jnp.linspace(0.8, 1.4, 25, dtype=jnp.float32).reshape(5, 5)
    graph = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
    distance = ResistanceDistance(solver=solver)
    reference = ResistanceDistance(solver=None)
    state = distance.init(graph)
    tolerances = solver_tolerances(solver)
    nodes = jnp.array([0, 7, 24, 7], dtype=jnp.int32)

    for query in ({}, {"nodes": nodes}):

        def objective(values):
            current = GridGraph(values, fun=lambda x, y: (x + y) / 2)
            return jnp.sum(distance(current, state=state, **query) ** 2)

        # The dense reference runs in float64: a float32 pseudo-inverse on a
        # GPU leaves gradient errors of about 1e-4.
        def reference_objective(values):
            current = GridGraph(
                values.astype(jnp.float64), fun=lambda x, y: (x + y) / 2
            )
            return jnp.sum(reference(current, **query) ** 2)

        actual = distance(graph, state=state, **query)
        expected = reference(
            GridGraph(permeability.astype(jnp.float64), fun=lambda x, y: (x + y) / 2),
            **query,
        )
        assert jnp.allclose(
            actual,
            expected,
            rtol=tolerances["solve_rtol"],
            atol=tolerances["solve_atol"],
        )
        gradient = jax.jit(jax.grad(objective))(permeability)
        expected_gradient = jax.jit(jax.grad(reference_objective))(permeability)
        assert jnp.allclose(
            gradient,
            expected_gradient,
            rtol=tolerances["grad_rtol"],
            atol=tolerances["grad_atol"],
        )


def test_padded_sparse_resistance_has_zero_padding_gradient():
    """Duplicate positions, self-loops, and unused capacity retain their meaning."""
    edges = jnp.array([[0, 2], [2, 0], [1, 2], [2, 1]], dtype=jnp.int32)
    indices = jnp.concatenate([edges, edges, jnp.array([[0, 0], [3, 3], [3, 3]])])
    values = jnp.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    distance = ResistanceDistance(solver=lx.CG(rtol=1e-6, atol=1e-6))

    def objective(data):
        graph = Graph(BCOO((data, indices), shape=(3, 3)))
        return jnp.sum(distance(graph) ** 2)

    gradient = jax.jit(jax.grad(objective))(values)
    # On this two-edge tree, R01=1/2+1/4, R02=1/2, R12=1/4. With the ground at
    # the middle vertex 1 and the directed conductances a = a02, b = a20 and
    # c = a21, R01 = 1/a + b/(a c), R02 = 1/a and R12 = 1/c, so the loss
    # 2 (R01^2 + R02^2 + R12^2) has the derivatives -1.625, 0.375 and -0.25.
    # Each duplicated coefficient has the same derivative; the ground row
    # (a12), self-loop and padding do not contribute to the grounded Laplacian.
    expected = jnp.array(
        [-1.625, 0.375, 0.0, -0.25, -1.625, 0.375, 0.0, -0.25, 0.0, 0.0, 0.0]
    )
    assert jnp.allclose(gradient, expected)


SOLVERS = {
    "dense": lambda: None,
    "cholmod": lambda: CholmodSolver(),
    "cg": lambda: lx.CG(rtol=1e-12, atol=1e-12, max_steps=500),
}
# The sources, and the targets or every vertex, of a grid of `side` x `side`
# pixels: the middle vertex, the last one, duplicates, and sources among
# targets. Many sources to every vertex are the query of a tile.
MANY_SOURCES = (11, jnp.array([120, 60, 0, 60, 37]), None)
FEW_PAIRS = (5, jnp.array([24, 12, 3, 24]), jnp.array([12, 7, 3, 0, 24]))


@pytest.mark.parametrize(
    "solver, pairs",
    [("dense", MANY_SOURCES), ("cholmod", FEW_PAIRS), ("cg", FEW_PAIRS)],
    ids=["dense-many-sources", "cholmod-few-pairs", "cg-few-pairs"],
)
def test_sources_to_targets_match_all_pairs(solver, pairs, monkeypatch):
    if solver == "cholmod":
        pytest.importorskip("cholespy")
    # Queries whose potentials of all targets exceed a budget of values solve
    # for the targets in chunks: here every query, in chunks of at most 50,
    # three of 41 for the many sources, the last padded with vertex 0.
    monkeypatch.setattr(resistance_distance, "_VALUES_PER_STEP", 0)
    monkeypatch.setattr(resistance_distance, "_TARGET_CHUNK_SIZE", 50)
    jax.clear_caches()
    distance = ResistanceDistance(solver=SOLVERS[solver]())
    side, sources, targets = pairs
    selected = jnp.arange(side**2) if targets is None else targets
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-1, 0, (side, side)))
    weights = jnp.asarray(rng.uniform(size=(sources.size, selected.size)))

    def objective(permeability):
        grid = GridGraph(permeability, fun=avg)
        distances = distance(grid, sources=sources, targets=targets)
        return jnp.sum(weights * distances), distances

    def all_pairs(permeability):
        grid = GridGraph(permeability, fun=avg)
        distances = ResistanceDistance(solver=None)(grid)
        distances = distances[sources[:, None], selected[None, :]]
        return jnp.sum(weights * distances), distances

    (_, actual), gradient = jax.value_and_grad(objective, has_aux=True)(permeability)
    (_, expected), expected_gradient = jax.value_and_grad(all_pairs, has_aux=True)(
        permeability
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-9, atol=1e-12)


def test_spielman_sources_to_targets_match_its_all_pairs(monkeypatch):
    method = SpielmanApproximation(epsilon=0.3, seed=2, projection_batch_size=8)
    distance = ResistanceDistance(method=method)
    side, sources, targets = FEW_PAIRS
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-1, 0, (side, side)))
    weights = jnp.asarray(rng.uniform(size=(sources.size, targets.size)))

    def value_and_grad(query):
        def objective(permeability):
            return jnp.sum(weights * query(GridGraph(permeability, fun=avg)))

        return jax.value_and_grad(objective)(permeability)

    expected = value_and_grad(lambda grid: distance(grid)[sources[:, None], targets])
    # Queries whose random signs would hold more entries than this draw them
    # `projection_batch_size` projections at a time, from the same keys.
    monkeypatch.setattr(
        resistance_distance, "SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES", 0
    )
    jax.clear_caches()
    actual = value_and_grad(
        lambda grid: distance(grid, sources=sources, targets=targets)
    )
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("gathered", [True, False])
def test_spielman_window_queries_stream_the_projections_of_sources_queries(
    gathered, monkeypatch
):
    # Every query draws the signs of each projection from the seed and its
    # index: both queries estimate the distances with the same projections.
    # Here ceil(log(99) / 0.3**2) = 52 projections, in batches of 8 and a last
    # one of 4. The windows are gathered several at a time, and one at a time
    # without a budget of values per step.
    if not gathered:
        monkeypatch.setattr(resistance_distance, "_VALUES_PER_STEP", 0)
        jax.clear_caches()
    method = SpielmanApproximation(epsilon=0.3, seed=2, projection_batch_size=8)
    distance = ResistanceDistance(solver=None, method=method)
    rng = np.random.default_rng(0)
    grid = GridGraph(jnp.asarray(10 ** rng.uniform(-1, 0, (9, 11))), fun=avg)
    # Windows of 5 x 5 pixels, the first in the corner of the raster.
    rows, columns = np.array([2, 4, 6, 4]), np.array([2, 5, 8, 5])
    sources = jnp.asarray(rows * 11 + columns)

    actual = distance.sources_to_windows_distance(grid, sources, 5)

    rasters = distance(grid, sources=sources).reshape(4, 9, 11)
    expected = [
        raster[r - 2 : r + 3, c - 2 : c + 3]
        for raster, r, c in zip(rasters, rows, columns)
    ]
    np.testing.assert_allclose(actual, np.stack(expected), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("averaged", [False, True])
def test_spielman_window_query_gradients_estimate_those_of_exact_resistances(
    averaged,
):
    # The gradient of a weighted sum of the estimates, over 200 seeds of 20
    # projections, against that of the exact resistances, with negative and
    # positive weights, from the sources or from source weights on 3 x 3
    # pixels. Measured: at most 2.3 standard errors apart, and a median
    # relative standard deviation of 0.24 per pixel, against 0.46 for the
    # gradient of the estimates themselves, from the sources.
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-1, 0, (11, 13)))
    rows, columns = np.array([3, 5, 7, 5]), np.array([3, 6, 9, 6])
    sources = jnp.asarray(rows * 13 + columns)
    weights = -jnp.asarray(rng.uniform(size=(4, 7, 7))).at[1, 2, 2].set(0.7)
    source_weights = rng.uniform(size=(4, 3, 3))
    source_weights /= source_weights.sum(axis=(1, 2), keepdims=True)
    source_weights = jnp.asarray(source_weights) if averaged else None

    def objective(permeability, distance):
        grid = GridGraph(permeability, fun=avg)
        distances = distance.sources_to_windows_distance(
            grid, sources, 7, source_weights=source_weights
        )
        return jnp.sum(weights * distances)

    method = SpielmanApproximation(epsilon=0.5, projection_batch_size=16)
    randomized = ResistanceDistance(method=method)

    @jax.vmap
    def gradient(seed):
        return jax.grad(objective)(permeability, randomized.reseeded(seed))

    gradients = gradient(jnp.arange(200))
    expected = jax.grad(objective)(permeability, ResistanceDistance())
    spread = jnp.std(gradients, axis=0)
    error = jnp.abs(jnp.mean(gradients, axis=0) - expected) / (spread / np.sqrt(200))
    assert jnp.all(error < 4)
    assert jnp.median(spread / jnp.abs(expected)) < 0.3


@pytest.mark.parametrize(
    "method", [ExactResistance(), SpielmanApproximation(epsilon=0.3, seed=2)]
)
def test_window_queries_from_source_weights_average_the_distances_of_their_pixels(
    method,
):
    # Source weights on the 3 x 3 pixels around each source, on one pixel for
    # the last: the mean of the distances from these pixels, of the all-pairs
    # query, whose Spielman projections are those of the window query. Exact
    # distances have its gradient too.
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-1, 0, (9, 11)))
    rows, columns = np.array([2, 4, 6]), np.array([2, 5, 8])
    sources = jnp.asarray(rows * 11 + columns)
    weights = rng.uniform(size=(3, 3, 3)) * (rng.uniform(size=(3, 3, 3)) < 0.7)
    weights[2] = np.eye(3)[1][:, None] * np.eye(3)[1]
    weights = jnp.asarray(weights / weights.sum(axis=(1, 2), keepdims=True))
    distance = ResistanceDistance(solver=None, method=method)

    def query(permeability):
        grid = GridGraph(permeability, fun=avg)
        return distance.sources_to_windows_distance(
            grid, sources, 5, source_weights=weights
        )

    def mean(permeability):
        pairs = distance(GridGraph(permeability, fun=avg)).reshape(9, 11, 9, 11)
        windows = [
            pairs[r - 1 : r + 2, c - 1 : c + 2, r - 2 : r + 3, c - 2 : c + 3]
            for r, c in zip(rows, columns)
        ]
        return jnp.einsum("sxy,sxyab->sab", weights, jnp.stack(windows))

    expected = mean(permeability)
    np.testing.assert_allclose(query(permeability), expected, rtol=1e-9, atol=1e-12)
    if isinstance(method, ExactResistance):
        cotangent = jnp.asarray(rng.normal(size=(3, 5, 5)))

        def gradient(f):
            return jax.grad(lambda p: jnp.sum(cotangent * f(p)))(permeability)

        np.testing.assert_allclose(gradient(query), gradient(mean), rtol=1e-8)


def test_spielman_probe_right_hand_sides_add_the_windows_of_their_sources():
    # Windows of 7 x 7 pixels of a grid of sources, row after row, and two
    # sources out of that order, rows apart from their neighbours, in chunks of
    # as many sources as probes, with source weights on 3 x 3 pixels.
    rng = np.random.default_rng(0)
    rows, columns = np.meshgrid([3, 6, 9, 12, 16], [3, 8, 13, 19], indexing="ij")
    rows = np.r_[rows.ravel(), 16, 3]
    columns = np.r_[columns.ravel(), 3, 19]
    m, size, probes = rows.size, 7, 4
    roots = rng.normal(size=(m, size, size, 2))
    source_weights = rng.uniform(size=(m, 3, 3))
    per_source = rng.choice([-1.0, 1.0], (m, probes))
    per_pixel = rng.choice([-1.0, 1.0], (20, 23, probes))

    actual = resistance_distance._probe_right_hand_sides(
        jnp.asarray(roots),
        jnp.asarray(source_weights),
        jnp.asarray(rows),
        jnp.asarray(columns),
        jnp.asarray(per_source),
        jnp.asarray(per_pixel),
    )

    # Each source adds minus its roots, times its sign and the signs of the
    # pixels, to its window, and the roots times the signs of the pixels of its
    # window, times its sign, to its pixels with its source weights.
    expected = np.zeros(per_pixel.shape + (2,))
    for s, (row, column) in enumerate(zip(rows, columns)):
        window = (slice(row - 3, row + 4), slice(column - 3, column + 4))
        signs = per_pixel[window]
        added = roots[s][:, :, None, :] * per_source[s][:, None]
        expected[window] -= signs[..., None] * added
        own = per_source[s][:, None] * np.einsum("xyb,xyc->bc", signs, roots[s])
        block = (slice(row - 1, row + 2), slice(column - 1, column + 2))
        expected[block] += source_weights[s][:, :, None, None] * own
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_spielman_window_queries_of_float32_without_64_bit_types_match_float64():
    # Processes without 64-bit types query in float32: with one seed, their
    # projections and probes are those of float64 queries.
    method = SpielmanApproximation(epsilon=0.3, seed=2, projection_batch_size=8)
    distance = ResistanceDistance(method=method).reseeded(jnp.array(5))
    rng = np.random.default_rng(0)
    permeability = 10 ** rng.uniform(-1, 0, (9, 11))
    rows, columns = np.array([2, 4, 6]), np.array([2, 5, 8])
    weights = -rng.uniform(size=(3, 5, 5))

    def query(dtype):
        def objective(permeability):
            grid = GridGraph(permeability, fun=avg)
            sources = jnp.asarray(rows * 11 + columns)
            distances = distance.sources_to_windows_distance(grid, sources, 5)
            return jnp.sum(jnp.asarray(weights, dtype) * distances), distances

        gradient, distances = jax.grad(objective, has_aux=True)(
            jnp.asarray(permeability, dtype)
        )
        return distances, gradient

    expected = query(jnp.float64)
    with jax.enable_x64(False):
        actual = query(jnp.float32)
    for actual, expected in zip(actual, expected):
        assert actual.dtype == jnp.float32
        error = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
        assert error < 1e-5


def test_debiased_proximity_of_spielman_estimates_has_the_exact_expectation():
    # The proximity is convex in the distance: applied to unbiased randomized
    # distances, its expectation exceeds the proximity of the exact distances.
    rng = np.random.default_rng(0)
    grid = GridGraph(jnp.asarray(10 ** rng.uniform(-1, 0, (5, 5))), fun=avg)
    off_diagonal = ~jnp.eye(grid.nv, dtype=bool)
    distance = ResistanceDistance(solver=None, method=SpielmanApproximation(0.3))
    debiased = distance.debiased_proximity(lambda d: jnp.exp(-d), grid)

    def total(proximity, distances):
        return jnp.sum(jnp.where(off_diagonal, proximity(distances), 0))

    @jax.vmap
    def totals(seed):
        estimates = distance.reseeded(seed)(grid)
        return total(lambda d: jnp.exp(-d), estimates), total(debiased, estimates)

    exact = total(lambda d: jnp.exp(-d), ResistanceDistance(solver=None)(grid))
    raw, debiased = totals(jnp.arange(64))
    # Measured: +12 % and -2 %, with a standard error of 1.8 %.
    assert jnp.mean(raw) / exact - 1 > 0.08
    assert abs(jnp.mean(debiased) / exact - 1) < 0.04


def test_reseeded_spielman_distances_draw_other_projections():
    rng = np.random.default_rng(0)
    grid = GridGraph(jnp.asarray(10 ** rng.uniform(-1, 0, (5, 5))), fun=avg)
    distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.3))
    sources = jnp.array([0, 12])

    @jax.jit
    def reseeded(index):
        return distance.reseeded(index)(grid, sources=sources)

    # Equal to rounding: on an RTX 3090, two runs of one program with the
    # default `GridMultigridSolver` differed by up to 4e-15.
    np.testing.assert_allclose(reseeded(1), reseeded(1), rtol=1e-12, atol=1e-12)
    assert not np.allclose(reseeded(1), reseeded(2))
    assert not np.allclose(reseeded(1), distance(grid, sources=sources))


@pytest.mark.parametrize(
    "distance", [ResistanceDistance(), LCPDistance()], ids=["exact", "lcp"]
)
def test_exact_distances_keep_their_seed_and_proximity(distance):
    def proximity(distance):
        return jnp.exp(-distance)

    grid = GridGraph(jnp.ones((3, 3)), fun=avg)
    assert distance.reseeded(jnp.array(3)) is distance
    assert distance.debiased_proximity(proximity, grid) is proximity


def floored_window(rng, side=41):
    """The window of a pixel near a corner of a raster, whose padding has a
    permeability floor 1e-4 times the maximum, as in windowed analyses."""
    permeability = 10 ** rng.uniform(-1, 0, (side, side))
    permeability[-12:, :] = 1e-4
    permeability[:, -12:] = 1e-4
    return permeability, jnp.array([side**2 // 2])


def barrier_tile(rng, side=81):
    """A tile whose central pixel, and the pixels around it, are barriers at a
    permeability floor 1e-4 times the maximum."""
    permeability = 10 ** rng.uniform(-1, 0, (side, side))
    middle = side // 2
    permeability[middle - 2 : middle + 3, middle - 2 : middle + 3] = 1e-4
    sources = [0, 3 * side + 5, side**2 - 1, (middle - 6) * side + middle]
    return permeability, jnp.array(sources)


@pytest.mark.parametrize(
    "landscape, solver, tolerance",
    [
        (floored_window, "dense", 1e-3),
        (barrier_tile, "cholmod", 1e-3),
        pytest.param(barrier_tile, "dense", 5e-3, marks=pytest.mark.slow),
        pytest.param(barrier_tile, "cudss", 5e-3, marks=pytest.mark.gpu),
    ],
    ids=["window-dense", "tile-cholmod", "tile-dense", "tile-cudss"],
)
def test_float32_queries_keep_their_precision_on_floored_pixels(
    landscape, solver, tolerance
):
    # Measured: errors of float32 distances and gradients against float64 of
    # 6e-6 on the window, and on the tile 0.16 % with the dense solve and 4e-6
    # with CHOLMOD. Grounded at the last pixel, in a corner, the window was off
    # by 0.9 %; grounded at the central pixel, distances on the tile were off
    # by 35 % and 1 %, and gradients by 9 % and 27 %.
    if solver == "cholmod":
        pytest.importorskip("cholespy")
    if solver == "cudss":
        try:
            distance = ResistanceDistance(solver=CuDSSSolver())
        except (ImportError, RuntimeError) as error:
            pytest.skip(f"Needs an NVIDIA GPU, spineax and cuDSS: {error}")
    else:
        distance = ResistanceDistance(solver=SOLVERS[solver]())
    rng = np.random.default_rng(0)
    permeability, sources = landscape(rng)
    quality = jnp.asarray(rng.uniform(size=permeability.shape) * (permeability > 1e-3))

    def query(dtype):
        def connectivity(permeability):
            grid = GridGraph(permeability, fun=avg)
            distances = distance(grid, sources=sources)
            return jnp.sum(jnp.exp(-distances / 2) * quality.ravel()), distances

        gradient, distances = jax.grad(connectivity, has_aux=True)(
            jnp.asarray(permeability, dtype)
        )
        return distances, gradient

    for actual, expected in zip(query(jnp.float32), query(jnp.float64)):
        error = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
        assert error < tolerance
