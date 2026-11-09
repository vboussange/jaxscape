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
from jaxscape.graph import QUEEN_CONTIGUITY, ROOK_CONTIGUITY
from jaxscape.resistance_distance import (
    p_inv_resistance_distance,
    ResistanceDistance,
)
from jaxscape.solvers import (
    AMJaxCGSolver,
    CholmodSolver,
    CuDSSSolver,
    GridMultigridSolver,
    GridMultigridState,
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


SOURCES_TO_TARGETS = {
    # One source to every vertex: the query of a moving window, whose 2209
    # targets the query solves for in two chunks.
    "one-source": (47, jnp.array([1104]), None),
    # Many sources to every vertex, as on a tile: the ground vertex (the middle
    # one), the last one and duplicates among the sources.
    "many-sources": (47, jnp.array([2208, 1104, 0, 1104, 37]), None),
    # The ground vertex, the last one, duplicates, and sources among targets.
    "few-pairs": (5, jnp.array([24, 12, 3, 24]), jnp.array([12, 7, 3, 0, 24])),
}
SOURCES_TO_TARGETS_SOLVERS = {
    "dense": lambda: None,
    "cholmod": lambda: CholmodSolver(),
    "cg": lambda: lx.CG(rtol=1e-12, atol=1e-12, max_steps=500),
}


@pytest.mark.parametrize(
    "solver, pairs",
    [
        (solver, pairs)
        for solver in SOURCES_TO_TARGETS_SOLVERS
        for pairs in SOURCES_TO_TARGETS
        # CG takes a minute on the large graphs, whose chunks do not depend on
        # the solver.
        if solver != "cg" or SOURCES_TO_TARGETS[pairs][0] == 5
    ],
)
def test_sources_to_targets_match_all_pairs(solver, pairs):
    if solver == "cholmod":
        pytest.importorskip("cholespy")
    distance = ResistanceDistance(solver=SOURCES_TO_TARGETS_SOLVERS[solver]())
    side, sources, targets = SOURCES_TO_TARGETS[pairs]
    selected = jnp.arange(side**2) if targets is None else targets
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-1, 0, (side, side)))
    weights = jnp.asarray(rng.uniform(size=(sources.size, selected.size)))

    def objective(permeability):
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        distances = distance(grid, sources=sources, targets=targets)
        return jnp.sum(weights * distances), distances

    def all_pairs(permeability):
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        distances = ResistanceDistance(solver=None)(grid)
        distances = distances[sources[:, None], selected[None, :]]
        return jnp.sum(weights * distances), distances

    (_, actual), gradient = jax.value_and_grad(objective, has_aux=True)(permeability)
    (_, expected), expected_gradient = jax.value_and_grad(all_pairs, has_aux=True)(
        permeability
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("pairs", SOURCES_TO_TARGETS)
def test_spielman_sources_to_targets_match_its_all_pairs(pairs):
    method = SpielmanApproximation(epsilon=0.3, seed=2)
    distance = ResistanceDistance(method=method)
    side, sources, targets = SOURCES_TO_TARGETS[pairs]
    selected = jnp.arange(side**2) if targets is None else targets
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-1, 0, (side, side)))
    weights = jnp.asarray(rng.uniform(size=(sources.size, selected.size)))

    def objective(query, permeability):
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(weights * query(grid))

    def sources_to_targets(grid):
        return distance(grid, sources=sources, targets=targets)

    def all_pairs(grid):
        return distance(grid)[sources[:, None], selected[None, :]]

    for argnums in (None, 1):
        actual, expected = (
            jax.grad(objective, argnums)(query, permeability)
            if argnums
            else objective(query, permeability)
            for query in (sources_to_targets, all_pairs)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_spielman_sources_to_targets_solve_for_the_projections():
    # One source to every vertex, as in a window: the projections need
    # ceil(log(3600) / 0.5**2) = 33 solves, the node potentials 3600.
    n = 3600
    permeability = jnp.linspace(0.5, 2.0, n).reshape(60, 60)
    method = SpielmanApproximation(epsilon=0.5)
    solver = lx.CG(rtol=1e-8, atol=1e-8, max_steps=500)
    distance = ResistanceDistance(solver=solver, method=method)

    def objective(permeability):
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(distance(grid, sources=jnp.array([1830])))

    # No dense matrix of all vertices: the random signs of the projections
    # take most of the memory.
    dense = n**2 * permeability.dtype.itemsize
    for program in (jax.jit(objective), jax.jit(jax.grad(objective))):
        memory = program.lower(permeability).compile().memory_analysis()
        assert memory is not None
        assert memory.temp_size_in_bytes < dense / 2


@pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="Cholmod not available")
def test_spielman_sources_to_targets_draw_the_signs_of_large_graphs_in_batches():
    # ceil(log(10000) / 0.04**2) = 5757 projections of 39600 directed edges:
    # more random signs than `SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES`, which
    # would take 1.8 GB at once. The query draws them `projection_batch_size`
    # projections at a time, and holds the potentials of 5757 projections.
    side = 100
    permeability = jnp.linspace(0.5, 2.0, side**2).reshape(side, side)
    method = SpielmanApproximation(epsilon=0.04)
    distance = ResistanceDistance(solver=CholmodSolver(), method=method)
    signs = 5757 * 4 * side * (side - 1)
    assert signs > resistance_distance.SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES

    def objective(permeability):
        grid = GridGraph(permeability, fun=avg)
        return jnp.sum(distance(grid, sources=jnp.array([side**2 // 2])))

    program = jax.jit(objective).lower(permeability).compile()
    memory = program.memory_analysis()
    assert memory is not None
    # Measured on a CPU: 0.98 GB, and 6.0 GB with the signs drawn at once.
    assert memory.temp_size_in_bytes < signs * permeability.dtype.itemsize


@pytest.mark.parametrize("solver", ["dense", "cholmod"])
@pytest.mark.parametrize("gathered", [True, False])
def test_spielman_window_queries_stream_the_projections_of_sources_queries(
    solver, gathered, monkeypatch
):
    # Every query draws the signs of each projection from the seed and its
    # index: both queries estimate the distances with the same projections.
    # Here ceil(log(99) / 0.3**2) = 52 projections, in batches of 8 and a last
    # one of 4. Small windows are gathered several at a time, large ones taken
    # one at a time.
    if solver == "cholmod":
        pytest.importorskip("cholespy")
    if not gathered:
        monkeypatch.setattr(resistance_distance, "_LARGEST_GATHERED_WINDOW", 0)
        jax.clear_caches()
    method = SpielmanApproximation(epsilon=0.3, seed=2, projection_batch_size=8)
    distance = ResistanceDistance(SOURCES_TO_TARGETS_SOLVERS[solver](), method=method)
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


def test_spielman_window_query_gradients_estimate_those_of_exact_resistances():
    # The gradient of a weighted sum of the estimates, over 200 seeds of 20
    # projections, against that of the exact resistances, with negative and
    # positive weights. Measured: at most 2.3 standard errors apart, and a
    # median relative standard deviation of 0.24 per pixel, against 0.46 for
    # the gradient of the estimates themselves.
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-1, 0, (11, 13)))
    rows, columns = np.array([3, 5, 7, 5]), np.array([3, 6, 9, 6])
    sources = jnp.asarray(rows * 13 + columns)
    weights = -jnp.asarray(rng.uniform(size=(4, 7, 7))).at[1, 2, 2].set(0.7)

    def objective(permeability, distance):
        grid = GridGraph(permeability, fun=avg)
        distances = distance.sources_to_windows_distance(grid, sources, 7)
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


@pytest.mark.parametrize("sources_per_strip", [64, 4])
def test_spielman_probe_right_hand_sides_add_the_windows_of_their_sources(
    sources_per_strip, monkeypatch
):
    # Windows of 7 x 7 pixels of a grid of sources, row after row, and two
    # sources out of that order, rows apart from their neighbours.
    monkeypatch.setattr(resistance_distance, "_SOURCES_PER_STRIP", sources_per_strip)
    rng = np.random.default_rng(0)
    rows, columns = np.meshgrid([3, 6, 9, 12, 16], [3, 8, 13, 19], indexing="ij")
    rows = np.r_[rows.ravel(), 16, 3]
    columns = np.r_[columns.ravel(), 3, 19]
    m, size, probes = rows.size, 7, 5
    weights = rng.normal(size=(m, size, size, 2))
    per_source = rng.choice([-1.0, 1.0], (m, probes))
    per_pixel = rng.choice([-1.0, 1.0], (20, 23, probes))

    actual = resistance_distance._probe_right_hand_sides(
        jnp.asarray(weights),
        jnp.asarray(rows),
        jnp.asarray(columns),
        jnp.asarray(per_source),
        jnp.asarray(per_pixel),
    )

    # Each source adds its weights, times its sign, to its window, and the
    # weights times the signs of the pixels of its window to itself.
    spread = np.zeros(per_pixel.shape + (2,))
    own = np.zeros((m, probes, 2))
    for s, (row, column) in enumerate(zip(rows - 3, columns - 3)):
        window = (slice(row, row + size), slice(column, column + size))
        spread[window] += weights[s][:, :, None, :] * per_source[s][:, None]
        signs = per_pixel[window]
        own[s] = per_source[s][:, None] * np.einsum("xyb,xyc->bc", signs, weights[s])
    expected = -per_pixel[..., None] * spread
    np.add.at(expected, (rows, columns), own)
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


def test_spielman_window_queries_hold_one_batch_of_projections():
    # ceil(log(3600) / epsilon**2) = 205 and 3276 projections of 3600
    # vertices, solved 16 at a time from 4 sources with windows of 21 x 21
    # pixels.
    side = 60
    permeability = jnp.linspace(0.5, 2.0, side**2).reshape(side, side)
    solver = lx.CG(rtol=1e-8, atol=1e-8, max_steps=500)
    sources = jnp.array([10 * side + 10, 30 * side + 30, 49 * side + 49, 1830])

    def memory(epsilon, gradient):
        method = SpielmanApproximation(epsilon=epsilon, projection_batch_size=16)
        distance = ResistanceDistance(solver=solver, method=method)

        def objective(permeability):
            grid = GridGraph(permeability, fun=avg)
            return jnp.sum(distance.sources_to_windows_distance(grid, sources, 21))

        program = jax.jit(jax.grad(objective) if gradient else objective)
        stats = program.lower(permeability).compile().memory_analysis()
        assert stats is not None
        return stats.temp_size_in_bytes

    # Neither the potentials of all projections nor their random signs, one
    # per stored edge, are held at once: 16 times more projections take no
    # more memory. Measured: 8.7 MB, and 13.4 MB for the gradient, against
    # 94 MB for the potentials of 3276 projections.
    for gradient in (False, True):
        assert memory(0.05, gradient) <= 1.01 * memory(0.2, gradient)


def test_few_sources_to_targets_solve_for_them_only():
    n = 900
    permeability = jnp.linspace(0.5, 2.0, n).reshape(30, 30)
    distance = ResistanceDistance(solver=lx.CG(rtol=1e-8, atol=1e-8, max_steps=500))
    sources = jnp.array([0, 100])
    targets = jnp.array([n - 1, 450, 31])

    def objective(permeability):
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(distance(grid, sources=sources, targets=targets))

    # Temporary buffers of the query and of its gradient stay far below one
    # dense matrix of all vertices.
    dense = n**2 * permeability.dtype.itemsize
    for program in (jax.jit(objective), jax.jit(jax.grad(objective))):
        memory = program.lower(permeability).compile().memory_analysis()
        assert memory is not None
        assert memory.temp_size_in_bytes < dense / 4


def test_debiased_proximity_of_spielman_estimates_has_the_exact_expectation():
    # The proximity is convex in the distance: applied to unbiased randomized
    # distances, its expectation exceeds the proximity of the exact distances.
    rng = np.random.default_rng(0)
    grid = GridGraph(jnp.asarray(10 ** rng.uniform(-1, 0, (5, 5))), fun=avg)
    off_diagonal = ~jnp.eye(grid.nv, dtype=bool)

    def proximity(distance):
        return jnp.exp(-distance)

    def total(proximity, distances):
        return float(jnp.sum(jnp.where(off_diagonal, proximity(distances), 0)))

    exact = total(proximity, ResistanceDistance(solver=None)(grid))
    raw, debiased = [], []
    for seed in range(64):
        # Each seed compiles its queries: dense solves compile fastest.
        method = SpielmanApproximation(epsilon=0.3, seed=seed)
        distance = ResistanceDistance(solver=None, method=method)
        estimates = distance(grid)
        raw.append(total(proximity, estimates))
        debiased.append(total(distance.debiased_proximity(proximity, grid), estimates))
    # Measured: +13 % and -1.3 %, with a standard error of 1.8 %.
    assert np.mean(raw) / exact - 1 > 0.08
    assert abs(np.mean(debiased) / exact - 1) < 0.04


@pytest.mark.parametrize(
    "query", [{}, {"sources": jnp.array([0, 12])}, {"nodes": jnp.array([3, 7])}]
)
def test_reseeded_spielman_distances_draw_other_projections(query):
    rng = np.random.default_rng(0)
    grid = GridGraph(jnp.asarray(10 ** rng.uniform(-1, 0, (5, 5))), fun=avg)
    distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.3))

    @jax.jit
    def reseeded(index):
        return distance.reseeded(index)(grid, **query)

    # Equal to rounding: on an RTX 3090, two runs of one program with the
    # default `GridMultigridSolver` differed by up to 4e-15.
    np.testing.assert_allclose(reseeded(1), reseeded(1), rtol=1e-12, atol=1e-12)
    assert not np.allclose(reseeded(1), reseeded(2))
    assert not np.allclose(reseeded(1), distance(grid, **query))


@pytest.mark.parametrize(
    "distance", [ResistanceDistance(), LCPDistance()], ids=["exact", "lcp"]
)
def test_exact_distances_keep_their_seed(distance):
    assert distance.reseeded(jnp.array(3)) is distance


@pytest.mark.parametrize(
    "distance", [ResistanceDistance(), LCPDistance()], ids=["exact", "lcp"]
)
def test_exact_distances_keep_their_proximity(distance):
    def proximity(distance):
        return jnp.exp(-distance)

    grid = GridGraph(jnp.ones((3, 3)), fun=avg)
    assert distance.debiased_proximity(proximity, grid) is proximity


def test_one_source_to_all_targets_streams_the_targets():
    # The query of a moving window: one source to every vertex of a large
    # window.
    side = 401
    n = side**2
    permeability = jnp.linspace(0.5, 2.0, n, dtype=jnp.float32).reshape(side, side)
    distance = ResistanceDistance(solver=lx.CG(rtol=1e-6, atol=1e-6, max_steps=500))
    source = jnp.array([n // 2])

    def objective(permeability):
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(distance(grid, sources=source))

    # Neither the distances nor their gradient hold the potentials of every
    # target at once.
    dense = n**2 * permeability.dtype.itemsize
    for program in (jax.jit(objective), jax.jit(jax.grad(objective))):
        memory = program.lower(permeability).compile().memory_analysis()
        assert memory is not None
        assert memory.temp_size_in_bytes < dense / 4


@pytest.mark.parametrize("solver", ["dense", "cholmod"])
def test_float32_permeability_gradients_of_a_window_on_the_floor(solver):
    if solver == "cholmod":
        pytest.importorskip("cholespy")
    distance = ResistanceDistance(solver=SOURCES_TO_TARGETS_SOLVERS[solver]())
    # The window of a pixel near a corner of a raster, whose padding has a
    # permeability floor 1e-4 times the maximum, as in windowed analyses.
    side = 41
    rng = np.random.default_rng(0)
    permeability = 10 ** rng.uniform(-1, 0, (side, side))
    permeability[-12:, :] = 1e-4
    permeability[:, -12:] = 1e-4
    quality = jnp.asarray(rng.uniform(size=(side, side)) * (permeability > 1e-3))
    centre = jnp.array([side**2 // 2])

    def gradient(dtype):
        def connectivity(permeability):
            grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
            distances = distance(grid, sources=centre)[0]
            return jnp.sum(jnp.exp(-distances / 2) * quality.ravel())

        return jax.grad(connectivity)(jnp.asarray(permeability, dtype))

    expected = gradient(jnp.float64)
    actual = gradient(jnp.float32)
    error = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
    assert error < 1e-3


@pytest.mark.parametrize(
    "solver", ["dense", "cholmod", pytest.param("cudss", marks=pytest.mark.gpu)]
)
def test_float32_queries_of_a_tile_whose_centre_is_a_barrier(solver):
    if solver == "cholmod":
        pytest.importorskip("cholespy")
    if solver == "cudss":
        try:
            distance = ResistanceDistance(solver=CuDSSSolver())
        except (ImportError, RuntimeError) as error:
            pytest.skip(f"Needs an NVIDIA GPU, spineax and cuDSS: {error}")
    else:
        distance = ResistanceDistance(solver=SOURCES_TO_TARGETS_SOLVERS[solver]())
    # A tile whose central pixel, and the pixels around it, are barriers at a
    # permeability floor 1e-4 times the maximum. Measured: float32 gradients
    # within 0.16 % of float64 with the dense solve, and 4e-6 with CHOLMOD;
    # grounded at the central pixel, distances were off by 35 % and 1 %, and
    # gradients by 9 % and 27 %.
    side = 81
    rng = np.random.default_rng(0)
    permeability = 10 ** rng.uniform(-1, 0, (side, side))
    middle = side // 2
    permeability[middle - 2 : middle + 3, middle - 2 : middle + 3] = 1e-4
    quality = jnp.asarray(rng.uniform(size=(side, side)) * (permeability > 1e-3))
    sources = jnp.array([0, 3 * side + 5, side**2 - 1, (middle - 6) * side + middle])

    def query(dtype):
        def connectivity(permeability):
            grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
            distances = distance(grid, sources=sources)
            return jnp.sum(jnp.exp(-distances / 2) * quality.ravel()), distances

        gradient, distances = jax.grad(connectivity, has_aux=True)(
            jnp.asarray(permeability, dtype)
        )
        return distances, gradient

    for actual, expected in zip(query(jnp.float32), query(jnp.float64)):
        error = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
        assert error < 5e-3


# The default solver, `GridMultigridSolver()`. Grids of up to 1024 pixels are
# the coarsest grid of its hierarchy, which a Cholesky factorization solves.
SMALL_GRID = (24, 40)


def small_landscape(dtype, shape=SMALL_GRID):
    rng = np.random.default_rng(3)
    return jnp.asarray(10 ** rng.uniform(-2, 0, shape), dtype)


def test_the_default_solver_is_grid_multigrid():
    assert ResistanceDistance().solver == GridMultigridSolver(rtol=1e-5, max_steps=200)
    assert ResistanceDistance(solver=None).solver is None


DEFAULT_QUERIES = {
    "all-pairs": {},
    "nodes": {"nodes": jnp.array([0, 37, 959, 480, 37])},
    "sources-to-targets": {
        "sources": jnp.array([5, 480]),
        "targets": jnp.array([959, 0, 123, 480]),
    },
}


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("query", DEFAULT_QUERIES)
def test_the_default_solves_small_grids_as_the_dense_solve(dtype, query):
    # Against the dense solve in float64. Measured: relative errors of the
    # weighted sums of at most 1.5e-6 and of their gradients 3.5e-6 in
    # float32, and 3e-12 in float64. In float32, those of the pseudo-inverse
    # of the all-pairs and nodes-to-nodes queries were 35 % and 42 %.
    weights = np.random.default_rng(4).uniform(size=(960, 960))

    def objective(permeability, distance):
        distances = distance(GridGraph(permeability, fun=avg), **DEFAULT_QUERIES[query])
        m, k = distances.shape
        return jnp.sum(jnp.asarray(weights[:m, :k], permeability.dtype) * distances)

    def value_and_grad(dtype, distance):
        return jax.value_and_grad(objective)(small_landscape(dtype), distance)

    actual = value_and_grad(dtype, ResistanceDistance())
    expected = value_and_grad(jnp.float64, ResistanceDistance(solver=None))
    for a, e in zip(actual, expected):
        assert a.dtype == dtype
        error = jnp.linalg.norm(a - e) / jnp.linalg.norm(e)
        assert error < (1e-5 if dtype == jnp.float32 else 1e-10)


def test_the_default_runs_eagerly_and_under_jax_transformations():
    # Its state, built from the shape and contiguity of a grid graph, also
    # inside `jax.jit` and for a graph that `eqx.filter_jit` traces whole.
    permeability = small_landscape(jnp.float64, (12, 15))
    nodes = jnp.array([0, 17, 179, 90])
    distance = ResistanceDistance()

    def objective(permeability, distance=distance):
        grid = GridGraph(permeability, fun=avg)
        return jnp.sum(jnp.exp(-distance(grid, nodes=nodes)))

    grid = GridGraph(permeability, fun=avg)
    expected = ResistanceDistance(solver=None)(grid, nodes=nodes)
    expected_gradient = jax.grad(objective)(permeability, ResistanceDistance(None))
    for actual in (
        distance(grid, nodes=nodes),
        jax.jit(lambda p: distance(GridGraph(p, fun=avg), nodes=nodes))(permeability),
        filter_jit(distance)(grid, nodes=nodes),
        distance.nodes_to_nodes_distance(grid, nodes),
        distance.all_pairs_distance(grid)[nodes[:, None], nodes],
        distance.sources_to_targets_distance(grid, nodes, nodes),
        jax.vmap(lambda p: distance(GridGraph(p, fun=avg), nodes=nodes))(
            jnp.stack([permeability, permeability])
        )[1],
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
    for gradient in (
        jax.grad(objective)(permeability),
        jax.jit(jax.grad(objective))(permeability),
        filter_jit(filter_grad(lambda g: jnp.sum(distance(g, nodes=nodes))))(grid).grid,
    ):
        assert gradient.shape == permeability.shape
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(permeability), expected_gradient, rtol=1e-8
    )


@pytest.mark.parametrize("neighbors", [ROOK_CONTIGUITY, QUEEN_CONTIGUITY])
def test_the_default_state_of_a_traced_grid_graph_is_that_of_the_graph(neighbors):
    grid = GridGraph(small_landscape(jnp.float64, (9, 11)), avg, neighbors)
    distance = ResistanceDistance()

    expected = distance.init(grid)
    assert isinstance(expected, GridMultigridState)
    for traced in (
        jax.jit(lambda p: distance.init(GridGraph(p, avg, neighbors)))(grid.grid),
        filter_jit(distance.init)(grid),
    ):
        assert jax.tree.structure(traced) == jax.tree.structure(expected)
        for a, e in zip(jax.tree.leaves(traced), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a, e)


def test_the_default_rejects_graphs_other_than_grids():
    ring = np.zeros((10, 10))
    for i in range(10):
        ring[i, (i + 1) % 10] = ring[(i + 1) % 10, i] = 1.0
    graph = Graph(BCOO.fromdense(jnp.asarray(ring)))
    with pytest.raises(ValueError, match=r"grid graphs.*`solver=None`"):
        ResistanceDistance()(graph, nodes=jnp.array([0, 5]))
    # The dense solve: the resistance of two paths of 5 edges in parallel.
    distances = ResistanceDistance(solver=None)(graph, nodes=jnp.array([0, 5]))
    np.testing.assert_allclose(distances[0, 1], 2.5)


@pytest.mark.parametrize("query", DEFAULT_QUERIES)
def test_the_default_raises_on_disconnected_graphs(query):
    # A column of zero permeability cuts the grid in two: the grounded
    # Laplacian is singular.
    permeability = small_landscape(jnp.float64).at[:, 20].set(0)
    grid = GridGraph(permeability, fun=jnp.minimum)
    with pytest.raises(Exception, match="non-finite"):
        jax.block_until_ready(ResistanceDistance()(grid, **DEFAULT_QUERIES[query]))


SINGLE_PIXEL_QUERIES = {
    "all-pairs": {},
    "nodes": {"nodes": jnp.array([0])},
    "sources-to-targets": {"sources": jnp.array([0]), "targets": jnp.array([0])},
}


@pytest.mark.parametrize("solver", ["default", "dense"])
@pytest.mark.parametrize("query", SINGLE_PIXEL_QUERIES)
def test_a_grid_of_one_pixel_is_at_zero_resistance_from_itself(solver, query):
    # The pixel has no neighbours: its ground has degree zero.
    distance = ResistanceDistance() if solver == "default" else ResistanceDistance(None)

    def objective(permeability):
        grid = GridGraph(permeability, fun=avg)
        distances = distance(grid, **SINGLE_PIXEL_QUERIES[query])
        return jnp.sum(distances), distances

    (_, distances), gradient = jax.value_and_grad(objective, has_aux=True)(
        jnp.ones((1, 1))
    )
    np.testing.assert_array_equal(distances, [[0.0]])
    np.testing.assert_array_equal(gradient, [[0.0]])
