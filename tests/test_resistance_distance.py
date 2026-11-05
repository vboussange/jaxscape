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
from jaxscape import Graph, GridGraph, LCPDistance, SpielmanApproximation
from jaxscape.resistance_distance import (
    p_inv_resistance_distance,
    ResistanceDistance,
)
from jaxscape.solvers import AMJaxCGSolver, CholmodSolver, PyAMGSolver


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
    dist_pinv = ResistanceDistance()(grid)

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
    reference = ResistanceDistance()
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
        distances = ResistanceDistance()(grid)[sources[:, None], selected[None, :]]
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

    exact = total(proximity, ResistanceDistance()(grid))
    raw, debiased = [], []
    for seed in range(64):
        method = SpielmanApproximation(epsilon=0.3, seed=seed)
        distance = ResistanceDistance(method=method)
        estimates = distance(grid)
        raw.append(total(proximity, estimates))
        debiased.append(total(distance.debiased_proximity(proximity, grid), estimates))
    # Measured: +13 % and -1.3 %, with a standard error of 1.8 %.
    assert np.mean(raw) / exact - 1 > 0.08
    assert abs(np.mean(debiased) / exact - 1) < 0.04


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
