import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape import Graph, GridGraph, LCPDistance
from jaxscape.distance import AbstractDistance
from jaxscape.graph import QUEEN_CONTIGUITY, ROOK_CONTIGUITY
from numpy.testing import assert_allclose
from scipy.sparse.csgraph import shortest_path


def _graph(weights, edges, n):
    return Graph(
        BCOO(
            (weights, jnp.asarray(edges, dtype=jnp.int32).reshape(-1, 2)), shape=(n, n)
        )
    )


def _check_dag(
    weights, edges, n, expected, gradient, *, sources=(0,), targets=None, seed=None
):
    sources = jnp.asarray(sources)
    targets = jnp.asarray((n - 1,) if targets is None else targets)

    def evaluate(a):
        values, pullback = jax.vjp(
            lambda x: LCPDistance()(
                _graph(x, edges, n), sources=sources, targets=targets
            ),
            a,
        )
        cotangent = (
            jnp.ones_like(values) if seed is None else jnp.asarray(seed, a.dtype)
        )
        return values, pullback(cotangent)[0]

    values, derivative = jax.jit(evaluate)(weights)
    assert values.dtype == derivative.dtype == weights.dtype
    tolerance = 3e-6 if weights.dtype == jnp.float32 else 1e-12
    assert_allclose(values, expected, rtol=tolerance)
    assert_allclose(derivative, gradient, rtol=tolerance, atol=0)


@pytest.mark.parametrize("algorithm", ["bellman-ford", "floyd-warshall"])
@pytest.mark.parametrize(
    "neighbors", [ROOK_CONTIGUITY, QUEEN_CONTIGUITY], ids=["rook", "queen"]
)
def test_lcp_distance(algorithm, neighbors):
    rng = np.random.default_rng(91)
    raster = jnp.asarray(rng.uniform(0.8, 2.1, (3, 4)))
    distance = LCPDistance(algorithm)
    sources, targets = jnp.array([0, 5, 0]), jnp.array([3, 11])

    def selected(x):
        graph = GridGraph(x, neighbors=neighbors, fun=lambda u, v: 2 / (u + v))
        return distance(graph, sources=sources, targets=targets)

    graph = GridGraph(raster, neighbors=neighbors, fun=lambda u, v: 2 / (u + v))
    adjacency = np.asarray(graph.get_adjacency_matrix().todense())
    costs = np.divide(
        1, adjacency, out=np.full_like(adjacency, np.inf), where=adjacency > 0
    )
    expected = shortest_path(costs, directed=True)[sources[:, None], targets[None, :]]
    assert_allclose(selected(raster), expected, rtol=1e-12)

    # A directional finite difference independently checks the raster chain rule.
    seed = jnp.array([[0.5, -1], [-0.7, 0.2], [1, -0.3]])
    loss = lambda x: jnp.sum(selected(x) * seed)
    direction = jnp.asarray(rng.normal(size=raster.shape))
    epsilon = 1e-5
    finite_difference = (
        loss(raster + epsilon * direction) - loss(raster - epsilon * direction)
    ) / (2 * epsilon)
    assert_allclose(
        jnp.sum(jax.jit(jax.grad(loss))(raster) * direction),
        finite_difference,
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "dtype,tiny,large", [(jnp.float32, 1e-30, 2**25), (jnp.float64, 1e-200, 2**54)]
)
def test_lcp_distance_dag(dtype, tiny, large):
    def triangle(a):
        return LCPDistance()(
            _graph(a, [(0, 2), (0, 1), (1, 2)], 5),
            sources=jnp.array([0]),
            targets=jnp.array([2]),
        )[0, 0]

    # Reuse one compilation across route changes, an exact tie, and a near tie.
    weights = jnp.array(
        [[1, 1, 1], [0.25, 1, 1], [0.5, 1, 1], [0.5, 1, 1 - 1e-5]], dtype
    )
    expected_values = [1, 2, 2, 2]
    expected_gradients = [[-1, 0, 0], [0, -1, -1], [-2, -0.5, -0.5], [-4, 0, 0]]
    compiled = jax.jit(jax.value_and_grad(triangle))
    for a, value, derivative in zip(weights, expected_values, expected_gradients):
        actual, gradient = compiled(a)
        assert_allclose(actual, value)
        assert_allclose(gradient, derivative)
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(triangle)))(weights)
    assert_allclose(values, expected_values)
    assert_allclose(gradients, expected_gradients)

    # Three tied paths receive probabilities 1/2, 1/4, 1/4 from local splitting.
    _check_dag(
        jnp.array([0.5, 1, 0.5, 1, 1, 1], dtype),
        [(0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 2)],
        5,
        [[3]],
        [-2, -0.5, -1, -0.5, -0.25, -0.25],
    )
    # Parallel edges stay distinct; unused tiny conductances and padding stay zero.
    _check_dag(
        jnp.array([0.5, 0.5, 1, 1, tiny, 0, 0], dtype),
        [(0, 2), (0, 2), (0, 1), (1, 2), (0, 2), (0, 2), (99, 99)],
        3,
        [[2]],
        [-4 / 3, -4 / 3, -1 / 3, -1 / 3, 0, 0, 0],
    )
    # Signed seeds accumulate repeated selections. Unreachable and source
    # distances contribute zero.
    _check_dag(
        jnp.array([0.5, 0.25], dtype),
        [(0, 1), (2, 3)],
        5,
        [[2, np.inf, 2, 0], [np.inf, 4, np.inf, np.inf], [2, np.inf, 2, 0]],
        [-4, 8],
        sources=(0, 2, 0),
        targets=(1, 3, 1, 0),
        seed=[[1, 0, -0.5, 3], [0, -0.5, 0, 0], [0.5, 0, 0, 0]],
    )
    # The full n-1 relaxation/adjoint bound, plus the empty one-vertex graph.
    _check_dag(
        jnp.ones(64, dtype), [(u, u + 1) for u in range(64)], 65, [[64]], -np.ones(64)
    )
    _check_dag(jnp.array([], dtype), [], 1, [[0]], [])

    for a in (tiny, 1 / tiny):
        for sign in (-1, 1):
            # Sequential division is representable even when a**2 is not.
            _check_dag(
                jnp.array([a], dtype),
                [(0, 1)],
                2,
                [[1 / a]],
                [-sign / a],
                seed=[[sign * a]],
            )

    def invalid(a, edges=((0, 1),), n=2):
        return LCPDistance()(_graph(a, edges, n), sources=jnp.array([0])).sum()

    for bad in (
        -1,
        np.nan,
        np.inf,
        np.finfo(np.dtype(dtype)).tiny / 16,
        np.finfo(np.dtype(dtype)).max,
    ):
        with pytest.raises(Exception, match="conductance|positive|finite"):
            invalid(jnp.array([bad], dtype)).block_until_ready()
    with pytest.raises(Exception, match="indices|bounds|index"):
        invalid(jnp.array([1], dtype), edges=((0, 2),)).block_until_ready()
    for differentiate in (False, True):
        # Gradient-only execution must retain the strict-order check too.
        loss = lambda a: invalid(a, edges=((0, 1), (1, 2)), n=3)
        calculation = jax.jit(jax.grad(loss) if differentiate else loss)
        with pytest.raises(Exception, match="strict|ordering|float64"):
            calculation(jnp.array([1 / large, 1], dtype)).block_until_ready()


def _average(x, y):
    return (x + y) / 2


def _edge_list_windows(graph, sources, size):
    """The window query of the edge-list Bellman–Ford, as the default slices it."""
    return AbstractDistance.sources_to_windows_distance(
        LCPDistance(), graph, sources, size
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(
    "neighbors", [ROOK_CONTIGUITY, QUEEN_CONTIGUITY], ids=["rook", "queen"]
)
@pytest.mark.parametrize("raster", ["contrast", "uniform"])
def test_grid_window_query_matches_the_edge_list(dtype, neighbors, raster):
    # High-contrast rasters give long, winding paths; uniform ones tie
    # everywhere, so that the gradient splits between tight edges.
    rng = np.random.default_rng(5)
    shape, size = (23, 27), 21
    values = 10 ** rng.uniform(-4, 0, shape) if raster == "contrast" else np.ones(shape)
    permeability = jnp.asarray(values, dtype)
    sources = jnp.array([11 * 27 + 13, 10 * 27 + 10, 12 * 27 + 16, 11 * 27 + 13])
    seed = jnp.asarray(rng.normal(size=(4, size, size)), dtype)

    def query(x, windows):
        return windows(GridGraph(x, fun=_average, neighbors=neighbors), sources, size)

    def grid_windows(graph, sources, size):
        return LCPDistance().sources_to_windows_distance(graph, sources, size)

    results = []
    for windows in (grid_windows, _edge_list_windows):
        distances, pullback = jax.jit(
            lambda x, windows=windows: jax.vjp(lambda y: query(y, windows), x)
        )(permeability)
        results.append((distances, pullback(seed)[0]))
    (distances, gradient), (expected, expected_gradient) = results
    assert distances.dtype == gradient.dtype == dtype
    # Both reach the unique fixed point of the floating-point Bellman equation.
    np.testing.assert_array_equal(distances, expected)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-12
    scale = np.max(np.abs(expected_gradient))
    assert_allclose(gradient, expected_gradient, rtol=tolerance, atol=tolerance * scale)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_grid_window_query_is_an_exact_fixed_point(dtype):
    # Every finite distance but the source's is the smallest rounded sum
    # `d_u + c_uv` over its incoming edges, as the gradient requires.
    rng = np.random.default_rng(8)
    side = 41
    np_dtype = np.dtype(dtype)
    raster = (10 ** rng.uniform(-4, 0, (side, side))).astype(np_dtype)
    graph = GridGraph(jnp.asarray(raster), fun=_average)
    source = side * side // 2
    distances = np.asarray(
        LCPDistance().sources_to_windows_distance(graph, jnp.array([source]), side)[0]
    )
    padded = np.pad(distances, 1, constant_values=np.inf)
    permeability = jnp.pad(jnp.asarray(raster), 1, constant_values=jnp.nan)
    best = np.full_like(distances, np.inf)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        tail = (slice(1 - dy, 1 - dy + side), slice(1 - dx, 1 - dx + side))
        # The device computes the costs: GPUs may round divisions otherwise.
        cost = np.asarray(1 / _average(permeability[tail], jnp.asarray(raster)))
        candidate = padded[tail] + cost
        best = np.fmin(best, np.where(np.isnan(cost), np.inf, candidate))
    best.flat[source] = 0
    assert best.dtype == np_dtype
    np.testing.assert_array_equal(distances, best)


def _loop_primitives(jaxpr, in_loop=False):
    """Names of the primitives inside the while loops of `jaxpr`, recursively."""
    names = set()
    for equation in jaxpr.eqns:
        if in_loop:
            names.add(equation.primitive.name)
        for value in equation.params.values():
            for sub in value if isinstance(value, (tuple, list)) else (value,):
                # Closed jaxprs hold their jaxpr.
                sub = getattr(sub, "jaxpr", sub)
                if hasattr(sub, "eqns"):
                    loop = in_loop or equation.primitive.name == "while"
                    names |= _loop_primitives(sub, loop)
    return names


def test_grid_window_query_relaxes_with_shifts():
    # The relaxation and adjoint rounds of grid windows neither gather nor
    # scatter, which is slow on GPUs.
    graph = GridGraph(jnp.ones((9, 9)), fun=_average)

    def loss(x):
        graph = GridGraph(x, fun=_average)
        return (
            LCPDistance().sources_to_windows_distance(graph, jnp.array([40]), 9).sum()
        )

    names = _loop_primitives(jax.make_jaxpr(jax.grad(loss))(graph.grid).jaxpr)
    assert "min" in names and "add" in names
    assert not {n for n in names if "scatter" in n or "gather" in n}, names


def test_grid_window_query_with_traced_neighbors_uses_the_edge_list():
    permeability = jnp.asarray(np.random.default_rng(2).uniform(0.1, 1, (9, 9)))
    sources = jnp.array([40, 30])
    graph = GridGraph(permeability, fun=_average)
    query = lambda graph: LCPDistance().sources_to_windows_distance(graph, sources, 5)
    # Traced neighbours, as those of a graph passed into `jax.jit`, give the
    # rounds of the sparse edges, which gather and scatter.
    names = _loop_primitives(jax.make_jaxpr(query)(graph).jaxpr)
    assert any("gather" in n for n in names), names
    assert any("scatter" in n for n in names), names
    np.testing.assert_array_equal(jax.jit(query)(graph), query(graph))


def test_grid_window_query_treats_zero_conductance_as_no_edge():
    # With conductance of the target pixel, a pixel of zero permeability cuts
    # every edge into it, and the pixels it walls off are unreachable.
    permeability = jnp.ones((5, 5)).at[:, 3].set(0)
    graph = GridGraph(permeability, fun=lambda x, y: y)
    sources = jnp.array([12])
    windows = LCPDistance().sources_to_windows_distance(graph, sources, 5)
    np.testing.assert_array_equal(windows, _edge_list_windows(graph, sources, 5))
    assert np.all(np.isinf(windows[0, :, 3:])) and np.all(
        np.isfinite(windows[0, :, :3])
    )


@pytest.mark.parametrize("dtype,large", [(jnp.float32, 2**25), (jnp.float64, 2**54)])
def test_grid_window_query_checks_conductances(dtype, large):
    def query(row):
        # The checks cover the whole graph, beyond the window of one pixel.
        graph = GridGraph(row[None, :], fun=lambda x, y: y)
        windows = LCPDistance().sources_to_windows_distance(graph, jnp.array([0]), 1)
        return windows.sum()

    for bad in (-1, np.nan, np.inf, np.finfo(np.dtype(dtype)).tiny / 16):
        with pytest.raises(Exception, match="conductance|positive|finite"):
            query(jnp.array([1, 1, bad], dtype)).block_until_ready()
    for differentiate in (False, True):
        calculation = jax.jit(jax.grad(query) if differentiate else query)
        with pytest.raises(Exception, match="strict|ordering|float64"):
            calculation(jnp.array([1, 1 / large, 1, 1], dtype)).block_until_ready()
