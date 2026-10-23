"""The Bellman-Ford DAG convention, checked only through the public distance API."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape import Graph, GridGraph, LCPDistance
from jaxscape.graph import QUEEN_CONTIGUITY, ROOK_CONTIGUITY
from numpy.testing import assert_allclose
from scipy.sparse.csgraph import shortest_path


ALGORITHM = "bellman-ford"


def graph_from_edges(weights, edges, n):
    return Graph(BCOO((weights, jnp.asarray(edges).reshape(-1, 2)), shape=(n, n)))


def scipy_distances(weights, edges, n):
    """Build costs explicitly; parallel edges compete instead of being summed."""
    costs = np.full((n, n), np.inf)
    np.fill_diagonal(costs, 0)
    for weight, (u, v) in zip(weights, edges):
        if weight > 0:
            costs[u, v] = np.minimum(costs[u, v], 1 / weight)
    return shortest_path(costs, directed=True)


def unique_path_derivative(weights, edges, n, sources, targets, cotangent):
    """Enumerate simple paths independently on tiny graphs with unique optima."""
    weights = np.asarray(weights)
    sources, targets, cotangent = map(np.asarray, (sources, targets, cotangent))
    result = np.zeros_like(weights)
    outgoing = [[] for _ in range(n)]
    for edge, ((u, v), weight) in enumerate(zip(edges, weights)):
        if weight > 0:
            outgoing[u].append((v, edge))

    def paths(u, target, visited, incidence):
        if u == target:
            yield incidence
        else:
            for v, edge in outgoing[u]:
                if v not in visited:
                    yield from paths(v, target, visited | {v}, incidence + [edge])

    for row, source in enumerate(sources):
        for column, target in enumerate(targets):
            candidates = list(paths(source, target, {source}, []))
            if not candidates:
                assert cotangent[row, column] == 0
                continue
            costs = np.array([sum(1 / weights[e] for e in p) for p in candidates])
            best = np.argmin(costs)
            assert np.count_nonzero(costs == costs[best]) == 1
            for edge in candidates[best]:
                result[edge] -= cotangent[row, column] / weights[edge] ** 2
    return result


@pytest.mark.parametrize("n", [1, 2, 7, 65])
def test_directed_corridor_and_isolated_source_match_scipy(n):
    # The last vertex is isolated; n=65 still needs 63 rounds to reach its end.
    edges = np.array([(u, u + 1) for u in range(max(0, n - 2))], dtype=np.int32)
    weights = jnp.asarray(np.linspace(0.5, 1.5, len(edges)))
    graph = graph_from_edges(weights, edges, n)
    sources = jnp.array([0, n - 1])
    result = LCPDistance(ALGORITHM)(graph, sources=sources)
    assert_allclose(result, scipy_distances(weights, edges, n)[sources], rtol=1e-6)
    assert result[0, 0] == 0
    assert result[-1, -1] == 0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_directed_values_unique_path_gradients_and_finite_differences(dtype):
    edges = np.array([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (0, 4)])
    weights = jnp.asarray(1 / np.array([1.2, 3.7, 0.6, 2.9, 0.8, 1.1, 8.3]), dtype)
    sources, targets = jnp.array([1, 0]), jnp.array([4, 2, 3])
    seed = jnp.asarray([[0.3, -1.1, 0.7], [-0.8, 1.7, -0.2]], dtype)

    def loss(a):
        return jnp.sum(
            LCPDistance(ALGORITHM)(
                graph_from_edges(a, edges, 6), sources=sources, targets=targets
            )
            * seed
        )

    actual = LCPDistance(ALGORITHM)(graph_from_edges(weights, edges, 6))
    assert actual.dtype == dtype
    tolerance = 3e-6 if dtype == jnp.float32 else 1e-12
    assert_allclose(actual, scipy_distances(weights, edges, 6), rtol=tolerance)
    value, derivative = jax.jit(jax.value_and_grad(loss))(weights)
    expected = unique_path_derivative(weights, edges, 6, sources, targets, seed)
    assert derivative.dtype == dtype
    assert_allclose(derivative, expected, rtol=tolerance, atol=tolerance)
    assert_allclose(value, loss(weights), rtol=2e-6)
    epsilon = 2e-3 if dtype == jnp.float32 else 1e-5
    for edge in range(len(edges)):
        step = jnp.zeros_like(weights).at[edge].set(epsilon)
        finite_difference = (loss(weights + step) - loss(weights - step)) / (
            2 * epsilon
        )
        fd_tolerance = 3e-3 if dtype == jnp.float32 else 1e-6
        assert_allclose(
            derivative[edge], finite_difference, rtol=fd_tolerance, atol=fd_tolerance
        )


@pytest.mark.parametrize("n", [3, 5, 6])
def test_two_route_tie_is_independent_of_isolated_vertices(n):
    edges = np.array([(0, 2), (0, 1), (1, 2)])
    weights = jnp.array([0.5, 1.0, 1.0])

    def loss(a):
        return LCPDistance(ALGORITHM)(
            graph_from_edges(a, edges, n),
            sources=jnp.array([0]),
            targets=jnp.array([2]),
        )[0, 0]

    value, derivative = jax.value_and_grad(loss)(weights)
    assert value == 2
    assert_allclose(derivative, [-2, -0.5, -0.5], atol=0, rtol=0)


def test_three_route_tie_shares_predecessors_not_complete_paths():
    # Paths s-a-t, s-b-t, s-c-b-t cost 3 and carry probabilities 1/2, 1/4, 1/4.
    edges = np.array([(0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 2)])
    weights = jnp.array([0.5, 1, 0.5, 1, 1, 1])

    def loss(a):
        return LCPDistance(ALGORITHM)(
            graph_from_edges(a, edges, 5),
            sources=jnp.array([0]),
            targets=jnp.array([4]),
        )[0, 0]

    value, derivative = jax.jit(jax.value_and_grad(loss))(weights)
    assert value == 3
    assert_allclose(derivative, [-2, -0.5, -1, -0.5, -0.25, -0.25], atol=0, rtol=0)


def test_signed_vjp_repeated_selections_and_linearity():
    edges = np.array([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
    weights = jnp.array([1, 0.25, 0.5, 0.1, 0.75, 0.8])
    sources, targets = jnp.array([1, 0, 1]), jnp.array([4, 2, 4, 5])

    def selected(a):
        return LCPDistance(ALGORITHM)(
            graph_from_edges(a, edges, 6), sources=sources, targets=targets
        )

    values, pullback = jax.vjp(selected, weights)
    seed = np.array([[1, -2, 3, 0], [-0.3, 0.8, -1, 0], [0.5, -0.7, 2, 0]])
    other = np.array([[-1, 0.4, 0.2, 0], [2, -0.9, 1.2, 0], [-1, 0.3, -0.7, 0]])
    expected = unique_path_derivative(weights, edges, 6, sources, targets, seed)
    assert np.all(np.isinf(values[:, -1]))
    assert_allclose(pullback(jnp.asarray(seed))[0], expected, rtol=1e-6, atol=1e-6)
    assert_allclose(
        pullback(jnp.asarray(2 * seed - 3 * other))[0],
        2 * pullback(jnp.asarray(seed))[0] - 3 * pullback(jnp.asarray(other))[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_parallel_edge_identities_and_absent_capacity_have_separate_gradients():
    # Three cost-2 routes: two distinct direct edges and a route via vertex 1.
    # The cost-4 parallel edge loses; zero duplicate and out-of-bounds slots are absent.
    edges = np.array(
        [
            (0, 2),
            (0, 2),
            (0, 1),
            (1, 2),
            (0, 2),
            (0, 2),
            (3, 3),
            (-1, -1),
            (100, 1),
            (1, 100),
        ]
    )
    weights = jnp.array([0.5, 0.5, 1, 1, 0.25, 0, 0, 0, 0, 0])

    def loss(a, edge_order=edges):
        return LCPDistance(ALGORITHM)(
            graph_from_edges(a, edge_order, 3),
            sources=jnp.array([0]),
            targets=jnp.array([2]),
        )[0, 0]

    value, derivative = jax.jit(jax.value_and_grad(loss))(weights)
    assert value == 2
    assert_allclose(derivative, [-4 / 3, -4 / 3, -1 / 3, -1 / 3, 0, 0, 0, 0, 0, 0])
    order = np.array([8, 3, 1, 5, 7, 2, 4, 0, 9, 6])
    assert_allclose(
        jax.grad(lambda a: loss(a, edges[order]))(weights[order]), derivative[order]
    )


def test_vertex_relabeling_preserves_tied_values_and_gradients():
    edges = np.array([(0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 2)])
    weights = jnp.array([0.5, 1, 0.5, 1, 1, 1])
    labels = np.array([4, 1, 0, 2, 3])

    def loss(a, relabel):
        return LCPDistance(ALGORITHM)(
            graph_from_edges(a, relabel[edges], 5),
            sources=jnp.array([relabel[0]]),
            targets=jnp.array([relabel[4]]),
        )[0, 0]

    expected = jax.value_and_grad(lambda a: loss(a, np.arange(5)))(weights)
    actual = jax.value_and_grad(lambda a: loss(a, labels))(weights)
    for observed, reference in zip(actual, expected):
        assert_allclose(observed, reference)


@pytest.mark.parametrize(
    "neighbors", [ROOK_CONTIGUITY, QUEEN_CONTIGUITY], ids=["rook", "queen"]
)
def test_grid_values_selections_coordinates_and_raster_chain_rule(neighbors):
    raster = jnp.asarray(np.random.default_rng(91).uniform(0.8, 2.1, (3, 4)))
    edges, costs = [], []
    for i, j in np.ndindex(raster.shape):
        for di, dj in np.asarray(neighbors):
            ni, nj = i + di, j + dj
            if 0 <= ni < 3 and 0 <= nj < 4:
                edges.append((i * 4 + j, ni * 4 + nj))
                costs.append((raster[i, j] + raster[ni, nj]) / 2)
    weights = 1 / np.asarray(costs)
    reference = scipy_distances(weights, edges, 12)
    distance = LCPDistance(ALGORITHM)
    grid = GridGraph(raster, neighbors=neighbors, fun=lambda x, y: 2 / (x + y))
    assert_allclose(distance(grid), reference, rtol=1e-6)
    sources, targets, nodes = (
        np.array([7, 0, 7]),
        np.array([11, 2, 0]),
        np.array([5, 1, 5]),
    )
    assert_allclose(
        distance(grid, sources=jnp.asarray(sources), targets=jnp.asarray(targets)),
        reference[sources[:, None], targets[None, :]],
        rtol=1e-6,
    )
    assert_allclose(
        distance(grid, targets=jnp.asarray(targets)), reference[:, targets], rtol=1e-6
    )
    assert_allclose(
        distance(grid, nodes=jnp.asarray(nodes)),
        reference[nodes[:, None], nodes[None, :]],
        rtol=1e-6,
    )
    assert_allclose(
        distance(
            grid,
            sources=grid.index_to_coord(jnp.asarray(sources)),
            targets=grid.index_to_coord(jnp.asarray(targets)),
        ),
        reference[sources[:, None], targets[None, :]],
        rtol=1e-6,
    )
    assert_allclose(
        distance(grid, nodes=grid.index_to_coord(jnp.asarray(nodes))),
        reference[nodes[:, None], nodes[None, :]],
        rtol=1e-6,
    )
    seed = np.array([[0.5, -1, 2], [-0.7, 0.2, 1.5], [1, -0.3, -0.6]])

    def loss(x):
        graph = GridGraph(x, neighbors=neighbors, fun=lambda u, v: 2 / (u + v))
        return jnp.sum(
            LCPDistance(ALGORITHM)(
                graph, sources=jnp.asarray(sources), targets=jnp.asarray(targets)
            )
            * seed
        )

    edge_derivative = unique_path_derivative(weights, edges, 12, sources, targets, seed)
    cost_derivative = -edge_derivative * weights**2
    expected = np.zeros(12)
    for (u, v), derivative in zip(edges, cost_derivative):
        expected[u] += derivative / 2
        expected[v] += derivative / 2
    actual = jax.jit(jax.grad(loss))(raster)
    assert_allclose(actual, expected.reshape(raster.shape), atol=1e-6, rtol=1e-6)


def test_compiled_reuse_recomputes_switched_optimal_route_and_vmap_gradients():
    edges = np.array([(0, 2), (0, 1), (1, 2)])

    def loss(a):
        return LCPDistance(ALGORITHM)(
            graph_from_edges(a, edges, 3),
            sources=jnp.array([0]),
            targets=jnp.array([2]),
        )[0, 0]

    compiled = jax.jit(jax.value_and_grad(loss))
    inputs = jnp.array([[1, 1, 1], [0.25, 1, 1], [0.5, 1, 1]])
    expected_values = [1, 2, 2]
    expected_gradients = [[-1, 0, 0], [0, -1, -1], [-2, -0.5, -0.5]]
    for weights, value, derivative in zip(inputs, expected_values, expected_gradients):
        actual_value, actual_derivative = compiled(weights)
        assert actual_value == value
        assert_allclose(actual_derivative, derivative)
    values, derivatives = jax.jit(jax.vmap(jax.value_and_grad(loss)))(inputs)
    assert_allclose(values, expected_values)
    assert_allclose(derivatives, expected_gradients)
    assert_allclose(
        jax.jit(jax.grad(lambda a: jax.vmap(loss)(a).sum()))(inputs), expected_gradients
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_near_ties_do_not_admit_nonoptimal_routes(dtype):
    edges = np.array([(0, 2), (0, 1), (1, 2)])
    weights = jnp.array([0.5, 1, 1 - 1e-5], dtype=dtype)

    def loss(a):
        return LCPDistance(ALGORITHM)(
            graph_from_edges(a, edges, 3),
            sources=jnp.array([0]),
            targets=jnp.array([2]),
        )[0, 0]

    assert_allclose(jax.grad(loss)(weights), [-4, 0, 0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad_weight",
    [-1, np.nan, np.inf, np.finfo(np.float32).tiny / 16, np.finfo(np.float32).max],
)
def test_invalid_present_conductances_are_rejected(bad_weight):
    graph = graph_from_edges(jnp.array([bad_weight], dtype=jnp.float32), [(0, 1)], 2)
    with pytest.raises(Exception, match="conductance|positive|finite"):
        LCPDistance(ALGORITHM)(graph, sources=jnp.array([0])).block_until_ready()


def test_present_out_of_bounds_edge_is_rejected():
    graph = graph_from_edges(jnp.array([1.0]), [(0, 2)], 2)
    with pytest.raises(Exception, match="indices|bounds|index"):
        LCPDistance(ALGORITHM)(graph, sources=jnp.array([0])).block_until_ready()


@pytest.mark.parametrize("dtype,large", [(jnp.float32, 2**25), (jnp.float64, 2**54)])
@pytest.mark.parametrize("differentiate", [False, True], ids=["forward", "gradient"])
def test_loss_of_strict_distance_ordering_is_rejected(dtype, large, differentiate):
    weights = jnp.array([1 / large, 1.0], dtype=dtype)

    def loss(a):
        graph = graph_from_edges(a, [(0, 1), (1, 2)], 3)
        return LCPDistance(ALGORITHM)(graph, sources=jnp.array([0])).sum()

    calculation = jax.jit(jax.grad(loss) if differentiate else loss)
    with pytest.raises(Exception, match="strict|ordering|float64"):
        calculation(weights).block_until_ready()


def test_default_backend_uses_dag_tie_behavior():
    edges = np.array([(0, 2), (0, 1), (1, 2)])

    def loss(a):
        return LCPDistance()(
            graph_from_edges(a, edges, 5),
            sources=jnp.array([0]),
            targets=jnp.array([2]),
        )[0, 0]

    assert LCPDistance().algorithm == "bellman-ford"
    assert_allclose(jax.grad(loss)(jnp.array([0.5, 1, 1])), [-2, -0.5, -0.5])


def test_full_length_corridor_forward_and_backward_bounds():
    n = 65
    edges = np.array([(u, u + 1) for u in range(n - 1)])
    weights = jnp.asarray(1 / np.linspace(0.5, 2.0, n - 1))

    def loss(a):
        return LCPDistance(ALGORITHM)(
            graph_from_edges(a, edges, n),
            sources=jnp.array([0]),
            targets=jnp.array([n - 1]),
        )[0, 0]

    value, derivative = jax.jit(jax.value_and_grad(loss))(weights)
    assert_allclose(value, np.sum(1 / weights), rtol=1e-6)
    assert_allclose(derivative, -1 / weights**2, rtol=1e-6)


def test_dag_is_not_a_separate_algorithm_selector():
    with pytest.raises(ValueError, match="algorithm"):
        LCPDistance("bellman-ford-dag")


def test_disconnected_components_and_zero_cotangents():
    edges = np.array([(0, 1), (2, 3)])
    weights = jnp.array([0.5, 0.25])

    def distances(a):
        return LCPDistance()(graph_from_edges(a, edges, 5))

    values, pullback = jax.vjp(distances, weights)
    assert_allclose(values, scipy_distances(weights, edges, 5))
    seed = jnp.zeros((5, 5)).at[0, 1].set(1.5).at[2, 3].set(-0.5)
    assert_allclose(pullback(seed)[0], [-6, 8])
