"""`GridMultigridSolver`, the default solver of `ResistanceDistance`, against
direct solves, on the CPU."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
from jax.experimental.sparse import BCOO
from jaxscape import Graph, GridGraph, ResistanceDistance
from jaxscape.graph import QUEEN_CONTIGUITY, ROOK_CONTIGUITY
from jaxscape.solvers import (
    BCOOLinearOperator,
    GridMultigridSolver,
    gridmultigridsolver,
    GridMultigridState,
)
from jaxscape.solvers.operator import _batched_linear_solve, _linear_solve


def avg(x, y):
    return (x + y) / 2


def landscape(shape, seed=0):
    """Conductances over three decades, with a two-pixel barrier and pixels on
    the floor of windowed analyses, `1e-4`."""
    rng = np.random.default_rng(seed)
    values = 10 ** rng.uniform(-3, 0, shape)
    values[rng.uniform(size=shape) < 0.1] = 1e-4
    values[shape[0] // 3 : shape[0] // 3 + 2, : 2 * shape[1] // 3] = 1e-4
    return jnp.asarray(values)


def system_matrix(conductance, neighbors=ROOK_CONTIGUITY):
    graph = GridGraph(conductance, fun=avg, neighbors=neighbors)
    return ResistanceDistance().system_matrix(graph)


def sparse_solve(matrix, b):
    """Solve `matrix`, a BCOO matrix with padding, with SciPy's sparse LU."""
    n = matrix.shape[0]
    indices = np.asarray(matrix.indices)
    stored = np.all((indices >= 0) & (indices < n), axis=1)
    data = np.asarray(matrix.data, np.float64)[stored]
    reference = scipy.sparse.csc_matrix((data, tuple(indices[stored].T)), (n, n))
    return scipy.sparse.linalg.spsolve(reference, np.asarray(b, np.float64))


@pytest.mark.parametrize(
    "shape, neighbors",
    [
        ((70, 83), ROOK_CONTIGUITY),
        ((65, 64), QUEEN_CONTIGUITY),
        ((2, 1200), ROOK_CONTIGUITY),
        ((1, 3000), ROOK_CONTIGUITY),
    ],
    ids=["rook", "queen", "two-rows", "one-row"],
)
def test_solves_match_direct_solves(shape, neighbors):
    # Two coarse grids, and one for the grid of two rows.
    matrix = system_matrix(landscape(shape), neighbors)
    solver = GridMultigridSolver(rtol=1e-12, max_steps=200)
    state = solver.init(BCOOLinearOperator(matrix), {})
    rng = np.random.default_rng(1)
    b = jnp.asarray(rng.standard_normal((matrix.shape[0], 3)))
    expected = sparse_solve(matrix, b)

    one = jax.jit(lambda b: _linear_solve(matrix, b, solver, state=state))(b[:, 0])
    batch = jax.jit(lambda b: _batched_linear_solve(matrix, b, solver, state=state))(b)

    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(one, expected[:, 0], rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(batch, expected, rtol=0, atol=1e-9 * scale)


@eqx.filter_jit
def grids_one_after_another(state):
    """The stencils of the grids of the multigrid hierarchy of `state`, and the
    interpolation weights from each coarse grid, each built from the grid
    before it in arrays of its own shape."""
    offsets, stencil = state.offsets, state.stencil
    stencils, interpolations = [stencil], []
    while stencil[0].size > 1024:
        coarse = ((stencil[0].shape[0] + 1) // 2, (stencil[0].shape[1] + 1) // 2)
        weights = gridmultigridsolver._interpolation(offsets, stencil)
        stencil = gridmultigridsolver._galerkin(offsets, stencil, weights, coarse)
        offsets = gridmultigridsolver._QUEEN
        stencils.append(stencil)
        interpolations.append(weights)
    return stencils, interpolations


def test_hierarchies_of_four_grids_match_grids_built_one_after_another():
    # A grid of 129 x 131 pixels has three coarse grids, of odd and even
    # sides: 65 x 66, 33 x 33 and 17 x 17. One loop builds the last two, in
    # arrays of the shape of the first.
    matrix = system_matrix(landscape((129, 131)))
    # Measured: 41 iterations.
    solver = GridMultigridSolver(rtol=1e-12, max_steps=60)
    state = solver.init(BCOOLinearOperator(matrix), {})
    hierarchy = gridmultigridsolver._hierarchy(state)

    shapes = [level.A[0].shape for level in hierarchy.levels]
    assert shapes == [(129, 131), (65, 66), (33, 33), (17, 17)]
    A = [level.A for level in hierarchy.levels]
    P = [level.P for level in hierarchy.levels]
    actual = jax.tree.leaves((A, P))
    expected = jax.tree.leaves(grids_one_after_another(state))
    for a, e in zip(actual, expected, strict=True):
        assert a.shape == e.shape
        scale = float(jnp.max(jnp.abs(e)))
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-14 * scale)

    b = np.random.default_rng(1).standard_normal((matrix.shape[0], 2))
    expected = sparse_solve(matrix, b)
    actual = jax.jit(lambda b: _batched_linear_solve(matrix, b, solver, state=state))(
        jnp.asarray(b)
    )
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9 * scale)


def test_float32_solves_converge_in_few_iterations():
    # Float32 solves converge to `rtol` in at most 20 iterations, or raise, and
    # their error against float64 stays small. Measured: 16 to 18 iterations.
    matrix = system_matrix(landscape((101, 120)))
    solver = GridMultigridSolver(rtol=1e-5, max_steps=20)
    matrix32 = BCOO(
        (matrix.data.astype(jnp.float32), matrix.indices), shape=matrix.shape
    )
    state = solver.init(BCOOLinearOperator(matrix32), {})
    b = jnp.asarray(np.random.default_rng(2).standard_normal((matrix.shape[0], 4)))

    actual = _batched_linear_solve(matrix32, b.astype(jnp.float32), solver, state=state)

    assert actual.dtype == jnp.float32
    expected = sparse_solve(matrix, b)
    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert error < 1e-3


def test_float32_solves_keep_their_precision_for_tiny_right_hand_sides():
    # Gradients of proximities that nearly underflow give right-hand sides of
    # about 1e-37 in float32. Right-hand sides below the smallest normal
    # number, 1.2e-38, solve to 0.
    matrix = system_matrix(landscape((20, 21)))
    rng = np.random.default_rng(0)
    b = rng.choice([-1, 1], (420, 3)) * rng.uniform(0.5, 1, (420, 3))
    expected = sparse_solve(matrix, b)
    matrix = BCOO((matrix.data.astype(jnp.float32), matrix.indices), shape=matrix.shape)
    solver = GridMultigridSolver(rtol=1e-6)
    state = solver.init(BCOOLinearOperator(matrix), {})

    def solve(b):
        b = jnp.asarray(b, jnp.float32)
        return _batched_linear_solve(matrix, b, solver, state=state)

    atol = 1e-3 * float(np.max(np.abs(expected)))
    for scale in [1e-37, 1.0, 1e30]:
        np.testing.assert_allclose(solve(scale * b) / scale, expected, atol=atol)
    np.testing.assert_array_equal(solve(1e-40 * b), 0)


def test_right_hand_sides_with_their_own_matrices_solve_one_by_one():
    solver = GridMultigridSolver(rtol=1e-12, max_steps=200)
    matrices = [system_matrix(landscape((9, 10), seed)) for seed in (0, 1)]
    states = [solver.init(BCOOLinearOperator(m), {}) for m in matrices]
    state = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
    b = jnp.asarray(np.random.default_rng(6).standard_normal((2, 90)))

    @jax.vmap
    def solve(data, state, b):
        matrix = BCOO((data, matrices[0].indices), shape=matrices[0].shape)
        return _linear_solve(matrix, b, solver, state=state)

    actual = solve(jnp.stack([m.data for m in matrices]), state, b)

    for matrix, x, rhs in zip(matrices, actual, b):
        np.testing.assert_allclose(x, sparse_solve(matrix, rhs), rtol=1e-8)


def test_unconverged_solves_raise():
    # Grids of up to 1024 pixels are solved exactly, in one step.
    matrix = system_matrix(landscape((64, 65)))
    solver = GridMultigridSolver(rtol=1e-12, max_steps=1)
    state = solver.init(BCOOLinearOperator(matrix), {})
    b = jnp.ones(matrix.shape[0])
    with pytest.raises(Exception, match="maximum number of solver steps"):
        jax.block_until_ready(_linear_solve(matrix, b, solver, state=state))


def test_the_state_of_another_graph_initializes_outside_jit():
    # That of a `GridGraph` builds from its shape and contiguity, also inside
    # `jax.jit`.
    distance = ResistanceDistance(solver=GridMultigridSolver())
    grid = GridGraph(landscape((6, 7)), fun=avg)
    graph = Graph(grid.get_adjacency_matrix())
    with pytest.raises(ValueError, match="outside `jax.jit`"):
        eqx.filter_jit(distance.init)(graph)
    assert isinstance(eqx.filter_jit(distance.init)(grid), GridMultigridState)


def test_a_state_rejects_graphs_of_another_grid():
    # Grids of 6 x 8 and 8 x 6 pixels store as many entries, at other indices.
    distance = ResistanceDistance(solver=GridMultigridSolver())
    state = distance.init(GridGraph(jnp.ones((6, 8)), fun=avg))
    for shape in [(8, 6), (6, 9)]:
        with pytest.raises(Exception, match="sparse indices differ"):
            grid = GridGraph(jnp.ones(shape), fun=avg)
            jax.block_until_ready(distance(grid, nodes=jnp.array([0, 5]), state=state))


def test_reinit_takes_new_values_inside_jit():
    distance = ResistanceDistance(solver=GridMultigridSolver(rtol=1e-12))
    state = distance.init(GridGraph(jnp.ones((16, 18)), fun=avg))
    conductance = landscape((16, 18))
    nodes = jnp.array([0, 100, 287])

    @eqx.filter_jit
    def step(conductance, state):
        grid = GridGraph(conductance, fun=avg)
        state = distance.reinit(grid, state)
        return distance(grid, nodes=nodes, state=state), state

    actual, new_state = step(conductance, state)
    expected = ResistanceDistance(solver=None)(
        GridGraph(conductance, fun=avg), nodes=nodes
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    assert jax.tree.structure(new_state) == jax.tree.structure(state)


# `ResistanceDistance()` solves with `GridMultigridSolver()`. Grids of up to
# 1024 pixels are the coarsest grid of its hierarchy, which a Cholesky
# factorization solves.
def small_landscape(dtype, shape=(24, 40)):
    rng = np.random.default_rng(3)
    return jnp.asarray(10 ** rng.uniform(-2, 0, shape), dtype)


QUERIES = {
    "all-pairs": {},
    "nodes": {"nodes": jnp.array([0, 37, 959, 480, 37])},
    "sources-to-targets": {
        "sources": jnp.array([5, 480]),
        "targets": jnp.array([959, 0, 123, 480]),
    },
}


@pytest.mark.parametrize("query", QUERIES)
def test_the_default_solves_small_float32_grids_as_the_dense_float64_solve(query):
    # Measured: relative errors of the weighted sums of at most 1.5e-6 and of
    # their gradients 3.5e-6. Those of the pseudo-inverse of the all-pairs and
    # nodes-to-nodes queries were 35 % and 42 %.
    assert ResistanceDistance().solver == GridMultigridSolver(rtol=1e-5, max_steps=200)
    weights = np.random.default_rng(4).uniform(size=(960, 960))

    def objective(permeability, distance):
        distances = distance(GridGraph(permeability, fun=avg), **QUERIES[query])
        m, k = distances.shape
        return jnp.sum(jnp.asarray(weights[:m, :k], permeability.dtype) * distances)

    def value_and_grad(dtype, distance):
        return jax.value_and_grad(objective)(small_landscape(dtype), distance)

    actual = value_and_grad(jnp.float32, ResistanceDistance())
    expected = value_and_grad(jnp.float64, ResistanceDistance(solver=None))
    for a, e in zip(actual, expected):
        assert a.dtype == jnp.float32
        assert jnp.linalg.norm(a - e) / jnp.linalg.norm(e) < 1e-5


def test_the_default_runs_eagerly_and_under_jax_transformations():
    # Its state, built from the shape and contiguity of a grid graph, also
    # inside `jax.jit` and for a graph that `eqx.filter_jit` traces whole.
    permeability = small_landscape(jnp.float64, (12, 15))
    nodes = jnp.array([0, 17, 179, 90])
    distance = ResistanceDistance()

    def distances(permeability):
        return distance(GridGraph(permeability, fun=avg), nodes=nodes)

    def objective(permeability, distance=distance):
        return jnp.sum(
            jnp.exp(-distance(GridGraph(permeability, fun=avg), nodes=nodes))
        )

    grid = GridGraph(permeability, fun=avg)
    expected = ResistanceDistance(solver=None)(grid, nodes=nodes)
    for actual in (
        distances(permeability),
        jax.jit(distances)(permeability),
        eqx.filter_jit(distance)(grid, nodes=nodes),
        jax.vmap(distances)(jnp.stack([permeability, permeability]))[1],
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
    expected = jax.grad(objective)(permeability, ResistanceDistance(None))
    for gradient in (
        jax.grad(objective)(permeability),
        jax.jit(jax.grad(objective))(permeability),
    ):
        np.testing.assert_allclose(gradient, expected, rtol=1e-8)


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


def test_the_default_raises_on_disconnected_graphs():
    # A column of zero permeability cuts the grid in two: the grounded
    # Laplacian is singular, and the Cholesky factor of the coarsest grid not
    # finite.
    permeability = small_landscape(jnp.float64).at[:, 20].set(0)
    grid = GridGraph(permeability, fun=jnp.minimum)
    with pytest.raises(Exception, match="non-finite"):
        jax.block_until_ready(ResistanceDistance()(grid, **QUERIES["nodes"]))


@pytest.mark.parametrize("solver", [GridMultigridSolver(), None])
def test_a_grid_of_one_pixel_has_zero_resistance_and_gradient(solver):
    # The pixel has no neighbours: its ground has degree zero.
    distance = ResistanceDistance(solver=solver)
    one = jnp.array([0])
    for query in [{}, {"nodes": one}, {"sources": one, "targets": one}]:

        def objective(permeability):
            distances = distance(GridGraph(permeability, fun=avg), **query)
            return jnp.sum(distances), distances

        (_, distances), gradient = jax.value_and_grad(objective, has_aux=True)(
            jnp.ones((1, 1))
        )
        np.testing.assert_array_equal(distances, [[0.0]])
        np.testing.assert_array_equal(gradient, [[0.0]])
