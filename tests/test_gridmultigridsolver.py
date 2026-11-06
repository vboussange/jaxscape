"""`GridMultigridSolver` against dense solves, on the CPU."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape import (
    ConnectivityAnalysis,
    Graph,
    GridGraph,
    ResistanceDistance,
    SensitivityAnalysis,
    SpielmanApproximation,
)
from jaxscape.graph import QUEEN_CONTIGUITY, ROOK_CONTIGUITY
from jaxscape.solvers import BCOOLinearOperator, GridMultigridSolver
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


@pytest.mark.parametrize(
    "shape, neighbors",
    [
        ((70, 83), ROOK_CONTIGUITY),
        ((65, 64), QUEEN_CONTIGUITY),
        ((2, 1200), ROOK_CONTIGUITY),
        ((1, 3000), ROOK_CONTIGUITY),
        ((7, 9), ROOK_CONTIGUITY),
    ],
    ids=["rook", "queen", "two-rows", "one-row", "coarsest-only"],
)
def test_solves_match_dense_solves(shape, neighbors):
    # Two coarse grids, one for the grid of two rows, and none for the last.
    matrix = system_matrix(landscape(shape), neighbors)
    solver = GridMultigridSolver(rtol=1e-12, max_steps=200)
    state = solver.init(BCOOLinearOperator(matrix), {})
    rng = np.random.default_rng(1)
    b = jnp.asarray(rng.standard_normal((matrix.shape[0], 3)))
    expected = np.linalg.solve(matrix.todense(), b)

    one = jax.jit(lambda b: _linear_solve(matrix, b, solver, state=state))(b[:, 0])
    batch = jax.jit(lambda b: _batched_linear_solve(matrix, b, solver, state=state))(b)

    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(one, expected[:, 0], rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(batch, expected, rtol=0, atol=1e-9 * scale)


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
    expected = np.linalg.solve(matrix.todense(), b)
    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert error < 1e-3


def test_distances_and_gradients_match_the_dense_solve():
    conductance = landscape((12, 14))
    nodes = jnp.array([0, 30, 167, 30, 100])
    solver = GridMultigridSolver(rtol=1e-12, max_steps=200)
    distance = ResistanceDistance(solver=solver)
    state = distance.init(GridGraph(conductance, fun=avg))

    def objective(conductance, distance, state=None):
        grid = GridGraph(conductance, fun=avg)
        return jnp.sum(jnp.exp(-distance(grid, nodes=nodes, state=state)))

    value, gradient = jax.jit(jax.value_and_grad(objective), static_argnums=1)(
        conductance, distance, state
    )
    expected, expected_gradient = jax.value_and_grad(objective)(
        conductance, ResistanceDistance()
    )
    np.testing.assert_allclose(value, expected, rtol=1e-10)
    scale = float(jnp.max(jnp.abs(expected_gradient)))
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-5 * scale)


def test_spielman_window_queries_and_gradients_match_the_dense_solve():
    method = SpielmanApproximation(epsilon=0.4, seed=3, projection_batch_size=8)
    solver = GridMultigridSolver(rtol=1e-12, max_steps=200)
    conductance = landscape((15, 17))
    sources = jnp.array([3 * 17 + 3, 7 * 17 + 8, 11 * 17 + 13])

    def objective(conductance, distance):
        grid = GridGraph(conductance, fun=avg)
        return jnp.sum(jnp.exp(-distance.sources_to_windows_distance(grid, sources, 7)))

    actual = jax.value_and_grad(objective)(
        conductance, ResistanceDistance(solver=solver, method=method)
    )
    expected = jax.value_and_grad(objective)(
        conductance, ResistanceDistance(method=method)
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-8, atol=1e-12)


def make(kind, quality, permeability, distance):
    return kind(
        quality_raster=quality,
        permeability_raster=permeability,
        distance=distance,
        dependency_range=4,
        batch_size=9,
        coarsening_factor=0.5,
    )


@pytest.mark.parametrize("analysis", ["connectivity", "sensitivity"])
def test_windowed_analyses_carry_the_state_across_tiles(analysis):
    # Sixteen tiles of 3 x 3 block centres, one per batch, each reinitializing
    # the state of the tile before it, from that of a uniform tile.
    rng = np.random.default_rng(4)
    quality = jnp.asarray(rng.uniform(size=(24, 26)))
    permeability = landscape((24, 26), seed=5)

    def run(distance):
        if analysis == "connectivity":
            return make(ConnectivityAnalysis, quality, permeability, distance).run()
        sensitivity = make(SensitivityAnalysis, quality, permeability, distance)
        return sensitivity.run("permeability")

    solver = GridMultigridSolver(rtol=1e-12, max_steps=200)
    actual = run(ResistanceDistance(solver=solver))
    expected = run(ResistanceDistance())
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_float32_solves_keep_their_precision_for_tiny_right_hand_sides():
    # Gradients of proximities that nearly underflow give right-hand sides of
    # about 1e-37 in float32. Right-hand sides below the smallest normal
    # number, 1.2e-38, solve to 0.
    matrix = system_matrix(landscape((20, 21)))
    rng = np.random.default_rng(0)
    b = rng.choice([-1, 1], (420, 3)) * rng.uniform(0.5, 1, (420, 3))
    expected = np.linalg.solve(matrix.todense(), b)
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
        np.testing.assert_allclose(x, np.linalg.solve(matrix.todense(), rhs), rtol=1e-8)


def test_unconverged_solves_raise():
    # Grids of up to 1024 pixels are solved exactly, in one step.
    matrix = system_matrix(landscape((64, 65)))
    solver = GridMultigridSolver(rtol=1e-12, max_steps=1)
    state = solver.init(BCOOLinearOperator(matrix), {})
    b = jnp.ones(matrix.shape[0])
    with pytest.raises(Exception, match="maximum number of solver steps"):
        jax.block_until_ready(_linear_solve(matrix, b, solver, state=state))


def test_matrices_of_other_graphs_are_rejected():
    ring = np.zeros((10, 10))
    for i in range(10):
        ring[i, (i + 1) % 10] = ring[(i + 1) % 10, i] = 1.0
    distance = ResistanceDistance(solver=GridMultigridSolver())
    with pytest.raises(ValueError, match="grid graphs"):
        distance.init(Graph(BCOO.fromdense(jnp.asarray(ring))))


def test_a_state_initializes_outside_jit():
    distance = ResistanceDistance(solver=GridMultigridSolver())
    grid = GridGraph(landscape((6, 7)), fun=avg)
    with pytest.raises(ValueError, match="outside `jax.jit`"):
        eqx.filter_jit(distance.init)(grid)


def test_a_state_rejects_graphs_of_another_grid():
    # Grids of 6 x 8 and 8 x 6 pixels store as many entries, at other indices.
    distance = ResistanceDistance(solver=GridMultigridSolver())
    state = distance.init(GridGraph(jnp.ones((6, 8)), fun=avg))
    for shape in [(8, 6), (6, 9)]:
        with pytest.raises(Exception, match="sparse indices differ"):
            grid = GridGraph(jnp.ones(shape), fun=avg)
            jax.block_until_ready(distance(grid, nodes=jnp.array([0, 5]), state=state))


def test_reinit_builds_the_hierarchy_inside_jit():
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
    expected = ResistanceDistance()(GridGraph(conductance, fun=avg), nodes=nodes)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    assert jax.tree.structure(new_state) == jax.tree.structure(state)
