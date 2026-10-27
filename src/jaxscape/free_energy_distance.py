from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax.experimental.sparse import BCOO

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph
from jaxscape.solvers import BCOOLinearOperator
from jaxscape.solvers.operator import (
    _batched_linear_solve_impl,
    _solver_state,
    _stop_gradient_operator,
)


class FreeEnergyDistance(AbstractDistance):
    r"""Symmetric first-hitting free-energy distance with edge costs $1/a_{ij}$.

    **Attributes:**

    - `theta`: Finite positive scalar inverse temperature. Larger values
      concentrate the path distribution on least-cost paths. Distances are
      differentiable with respect to `theta`.
    - `solver`: Optional Lineax solver, with the same options as
      `ResistanceDistance`. `None` uses a dense solve, suitable for small graphs.

    For conductances $a_{ij}$, let $c_{ij}=1/a_{ij}$ on present edges,
    $P_{ij}=a_{ij}/\sum_k a_{ik}$, and $Z=(I-P\odot e^{-\theta C})^{-1}$.
    The first-hitting partition is $z^h_{ij}=Z_{ij}/Z_{jj}$ and
    $d_{ij}=-(\log z^h_{ij}+\log z^h_{ji})/(2\theta)$.
    For the derivation, see
    [Kivimäki, Shimbo & Saerens](https://arxiv.org/abs/1212.1666) and
    [Françoisse et al.](https://apiacoa.org/publications/2017/francoissekivimakietal2017bag-of-path.pdf).

    The graph must be connected and symmetric, with positive finite
    conductances on present edges. Conductance enters both the edge costs and
    the transition probabilities. As `theta` grows, the distance approaches the
    least-cost distance with edge costs $c_{ij}$. On a graph without self-loops
    and with `m` undirected edges, the small-`theta` limit is `m` times the
    effective resistance. Solver state depends on the conductances and on
    `theta`; rebuild it when either changes.

    **Precision and errors:** float32 and float64 inputs are supported. Entries
    of $Z$ decay exponentially with path cost, and float64 resolves smaller
    entries, hence larger distances and larger `theta`. Each queried logarithm
    is checked against a residual-based error estimate; unresolved entries raise
    an `EquinoxRuntimeError` and are never clipped or substituted. The estimate
    includes heuristic rounding allowances, so accuracy should be validated for
    the intended workload. Checks use `eqx.error_if`; on GPU, a failed check may
    be reported without raising, depending on `EQX_ON_ERROR`.

    **Differentiation:** first-order reverse mode reuses the forward solutions.
    Forward-mode and higher-order derivatives are not validated.

    !!! example

        ```python
        import jax.numpy as jnp
        from jaxscape import FreeEnergyDistance, GridGraph

        raster = jnp.array([[0.8, 1.2], [1.4, 0.9]])
        grid = GridGraph(raster, fun=lambda x, y: (x + y) / 2)
        distances = FreeEnergyDistance(theta=0.5)(grid, nodes=jnp.array([0, 3]))
        ```
    """

    theta: Array
    solver: lx.AbstractLinearSolver | None = None

    def __init__(
        self, theta: float | Array, solver: lx.AbstractLinearSolver | None = None
    ):
        theta = jnp.asarray(theta)
        if theta.ndim != 0 or jnp.issubdtype(theta.dtype, jnp.complexfloating):
            raise ValueError("theta must be a finite positive real scalar.")
        self.theta = eqx.error_if(
            theta,
            ~jnp.isfinite(theta) | (theta <= 0),
            "theta must be a finite positive scalar.",
        )
        self.solver = solver

    def init(self, graph: AbstractGraph) -> Any:
        """Initialize state bound to the current conductances and `theta`.

        Rebuild the state when either changes. Gradients do not propagate
        through solver setup.

        **Arguments:**

        - `graph`: Connected symmetric graph whose conductances define the
          operator together with this distance's `theta`.

        **Returns:**

        Matrix-dependent solver state, or `None` for the dense default.
        """
        if self.solver is None:
            return None
        operator, _ = _operator(graph.get_adjacency_matrix(), self.theta)
        return self.solver.init(
            _stop_gradient_operator(BCOOLinearOperator(operator)), {}
        )

    def init_preconditioner(self, graph: AbstractGraph) -> Any:
        """Initialize preconditioner-only state when supported by the solver.

        For `AMJaxCGSolver`, the preconditioner can be reused with a fresh
        operator, as in `ResistanceDistance`. Other solvers return full solver
        state.

        **Arguments:**

        - `graph`: Connected symmetric graph used to build the preconditioner.

        **Returns:**

        Reusable preconditioner state when supported, otherwise full solver
        state. Returns `None` for the dense default.
        """
        if self.solver is None:
            return None
        initialize = getattr(self.solver, "init_preconditioner", None)
        if initialize is None:
            return self.init(graph)
        operator, _ = _operator(graph.get_adjacency_matrix(), self.theta)
        return initialize(_stop_gradient_operator(BCOOLinearOperator(operator)), {})

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        """Compute free-energy distances between all vertices.

        **Arguments:**

        - `graph`: Connected symmetric graph with `graph.nv` vertices.
        - `state`: Optional solver state for the current operator. `None`
          initializes state automatically; AMJax setup must precede JIT tracing.

        **Returns:**

        Distance array with shape `(graph.nv, graph.nv)` and exact zero diagonal.
        """
        return self.nodes_to_nodes_distance(graph, jnp.arange(graph.nv), state)

    @eqx.filter_jit
    def nodes_to_nodes_distance(
        self, graph: AbstractGraph, nodes: Array, state: Any = None
    ) -> Array:
        """Compute pairwise free-energy distances among selected vertices.

        **Arguments:**

        - `graph`: Connected symmetric graph.
        - `nodes`: Vertex indices with shape `(k,)`, possibly repeated. Use the
          callable distance interface for grid coordinates.
        - `state`: Optional solver state for the current operator. `None`
          initializes state automatically; AMJax setup must precede JIT tracing.

        **Returns:**

        Distance array with shape `(k, k)`, ordered as `nodes`.
        """
        return self._distances(graph, nodes, nodes, True, state)

    @eqx.filter_jit
    def sources_to_targets_distance(
        self, graph: AbstractGraph, sources: Array, targets: Array, state: Any = None
    ) -> Array:
        """Compute free-energy distances from selected sources to targets.

        **Arguments:**

        - `graph`: Connected symmetric graph.
        - `sources`: Source vertex indices with shape `(s,)`.
        - `targets`: Target vertex indices with shape `(t,)`. Sources and
          targets may contain repeated indices. Use the callable distance
          interface for grid coordinates or omitted endpoints.
        - `state`: Optional solver state for the current operator. `None`
          initializes state automatically; AMJax setup must precede JIT tracing.

        **Returns:**

        Distance array with shape `(s, t)`, ordered as `sources` and `targets`.
        """
        return self._distances(graph, sources, targets, False, state)

    def _distances(
        self,
        graph: AbstractGraph,
        sources: Array,
        targets: Array,
        same_nodes: bool,
        state: Any,
    ) -> Array:
        operator, killing = _operator(graph.get_adjacency_matrix(), self.theta)
        nodes = sources if same_nodes else jnp.concatenate([sources, targets])
        selected: Array
        errors: Array
        selected, errors = _selected_inverse(
            operator.data,
            operator.indices,
            operator.shape,
            nodes,
            self.solver,
            jax.tree.map(jax.lax.stop_gradient, state),
            jax.lax.stop_gradient(killing),
        )
        rows = jnp.arange(sources.size)
        cols = jnp.arange(targets.size) + (0 if same_nodes else sources.size)
        diagonal = jnp.diag(selected)
        diagonal_errors = jnp.diag(errors)
        self_pair = sources[:, None] == targets[None, :]
        # Mask self-pairs before logarithms. Their value and derivative are
        # structurally zero, including repeated selections of the same vertex.
        values = tuple(
            jnp.where(self_pair, 1, value)
            for value in (
                diagonal[rows, None],
                diagonal[None, cols],
                selected[rows[:, None], cols[None, :]],
                selected[cols[:, None], rows[None, :]].T,
            )
        )
        entry_errors = tuple(
            jnp.where(self_pair, 0, error)
            for error in (
                diagonal_errors[rows, None],
                diagonal_errors[None, cols],
                errors[rows[:, None], cols[None, :]],
                errors[cols[:, None], rows[None, :]].T,
            )
        )
        logs = tuple(jnp.log(value) for value in values)

        # Near unity, log1p avoids subtracting two almost equal logarithms.
        def difference(log_a: Array, log_b: Array, a: Array, b: Array) -> Array:
            near = jnp.abs(a - b) < 0.5 * b
            ratio_delta = (a - b) / jnp.where(near, b, 1)
            return jnp.where(
                near, jnp.log1p(jnp.where(near, ratio_delta, 0)), log_a - log_b
            )

        numerator = difference(logs[0], logs[3], values[0], values[3]) + difference(
            logs[1], logs[2], values[1], values[2]
        )
        distances = jnp.where(self_pair, 0, numerator / (2 * self.theta))
        return _check_distances(distances, values, entry_errors, self.theta, self_pair)


def _operator(adjacency: BCOO, theta: Array) -> tuple[BCOO, Array]:
    """Assemble H=L_B+diag(kappa) sparsely after aggregating adjacency entries."""
    if (
        len(adjacency.shape) != 2
        or adjacency.shape[0] != adjacency.shape[1]
        or adjacency.n_batch != 0
        or adjacency.n_dense != 0
    ):
        raise ValueError(
            "FreeEnergyDistance requires a square scalar BCOO conductance matrix."
        )
    if not jnp.issubdtype(adjacency.dtype, jnp.floating):
        raise ValueError("FreeEnergyDistance requires floating conductances.")
    raw_valid = jnp.all(
        (adjacency.indices >= 0) & (adjacency.indices < adjacency.shape[0]), axis=1
    )
    data: Array = eqx.error_if(
        adjacency.data,
        jnp.any((~raw_valid) & (adjacency.data != 0)),
        "Present conductances require valid vertex indices.",
    )
    adjacency = BCOO((data, adjacency.indices), shape=adjacency.shape).sum_duplicates(
        nse=adjacency.nse
    )
    valid = jnp.all(
        (adjacency.indices >= 0) & (adjacency.indices < adjacency.shape[0]), axis=1
    )
    data = eqx.error_if(
        adjacency.data,
        jnp.any(valid & (~jnp.isfinite(adjacency.data) | (adjacency.data < 0))),
        "Present conductances must be positive and finite.",
    )
    adjacency = BCOO((data, adjacency.indices), shape=adjacency.shape)
    asymmetry = (adjacency - adjacency.T).sum_duplicates(nse=2 * adjacency.nse)
    data = eqx.error_if(
        data,
        jnp.any(asymmetry.data != 0),
        "FreeEnergyDistance requires symmetric conductances; use a symmetric "
        "GridGraph weight function.",
    )
    present = valid & (data != 0)
    conductance = jnp.where(present, data, 1)
    exponent = -theta / conductance
    weighted = jnp.where(present, conductance * jnp.exp(exponent), 0)
    kill = jnp.where(present, conductance * -jnp.expm1(exponent), 0)
    n = adjacency.shape[0]
    rows, cols = adjacency.indices.T
    off_diagonal = jnp.where(rows != cols, weighted, 0)
    killing = jnp.zeros(n, dtype=kill.dtype).at[rows].add(kill)
    diagonal = jnp.zeros(n, dtype=weighted.dtype).at[rows].add(off_diagonal) + killing
    data = jnp.concatenate([-off_diagonal, diagonal])
    indices = jnp.concatenate(
        [adjacency.indices, jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)]
    )
    data = eqx.error_if(
        data,
        ~jnp.all(jnp.isfinite(data))
        | ~jnp.all(jnp.isfinite(killing))
        | jnp.any(killing <= 0),
        "FreeEnergyDistance operator is not finite or resolvable; require a "
        "nontrivial connected graph and choose a different theta or precision.",
    )
    return BCOO((data, indices), shape=adjacency.shape), killing


@eqx.filter_custom_vjp
def _selected_inverse(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    nodes: Array,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
    killing: Array,
) -> tuple[Array, Array]:
    return _selected_inverse_fwd(
        None, data, indices, shape, nodes, solver, state, killing
    )[0]


def _selected_inverse_fwd(
    perturbed: Any,
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    nodes: Array,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
    killing: Array,
) -> tuple[tuple[Array, Array], Array]:
    del perturbed
    operator = BCOO((data, indices), shape=shape)
    basis = jax.nn.one_hot(nodes, shape[0], dtype=data.dtype).T
    if solver is None:
        potentials = jnp.linalg.solve(operator.todense(), basis)
    else:
        state = _solver_state(operator, solver, None, state)
        potentials = _batched_linear_solve_impl(operator, basis, solver, state=state)
    potentials = eqx.error_if(
        potentials,
        ~jnp.all(jnp.isfinite(potentials)),
        "FreeEnergyDistance solve is not finite; choose a different theta, "
        "precision, or solver.",
    )
    errors = _entry_error(operator, basis, potentials, killing)
    return (potentials[nodes], errors[nodes]), potentials


_selected_inverse.def_fwd(_selected_inverse_fwd)


@_selected_inverse.def_bwd
def _selected_inverse_bwd(
    potentials: Array,
    cotangent: tuple[Array | None, Any],
    perturbed: Any,
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    nodes: Array,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
    killing: Array,
) -> Array | None:
    del data, nodes, solver, state, killing
    if cotangent[0] is None or not perturbed:
        return None
    # For S=E.T H^-1 E and G=dL/dS, H Lambda=E G gives
    # Lambda=X G. Reuse X from the forward solve; G need not be symmetric.
    # Contract only stored H entries: never form the dense -Lambda X.T.
    adjoint = potentials @ cotangent[0]
    valid = jnp.all((indices >= 0) & (indices < shape[0]), axis=1)
    safe = jnp.where(valid[:, None], indices, 0)
    gradient = -jnp.sum(adjoint[safe[:, 0]] * potentials[safe[:, 1]], axis=1)
    return jnp.where(valid, gradient, 0)


def _entry_error(
    operator: BCOO, basis: Array, potentials: Array, killing: Array
) -> Array:
    """Entrywise error envelopes for the computed columns of H^-1.

    H is inverse-positive. If rho >= |b-Hx| and 0 < delta <= Hw for w > 0,
    then |H^-1 b-x| <= H^-1 rho <= w*max(rho/delta). The constant w=1 is the
    fallback; a spatially decaying w is used when its rounded lower margin is
    positive. Rounding allowances for assembly and products are heuristic.
    """
    operator, basis, potentials, killing = jax.tree.map(
        jax.lax.stop_gradient, (operator, basis, potentials, killing)
    )
    residual = basis - operator @ potentials
    valid = jnp.all(
        (operator.indices >= 0) & (operator.indices < operator.shape[0]), axis=1
    )
    absolute = BCOO((jnp.abs(operator.data), operator.indices), shape=operator.shape)
    magnitude = absolute @ jnp.abs(potentials) + jnp.abs(basis)
    count = (
        jnp.zeros(operator.shape[0], dtype=jnp.int32)
        .at[operator.indices[:, 0]]
        .add(valid.astype(jnp.int32))
    )
    eps = jnp.finfo(potentials.dtype).eps
    # Heuristic allowance for sparse accumulation plus operator assembly;
    # the four extra rounding units do not make this a rigorous error bound.
    gamma = (jnp.max(count) + 4) * eps
    rho = jnp.abs(residual) + gamma * magnitude
    constant = jnp.max(rho / killing[:, None], axis=0)
    # A power just below one follows column decay while giving Hw a positive
    # margin. The floor constructs w only; it never changes inverse entries.
    weights = jnp.maximum(potentials, jnp.finfo(potentials.dtype).tiny) ** 0.99
    margin = operator @ weights - gamma * (absolute @ weights)
    usable = jnp.all(
        (potentials > 0) & jnp.isfinite(weights) & jnp.isfinite(margin) & (margin > 0),
        axis=0,
    )
    scale = jnp.max(rho / jnp.where(margin > 0, margin, 1), axis=0)
    weighted = jnp.where(usable[None, :], weights * scale[None, :], jnp.inf)
    return jnp.minimum(constant[None, :], weighted)


def _check_distances(
    distances: Array,
    values: tuple[Array, ...],
    errors: tuple[Array, ...],
    theta: Array,
    self_pair: Array,
) -> Array:
    """Reject unresolved queried logarithms without differentiating the check.

    The relative tolerance 16*sqrt(eps), scaled by max(1, abs(d)), accepts
    well-resolved float32 and float64 solves and rejects ill-conditioned
    small-theta queries and under-resolved distant or large-theta pairs. The
    tolerance is heuristic.
    """
    checked_values: tuple[Array, ...]
    checked_errors: tuple[Array, ...]
    checked_theta: Array
    checked_distances: Array
    checked_values, checked_errors, checked_theta, checked_distances = jax.tree.map(
        jax.lax.stop_gradient, (values, errors, theta, distances)
    )
    invalid = ~jnp.all(jnp.isfinite(checked_distances)) | jnp.any(
        (~self_pair) & (checked_distances <= 0)
    )
    log_error = jnp.zeros_like(checked_distances)
    log_magnitude = jnp.zeros_like(checked_distances)
    for value, error in zip(checked_values, checked_errors):
        resolved = jnp.isfinite(value) & (value > error) & (value > 0)
        invalid = invalid | ~jnp.all(resolved)
        relative = jnp.where(resolved, error / jnp.where(resolved, value, 1), 0)
        log_error = log_error - jnp.log1p(-relative)
        log_magnitude = log_magnitude + jnp.abs(jnp.log(jnp.where(resolved, value, 1)))
    eps = jnp.finfo(distances.dtype).eps
    estimate = (log_error + 4 * eps * log_magnitude) / (
        2 * checked_theta
    ) + 4 * eps * jnp.abs(checked_distances)
    threshold = 16 * jnp.sqrt(eps) * jnp.maximum(1, jnp.abs(checked_distances))
    invalid = invalid | jnp.any(~jnp.isfinite(estimate) | (estimate > threshold))
    return eqx.error_if(
        distances,
        invalid,
        "FreeEnergyDistance cannot resolve the requested logarithms accurately; "
        "choose a different theta, float64 precision, or tighter solver "
        "tolerances.",
    )
