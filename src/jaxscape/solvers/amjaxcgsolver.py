from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import lineax as lx
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver, RESULTS

from jaxscape.utils import zero_copy_jax_csr_to_scipy_csr


try:
    import pyamg as _pyamg
    from amjax import AMJAXSolver as _AMJAXSolver

    AMJAX_AVAILABLE = True
except ImportError:
    _pyamg = None
    _AMJAXSolver = None
    AMJAX_AVAILABLE = False


class AbstractAMJaxCGSolverState(eqx.Module):
    """Abstract state for `AMJaxCGSolver`."""

    cg_state: eqx.AbstractVar[PyTree[Any] | None]
    preconditioner: eqx.AbstractVar[AbstractLinearOperator]


@final
class AMJaxCGSolverState(AbstractAMJaxCGSolverState):
    """State for `AMJaxCGSolver`.

    `cg_state=None` means that only the AMJax preconditioner has been initialized.
    JAXScape's `linear_solve` wrappers complete such a state against the current
    operator before dispatching to Lineax.
    """

    cg_state: PyTree[Any] | None
    preconditioner: AbstractLinearOperator

    @property
    def has_cg_state(self) -> bool:
        return self.cg_state is not None


def amjax_preconditioner_operator(
    amjax_solver: Any,
    input_structure: PyTree[Any],
    *,
    cycle: str = "V",
) -> lx.FunctionLinearOperator:
    """Wrap an AMJax preconditioner as a Lineax linear operator."""
    preconditioner = amjax_solver.aspreconditioner(cycle=cycle)
    tags = (lx.positive_semidefinite_tag,)
    if getattr(amjax_solver, "symmetric_smoothing", False):
        # Lineax transposes solver states during reverse-mode autodiff. If this
        # tag is omitted, it tries to build a transpose of AMJax's V-cycle via
        # `jax.linear_transpose`, which currently exposes JAX internal cotangent
        # wrappers to AMJax. For symmetric smoothers, AMJax's V-cycle is safe to
        # reuse as its own transpose, so mark that property explicitly.
        tags = tags + (lx.symmetric_tag,)

    def _matvec(rhs: Any) -> Array:
        return preconditioner(rhs)

    return lx.FunctionLinearOperator(
        _matvec,
        input_structure,
        tags=tags,
    )


def build_amjax_solver(
    matrix: BCOO,
    *,
    pyamg_method: Callable | None = None,
    pyamg_kwargs: dict[str, Any] | None = None,
    presmoother: Any | None = None,
    postsmoother: Any | None = None,
    coarse_solver: str = "jacobi",
    coarse_solver_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build an AMJax multigrid hierarchy from a JAX sparse matrix."""
    if not AMJAX_AVAILABLE:
        raise ImportError(
            "AMJaxCGSolver requires amjax and pyamg. "
            "Run with the optional extras enabled, for example: "
            "uv run --extra pyamg --extra amjax ..."
        )

    if pyamg_method is None:
        assert _pyamg is not None
        pyamg_method = _pyamg.smoothed_aggregation_solver

    matrix = _unbatched_bcoo(matrix).sum_duplicates(nse=matrix.nse)
    scipy_matrix = zero_copy_jax_csr_to_scipy_csr(BCSR.from_bcoo(matrix))
    pyamg_hierarchy = pyamg_method(scipy_matrix, **(pyamg_kwargs or {}))

    from_pyamg_kwargs: dict[str, Any] = {
        "coarse_solver": coarse_solver,
    }
    if presmoother is not None:
        from_pyamg_kwargs["presmoother"] = presmoother
    if postsmoother is not None:
        from_pyamg_kwargs["postsmoother"] = postsmoother
    if coarse_solver_kwargs is not None:
        from_pyamg_kwargs["coarse_solver_kwargs"] = coarse_solver_kwargs

    assert _AMJAXSolver is not None
    return _AMJAXSolver.from_pyamg(pyamg_hierarchy, **from_pyamg_kwargs)


class AMJaxCGSolver(AbstractLinearSolver):
    """
    A Lineax CG solver with an AMJax algebraic multigrid preconditioner.

    The PyAMG hierarchy setup is performed once before entering Lineax's traced
    solve path. The resulting AMJax V-cycle is wrapped as a
    `lineax.FunctionLinearOperator` and supplied to `lineax.CG` through the
    `preconditioner` option.

    !!! example

        ```python
        from jaxscape.solvers import AMJaxCGSolver, BCOOLinearOperator, linear_solve

        solver = AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=1_000)
        state = solver.init(BCOOLinearOperator(A), {})
        x = linear_solve(A, b, solver, state=state)
        ```

    !!! warning
        `amjax` and `pyamg` must be installed to use this solver.

        Reverse-mode gradients through solver calls require a symmetric AMJax
        preconditioner. This holds for the default matching pre/post Jacobi
        smoothers, but may not hold for custom non-symmetric smoother choices.
        Such configurations can still be valid for forward solves, but Lineax
        may need to transpose the preconditioner during autodiff.
    """

    rtol: float = 1e-3
    atol: float = 1e-3
    max_steps: int | None = None
    stabilise_every: int | None = 10
    cycle: str = "V"
    pyamg_method: Callable | None = None
    pyamg_kwargs: dict[str, Any] | None = None
    presmoother: Any | None = None
    postsmoother: Any | None = None
    coarse_solver: str = "jacobi"
    coarse_solver_kwargs: dict[str, Any] | None = None

    def __check_init__(self):
        if not AMJAX_AVAILABLE:
            raise ImportError(
                "AMJaxCGSolver requires amjax and pyamg. "
                "Run with the optional extras enabled, for example: "
                "uv run --extra pyamg --extra amjax ..."
            )

        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if isinstance(self.atol, (int, float)) and self.atol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")
        if self.cycle != "V":
            raise NotImplementedError(
                "AMJax currently supports only V-cycle preconditioning."
            )

        if self.pyamg_method is None:
            assert _pyamg is not None
            object.__setattr__(self, "pyamg_method", _pyamg.smoothed_aggregation_solver)
        if self.pyamg_kwargs is None:
            object.__setattr__(self, "pyamg_kwargs", {})

    def _cg_solver(self) -> lx.CG:
        return lx.CG(
            rtol=self.rtol,
            atol=self.atol,
            stabilise_every=self.stabilise_every,
            max_steps=self.max_steps,
        )

    def init_preconditioner(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any] | None = None,
    ) -> AMJaxCGSolverState:
        """Initialize only the AMJax preconditioner for `operator`.

        The returned state intentionally leaves `cg_state` unset, so callers can
        reuse the preconditioner with later operators that have the same
        structure but different values.
        """
        options = {} if options is None else options
        self._check_options(options)
        preconditioner = self._init_preconditioner(operator, options)
        return AMJaxCGSolverState(cg_state=None, preconditioner=preconditioner)

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> AMJaxCGSolverState:
        self._check_options(options)
        state = self.init_preconditioner(operator, options)
        return self.materialize_state(operator, options, state)

    def materialize_state(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any],
        state: Any,
    ) -> Any:
        """Fill in operator-dependent CG state when given preconditioner state."""
        state = self._normalise_state(state)
        if state.has_cg_state:
            return state
        cg_state = self._cg_solver().init(operator, options)
        return AMJaxCGSolverState(
            cg_state=cg_state,
            preconditioner=state.preconditioner,
        )

    def _init_preconditioner(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> AbstractLinearOperator:
        option_preconditioner = options.get("preconditioner")
        if option_preconditioner is not None:
            if not isinstance(option_preconditioner, AbstractLinearOperator):
                raise TypeError(
                    "`preconditioner` must be a lineax.AbstractLinearOperator."
                )
            return option_preconditioner
        matrix = operator.as_matrix()
        if not isinstance(matrix, BCOO):
            matrix = BCOO.fromdense(matrix)
        amjax_solver = build_amjax_solver(
            matrix,
            pyamg_method=self.pyamg_method,
            pyamg_kwargs=self.pyamg_kwargs,
            presmoother=self.presmoother,
            postsmoother=self.postsmoother,
            coarse_solver=self.coarse_solver,
            coarse_solver_kwargs=self.coarse_solver_kwargs,
        )
        preconditioner = amjax_preconditioner_operator(
            amjax_solver,
            operator.in_structure(),
            cycle=self.cycle,
        )
        return preconditioner

    def compute(
        self,
        state: AMJaxCGSolverState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        self._check_options(options)
        state = self._normalise_state(state)
        if state.cg_state is None:
            raise ValueError(
                "AMJaxCGSolver received preconditioner-only state without "
                "operator-specific CG state. Use `jaxscape.solvers.linear_solve`, "
                "`jaxscape.solvers.batched_linear_solve`, or call "
                "`solver.materialize_state(operator, options, state)` before "
                "passing the state to `lineax.linear_solve` directly."
            )
        cg_options = dict(options)
        cg_options["preconditioner"] = state.preconditioner
        return self._cg_solver().compute(state.cg_state, vector, cg_options)

    def transpose(
        self, state: AMJaxCGSolverState, options: dict[str, Any]
    ) -> tuple[AMJaxCGSolverState, dict[str, Any]]:
        self._check_options(options)
        state = self._normalise_state(state)
        if state.cg_state is None:
            raise ValueError(
                "Cannot transpose preconditioner-only AMJaxCGSolver state."
            )
        cg_options = {"preconditioner": state.preconditioner}
        transposed_cg_state, transposed_options = self._cg_solver().transpose(
            state.cg_state,
            cg_options,
        )
        transposed_preconditioner = transposed_options["preconditioner"]
        return (
            AMJaxCGSolverState(
                cg_state=transposed_cg_state,
                preconditioner=transposed_preconditioner,
            ),
            {},
        )

    def conj(
        self, state: AMJaxCGSolverState, options: dict[str, Any]
    ) -> tuple[AMJaxCGSolverState, dict[str, Any]]:
        self._check_options(options)
        state = self._normalise_state(state)
        if state.cg_state is None:
            raise ValueError(
                "Cannot conjugate preconditioner-only AMJaxCGSolver state."
            )
        cg_options = {"preconditioner": state.preconditioner}
        conj_cg_state, conj_options = self._cg_solver().conj(state.cg_state, cg_options)
        conj_preconditioner = conj_options["preconditioner"]
        return (
            AMJaxCGSolverState(
                cg_state=conj_cg_state,
                preconditioner=conj_preconditioner,
            ),
            {},
        )

    def assume_full_rank(self):
        return True

    @staticmethod
    def _check_options(options: dict[str, Any]) -> None:
        option_preconditioner = options.get("preconditioner")
        if option_preconditioner is not None and not isinstance(
            option_preconditioner, AbstractLinearOperator
        ):
            raise TypeError(
                "`preconditioner` must be a lineax.AbstractLinearOperator."
            )

    @staticmethod
    def _normalise_state(state: Any) -> AMJaxCGSolverState:
        if isinstance(state, AMJaxCGSolverState):
            return state
        if isinstance(state, tuple) and len(state) == 2:
            cg_state, preconditioner = state
            return AMJaxCGSolverState(
                cg_state=cg_state,
                preconditioner=preconditioner,
            )
        raise TypeError(
            "AMJaxCGSolver state must be an AMJaxCGSolverState returned by "
            "`init` or `init_preconditioner`."
        )


def _unbatched_bcoo(matrix: BCOO) -> BCOO:
    if matrix.n_batch == 0:
        return matrix

    batch_shape = matrix.shape[: matrix.n_batch]
    if any(size != 1 for size in batch_shape):
        raise ValueError("AMJaxCGSolver supports only singleton matrix batches.")

    # Avoid `squeeze()`: it can accidentally remove length-one matrix axes.
    # Explicit batch indexing drops only BCOO batch dimensions.
    batch_index = (0,) * matrix.n_batch
    return BCOO(
        (matrix.data[batch_index], matrix.indices[batch_index]),
        shape=matrix.shape[-2:],
        indices_sorted=matrix.indices_sorted,
        unique_indices=matrix.unique_indices,
    )
