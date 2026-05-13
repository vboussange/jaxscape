from collections.abc import Callable
from typing import Any, TypeAlias

import lineax as lx
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver, RESULTS

from jaxscape.utils import zero_copy_jax_csr_to_scipy_csr


try:
    import pyamg
    from amjax import AMJAXSolver

    AMJAX_AVAILABLE = True
except ImportError:
    AMJAX_AVAILABLE = False


_AMJaxCGSolverState: TypeAlias = tuple[PyTree[Any], AbstractLinearOperator]


def amjax_preconditioner_operator(
    amjax_solver: Any,
    input_structure: PyTree[Any],
    *,
    cycle: str = "V",
) -> lx.FunctionLinearOperator:
    """Wrap an AMJax preconditioner as a Lineax linear operator."""
    return lx.FunctionLinearOperator(
        amjax_solver.aspreconditioner(cycle=cycle),
        input_structure,
        tags=(lx.positive_semidefinite_tag,),
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
            "Install AMJax from the local repository with: uv add ../AMJax/"
        )

    if pyamg_method is None:
        pyamg_method = pyamg.smoothed_aggregation_solver

    matrix = _unbatched_bcoo(matrix).sum_duplicates(nse=matrix.nse)
    scipy_matrix = zero_copy_jax_csr_to_scipy_csr(BCSR.from_bcoo(matrix))
    pyamg_hierarchy = pyamg_method(scipy_matrix, **(pyamg_kwargs or {}))

    from_pyamg_kwargs = {
        "coarse_solver": coarse_solver,
    }
    if presmoother is not None:
        from_pyamg_kwargs["presmoother"] = presmoother
    if postsmoother is not None:
        from_pyamg_kwargs["postsmoother"] = postsmoother
    if coarse_solver_kwargs is not None:
        from_pyamg_kwargs["coarse_solver_kwargs"] = coarse_solver_kwargs

    return AMJAXSolver.from_pyamg(pyamg_hierarchy, **from_pyamg_kwargs)


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
    """

    rtol: float = 1e-6
    atol: float = 1e-6
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
                "Install AMJax from the local repository with: uv add ../AMJax/"
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
            object.__setattr__(self, "pyamg_method", pyamg.smoothed_aggregation_solver)
        if self.pyamg_kwargs is None:
            object.__setattr__(self, "pyamg_kwargs", {})

    def _cg_solver(self) -> lx.CG:
        return lx.CG(
            rtol=self.rtol,
            atol=self.atol,
            stabilise_every=self.stabilise_every,
            max_steps=self.max_steps,
        )

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _AMJaxCGSolverState:
        self._check_options(options)

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
        cg_state = self._cg_solver().init(operator, options)
        return cg_state, preconditioner

    def compute(
        self,
        state: _AMJaxCGSolverState,
        b_jax: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        self._check_options(options)
        cg_state, preconditioner = state
        cg_options = dict(options)
        cg_options["preconditioner"] = preconditioner
        return self._cg_solver().compute(cg_state, b_jax, cg_options)

    def transpose(
        self, state: _AMJaxCGSolverState, options: dict[str, Any]
    ) -> tuple[_AMJaxCGSolverState, dict[str, Any]]:
        self._check_options(options)
        cg_state, preconditioner = state
        cg_options = {"preconditioner": preconditioner}
        transposed_cg_state, transposed_options = self._cg_solver().transpose(
            cg_state,
            cg_options,
        )
        transposed_preconditioner = transposed_options["preconditioner"]
        return (transposed_cg_state, transposed_preconditioner), {}

    def conj(
        self, state: _AMJaxCGSolverState, options: dict[str, Any]
    ) -> tuple[_AMJaxCGSolverState, dict[str, Any]]:
        self._check_options(options)
        cg_state, preconditioner = state
        cg_options = {"preconditioner": preconditioner}
        conj_cg_state, conj_options = self._cg_solver().conj(cg_state, cg_options)
        conj_preconditioner = conj_options["preconditioner"]
        return (conj_cg_state, conj_preconditioner), {}

    def assume_full_rank(self):
        return True

    @staticmethod
    def _check_options(options: dict[str, Any]) -> None:
        if "preconditioner" in options:
            raise ValueError(
                "AMJaxCGSolver constructs its own AMJax preconditioner; "
                "do not pass a `preconditioner` option."
            )


def _unbatched_bcoo(matrix: BCOO) -> BCOO:
    if matrix.n_batch == 0:
        return matrix
    return BCOO((matrix.data.squeeze(), matrix.indices.squeeze()), shape=matrix.shape)