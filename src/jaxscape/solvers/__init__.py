"""Linear solvers for jaxscape."""

import warnings

from .amjaxcgsolver import (
    amjax_preconditioner_operator as amjax_preconditioner_operator,
    AMJaxCGSolver as AMJaxCGSolver,
    AMJaxCGState as AMJaxCGState,
    build_amjax_solver as build_amjax_solver,
)
from .cholmodsolver import CholmodSolver as CholmodSolver
from .cudsssolver import CuDSSSolver as CuDSSSolver
from .gridmultigridcgsolver import (
    GridMultigridCGSolver as GridMultigridCGSolver,
    GridMultigridCGState as GridMultigridCGState,
)
from .operator import (
    AbstractReusableStateSolver as AbstractReusableStateSolver,
    BCOOLinearOperator as BCOOLinearOperator,
)
from .pyamgcgsolver import PyAMGCGSolver as PyAMGCGSolver
from .sparsedirectsolver import (
    AbstractSparseDirectSolver as AbstractSparseDirectSolver,
    CSRPattern as CSRPattern,
    SparseDirectState as SparseDirectState,
)


# Former names, kept as deprecated aliases of the classes they name.
_RENAMED = {
    "AMJaxCGSolverState": "AMJaxCGState",
    "GridMultigridSolver": "GridMultigridCGSolver",
    "GridMultigridState": "GridMultigridCGState",
    "PyAMGSolver": "PyAMGCGSolver",
}


def __getattr__(name: str):
    if name in _RENAMED:
        new_name = _RENAMED[name]
        warnings.warn(
            f"`jaxscape.solvers.{name}` is deprecated and will be removed in a "
            f"future release: it was renamed `jaxscape.solvers.{new_name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
