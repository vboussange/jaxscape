"""Linear solvers for jaxscape."""

from .amjaxcgsolver import AMJaxCGSolver as AMJaxCGSolver
from .cholmodsolver import CholmodSolver as CholmodSolver
from .cudsssolver import CuDSSSolver as CuDSSSolver
from .gridmultigridsolver import GridMultigridSolver as GridMultigridSolver
from .operator import (
    AbstractReusableStateSolver as AbstractReusableStateSolver,
    BCOOLinearOperator as BCOOLinearOperator,
)
from .pyamgsolver import PyAMGSolver as PyAMGSolver
from .sparsedirectsolver import (
    AbstractSparseDirectSolver as AbstractSparseDirectSolver,
    SparseDirectState as SparseDirectState,
)
from .sparseiterativesolver import (
    AbstractSparseIterativeSolver as AbstractSparseIterativeSolver,
    SparseIterativeState as SparseIterativeState,
)
from .sparsesolver import (
    AbstractSparseSolver as AbstractSparseSolver,
    AbstractSparseState as AbstractSparseState,
    CSRPattern as CSRPattern,
)


_REMOVED = {
    "AMJaxCGSolverState": "SparseIterativeState",
    "GridMultigridState": "SparseIterativeState",
    "amjax_preconditioner_operator": "AMJaxCGSolver",
    "build_amjax_solver": "AMJaxCGSolver",
}


def __getattr__(name: str):
    if name in _REMOVED:
        # ImportError keeps this message for `from jaxscape.solvers import ...`.
        raise ImportError(
            f"{name} was removed in JAXScape 0.0.18: the solvers of the iterative "
            f"family share one state type and conjugate gradients. Use "
            f"`{_REMOVED[name]}`. See "
            "https://vboussange.github.io/jaxscape/api/linear_solvers/.",
            name=__name__,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
