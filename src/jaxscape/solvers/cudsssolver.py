from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver, RESULTS
from lineax._solver.misc import (
    pack_structures,
    PackedStructures,
    transpose_packed_structures,
    unravel_solution,
)


try:
    from spineax.cudss.solver import solve as cudss_solve

    CUDSS_AVAILABLE = True
except Exception:
    cudss_solve = None
    CUDSS_AVAILABLE = False

_CuDSSSolverState: TypeAlias = tuple[BCOO, PackedStructures]

_CUDSS_SPD_MATRIX_TYPE = 3
_CUDSS_FULL_MATRIX_VIEW = 0


class CuDSSSolver(AbstractLinearSolver):
    """
    A linear solver that uses cuDSS (via spineax) to solve a sparse linear
    system on GPU.

    The solver is configured for symmetric positive definite matrices, so it is
    suitable for Laplacian-based systems used in `JAXScape`.

    !!! example

        ```python
        from jaxscape.solvers import CuDSSSolver

        solver = CuDSSSolver()
        distance = ResistanceDistance(solver=solver)
        dist = distance(grid)
        ```

    !!! warning
        `spineax` with cuDSS support must be installed to use this solver.
    """

    device_id: int = 0

    def __check_init__(self):
        if not CUDSS_AVAILABLE:
            raise ImportError(
                "spineax with cuDSS support is required for CuDSSSolver. "
                "Follow the installation instructions at "
                "https://github.com/johnviljoen/spineax"
            )

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _CuDSSSolverState:
        del options

        A_bcoo = operator.as_matrix()
        packed_structures = pack_structures(operator)
        return A_bcoo, packed_structures

    def _solve(self, A_bcoo: BCOO, b_jax: Array) -> Array:
        if A_bcoo.n_batch > 0:
            A_bcoo = BCOO(
                (A_bcoo.data.squeeze(), A_bcoo.indices.squeeze()), shape=A_bcoo.shape
            )

        if b_jax.ndim == 0:
            raise ValueError("Right-hand side must have at least one dimension.")

        rhs_size = b_jax.shape[0]
        if rhs_size != A_bcoo.shape[0]:
            raise ValueError(
                "The first dimension of b must match the operator dimension."
            )

        A_bcsr = BCSR.from_bcoo(A_bcoo)
        b_flat_batch = b_jax.reshape((rhs_size, -1))

        def solve_rhs(rhs: Array) -> Array:
            solution, _ = cudss_solve(
                rhs,
                A_bcsr.data,
                A_bcsr.indptr,
                A_bcsr.indices,
                device_id=self.device_id,
                mtype_id=_CUDSS_SPD_MATRIX_TYPE,
                mview_id=_CUDSS_FULL_MATRIX_VIEW,
            )
            return solution

        x_flat_batch = jax.vmap(solve_rhs, in_axes=1, out_axes=1)(b_flat_batch)
        return x_flat_batch.reshape(b_jax.shape)

    def compute(
        self,
        state: _CuDSSSolverState,
        b_jax: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options

        A_bcoo, packed_structures = state
        solution = self._solve(A_bcoo, b_jax)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _CuDSSSolverState, options: dict[str, Any]):
        del options

        A_bcoo, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (A_bcoo.T, transposed_packed_structures)
        return transpose_state, {}

    def conj(self, state: _CuDSSSolverState, options: dict[str, Any]):
        del options

        A_bcoo, packed_structures = state
        A_conj = BCOO((jnp.conj(A_bcoo.data), A_bcoo.indices), shape=A_bcoo.shape)
        conj_state = (A_conj, packed_structures)
        return conj_state, {}

    def assume_full_rank(self):
        return True
