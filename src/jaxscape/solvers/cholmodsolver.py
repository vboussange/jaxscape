from typing import Any, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array as JaxArray
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver, RESULTS
from lineax._solver.misc import (
    pack_structures,
    PackedStructures,
    transpose_packed_structures,
    unravel_solution,
)

from ._callback import (
    collapse_broadcasted_bcoo,
    flatten_callback_rhs,
    restore_callback_rhs,
)


try:
    from cholespy import CholeskySolverD, CholeskySolverF, MatrixType  # pyright: ignore[reportMissingImports,reportAttributeAccessIssue]

    CHOLESPY_AVAILABLE = True
except ImportError:
    CholeskySolverD = CholeskySolverF = MatrixType = None
    CHOLESPY_AVAILABLE = False

_CholmodSolverState: TypeAlias = tuple[BCOO, PackedStructures]


class CholmodSolver(AbstractLinearSolver):
    """
    A linear solver that uses CHOLMOD (via cholespy) to solve a sparse linear system.
    Uses direct Cholesky factorization for symmetric positive definite matrices.


    !!! example

        ```python
        from jaxscape.solvers import CholmodSolver

        solver = CholmodSolver()
        distance = ResistanceDistance(solver=solver)
        dist = distance(grid)
        ```

    !!! warning
        `cholespy` must be installed to use this solver.
    """

    def __check_init__(self):
        if not CHOLESPY_AVAILABLE:
            raise ImportError(
                "cholespy is required for CholmodSolver. "
                "Install it with: pip install cholespy"
            )

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _CholmodSolverState:
        del options

        A_bcoo = cast(BCOO, operator.as_matrix())
        packed_structures = pack_structures(operator)
        return A_bcoo, packed_structures

    def _compute_host(self, A_bcoo: BCOO, b_jax: JaxArray) -> JaxArray:
        """
        Solve the linear system using CHOLMOD via cholespy.

        Args:
            A_bcoo: Sparse matrix in BCOO format
            b_jax: Right-hand side vector(s)

        Returns:
            Solution vector(s)
        """
        callback_batched = A_bcoo.n_batch > 0
        A_bcoo = collapse_broadcasted_bcoo(A_bcoo)
        A_bcoo = A_bcoo.sum_duplicates()  # required, otherwise cholespy fails

        operator_size = A_bcoo.shape[0]
        b_flat_batch, rhs_shape = flatten_callback_rhs(
            b_jax,
            operator_size,
            callback_batched,
        )
        use_float64 = (
            b_flat_batch.dtype == jnp.float64 or A_bcoo.data.dtype == jnp.float64
        )
        assert CholeskySolverD is not None
        assert CholeskySolverF is not None
        assert MatrixType is not None
        solver_cls = CholeskySolverD if use_float64 else CholeskySolverF

        # Initialize the Cholesky solver with CSR format
        with jax.enable_x64():
            b_host = jnp.asarray(
                b_flat_batch,
                dtype=jnp.float64 if use_float64 else jnp.float32,
            )
            x_host = jnp.zeros_like(b_host)
            a_data_host = jnp.asarray(A_bcoo.data, dtype=jnp.float64)
            row_host = jnp.asarray(A_bcoo.indices[:, 0], dtype=jnp.int32)
            col_host = jnp.asarray(A_bcoo.indices[:, 1], dtype=jnp.int32)
            solver = solver_cls(
                operator_size,
                row_host,
                col_host,
                a_data_host,
                MatrixType.COO,
            )
            solver.solve(b_host, x_host)

        x = jnp.asarray(x_host, dtype=b_jax.dtype)
        return restore_callback_rhs(x, rhs_shape, callback_batched)

    def compute(
        self,
        state: _CholmodSolverState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        """
        Compute the solution to the linear system.
        """
        del options

        A_bcoo, packed_structures = state
        result_shape = jax.ShapeDtypeStruct(vector.shape, vector.dtype)

        solution = eqx.filter_pure_callback(
            self._compute_host,
            A_bcoo,
            vector,
            result_shape_dtypes=result_shape,
            vmap_method="broadcast_all",
        )

        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _CholmodSolverState, options: dict[str, Any]):
        del options

        A_bcoo, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        A_bcoo_T = A_bcoo.T
        transpose_state = (A_bcoo_T, transposed_packed_structures)
        return transpose_state, {}

    def conj(self, state: _CholmodSolverState, options: dict[str, Any]):
        del options

        A_bcoo, packed_structures = state
        A_conj = BCOO(
            (jnp.conj(A_bcoo.data), A_bcoo.indices),
            shape=A_bcoo.shape,
            indices_sorted=A_bcoo.indices_sorted,
            unique_indices=A_bcoo.unique_indices,
        )
        conj_state = (A_conj, packed_structures)
        return conj_state, {}

    def assume_full_rank(self):
        return True
