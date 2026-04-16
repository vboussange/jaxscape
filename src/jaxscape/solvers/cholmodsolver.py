from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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


try:
    from cholespy import CholeskySolverD, CholeskySolverF, MatrixType

    CHOLESPY_AVAILABLE = True
except ImportError:
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

        A_bcoo = operator.as_matrix()
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
        if A_bcoo.n_batch > 0:
            # ugly trick to remove batch dimension
            # specific to BCOO behavior
            A_bcoo = BCOO(
                (A_bcoo.data.squeeze(), A_bcoo.indices.squeeze()), shape=A_bcoo.shape
            )
        A_bcoo = A_bcoo.sum_duplicates()  # required, otherwise cholespy fails

        if b_jax.ndim == 0:
            raise ValueError("Right-hand side must have at least one dimension.")

        operator_size = A_bcoo.shape[0]
        if b_jax.shape[0] != operator_size:
            raise ValueError(
                "The first dimension of b must match the operator dimension."
            )

        b_flat_batch = b_jax.reshape((operator_size, -1))
        use_float64 = b_flat_batch.dtype == jnp.float64 or A_bcoo.data.dtype == jnp.float64
        solver_cls = CholeskySolverD if use_float64 else CholeskySolverF
        buffer_dtype = np.float64 if use_float64 else np.float32

        b_host = np.array(b_flat_batch, dtype=buffer_dtype, order="C", copy=True)
        x_host = np.zeros_like(b_host, order="C")
        a_data_host = np.array(A_bcoo.data, dtype=np.float64, order="C", copy=True)
        row_host = np.array(A_bcoo.indices[:, 0], dtype=np.int32, order="C", copy=True)
        col_host = np.array(A_bcoo.indices[:, 1], dtype=np.int32, order="C", copy=True)

        # Initialize the Cholesky solver with CSR format
        with jax.enable_x64():
            solver = solver_cls(
                operator_size,
                row_host,
                col_host,
                a_data_host,
                MatrixType.COO,
            )
            solver.solve(b_host, x_host)

        x = jnp.asarray(x_host, dtype=b_jax.dtype).reshape(b_jax.shape)
        return x

    def compute(
        self,
        state: _CholmodSolverState,
        b_jax: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        """
        Compute the solution to the linear system.
        """
        A_bcoo, packed_structures = state
        result_shape = jax.ShapeDtypeStruct(b_jax.shape, b_jax.dtype)

        solution = eqx.filter_pure_callback(
            self._compute_host,
            A_bcoo,
            b_jax,
            result_shape_dtypes=result_shape,
            vmap_method="expand_dims",
        )

        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _CholmodSolverState, options: dict[str, Any]):
        A_bcoo, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        A_bcoo_T = A_bcoo.T
        transpose_state = (A_bcoo_T, transposed_packed_structures)
        return transpose_state, {}

    def conj(self, state: _CholmodSolverState, options: dict[str, Any]):
        A_bcoo, _, packed_structures = state
        A_conj = BCOO(
            (jnp.conj(A_bcoo.data), A_bcoo.indices, A_bcoo.indptr), shape=A_bcoo.shape
        )
        conj_state = (A_conj, packed_structures)
        return conj_state, {}

    def assume_full_rank(self):
        return True
