"""Utilities for linear solves."""

import jax
import lineax as lx
from jax.experimental.sparse import BCOO

class BCOOLinearOperator(lx.MatrixLinearOperator):
    def __check_init__(self):
        if not isinstance(self.matrix, BCOO):
            raise ValueError("The operator must be a BCOO matrix.")

    def mv(self, vector):
        return self.matrix @ vector

@lx.is_positive_semidefinite.register(BCOOLinearOperator)
def _(op):
    return True

def linear_solve(A, b, solver):
    operator = BCOOLinearOperator(A)
    return lx.linear_solve(operator, b, solver=solver).value # works

def batched_linear_solve(A, B, solver):
    return jax.vmap(lambda A, b: linear_solve(A, b, solver), in_axes=(None, 1), out_axes=1)(A, B)
