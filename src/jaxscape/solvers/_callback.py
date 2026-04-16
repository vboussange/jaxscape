import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO


def collapse_broadcasted_bcoo(A_bcoo: BCOO) -> BCOO:
    """Recover the original sparse operator from a broadcasted callback input."""
    if A_bcoo.n_batch == 0:
        return A_bcoo

    batch_index = (0,) * A_bcoo.n_batch
    data = A_bcoo.data[batch_index]
    indices = A_bcoo.indices[batch_index]
    return BCOO(
        (data, indices),
        shape=A_bcoo.shape,
        indices_sorted=A_bcoo.indices_sorted,
        unique_indices=A_bcoo.unique_indices,
    )


def flatten_callback_rhs(
    b_jax: Array,
    operator_size: int,
    callback_batched: bool,
) -> tuple[Array, tuple[int, ...]]:
    """Normalize RHS inputs so the operator axis is first on the host."""
    if b_jax.ndim == 0:
        raise ValueError("Right-hand side must have at least one dimension.")

    if callback_batched:
        if b_jax.shape[-1] != operator_size:
            raise ValueError(
                "The last dimension of b must match the operator dimension."
            )
        return jnp.moveaxis(b_jax, -1, 0).reshape((operator_size, -1)), b_jax.shape

    if b_jax.shape[0] != operator_size:
        raise ValueError("The first dimension of b must match the operator dimension.")

    return b_jax.reshape((operator_size, -1)), b_jax.shape


def restore_callback_rhs(
    flat_solution: Array,
    rhs_shape: tuple[int, ...],
    callback_batched: bool,
) -> Array:
    """Restore the solver output to the callback-visible RHS layout."""
    if callback_batched:
        reshaped = flat_solution.reshape((flat_solution.shape[0],) + rhs_shape[:-1])
        return jnp.moveaxis(reshaped, 0, -1)

    return flat_solution.reshape(rhs_shape)