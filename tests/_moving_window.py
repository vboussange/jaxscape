from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
def moving_window(matrix, window_shape):
    matrix_width = matrix.shape[1]
    matrix_height = matrix.shape[0]

    window_width = window_shape[0]
    window_height = window_shape[1]

    startsx = jnp.arange(matrix_width - window_width + 1)
    startsy = jnp.arange(matrix_height - window_height + 1)
    starts_xy = jnp.dstack(jnp.meshgrid(startsx, startsy)).reshape(-1, 2) # cartesian product => [[x,y], [x,y], ...]

    return vmap(lambda start: jax.lax.dynamic_slice(matrix, (start[1], start[0]), (window_height, window_width)))(starts_xy)

matrix = jnp.asarray([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

moving = moving_window(matrix, (2, 3))
print() # window width = 2, window height = 3