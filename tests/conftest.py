"""Precision configuration shared by the numerical test suite."""

import jax


# Native callbacks execute on worker threads, where a thread-local enable_x64
# context is insufficient. Float32 coverage creates explicit float32 inputs.
jax.config.update("jax_enable_x64", True)
