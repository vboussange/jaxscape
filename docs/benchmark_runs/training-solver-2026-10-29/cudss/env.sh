# Environment of the 29 October 2026 run: a separate Python 3.12 virtual environment
# (see build.md) with jax 0.10.0 (CUDA 12), lineax 0.1.1, equinox 0.13.8, spineax
# 0.0.5 built from source, nvidia-cudss-cu12 0.8.0.10, cholespy 2.2.0, scipy 1.18.1.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1
PY=${PY:-venv12/bin/python}
