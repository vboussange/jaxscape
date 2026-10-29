# Environment of the 29 October 2026 run. JAXSCAPE_ROOT is a JAXScape checkout at
# commit 005b92b (0.0.12 source); its .venv holds jax 0.10.0 (CUDA 12), lineax
# 0.1.0, equinox 0.13.6, amjax 0.0.3, pyamg 5.3.0 and cholespy 2.2.0.
JAXSCAPE_ROOT=${JAXSCAPE_ROOT:-$HOME/jaxscape}
export PYTHONPATH=$JAXSCAPE_ROOT/src
export JAX_ENABLE_X64=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=${PY:-$JAXSCAPE_ROOT/.venv/bin/python}
