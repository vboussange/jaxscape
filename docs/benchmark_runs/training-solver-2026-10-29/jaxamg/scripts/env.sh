# Environment of the 29 October 2026 run: the JAXScape environment of ../../amg/env.sh
# plus jaxamg 0.1.4 and a native AmgX 2.5.0 build (revision cc1cebdb) against the
# CUDA 12.9 toolkit, as documented for the reverted integration in #23.
JAXSCAPE_ROOT=${JAXSCAPE_ROOT:-$HOME/jaxscape}
AMGX_BUILD=${AMGX_BUILD:?set AMGX_BUILD to the AmgX build directory}
CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to the CUDA 12.9 toolkit}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1
SP=$JAXSCAPE_ROOT/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:$SP/cublas/lib:$SP/cusparse/lib:$SP/cusolver/lib:$SP/nvjitlink/lib:$SP/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=${PY:-$JAXSCAPE_ROOT/.venv/bin/python}
