# Tested native backend

The benchmark used JAX-AMG 0.1.4 from the PyPI source release, JAX/JAXlib 0.10.0
with the CUDA 12 plugin, NVIDIA AmgX 2.5.0, and CUDA toolkit 12.9.1. The device
was NVIDIA GeForce RTX 3090 (compute capability 8.6), restricted to GPU 0.
The AmgX source was unmodified at tag `v2.5.0`, commit
`cc1cebdbb32b14d33762d4ddabcb2e23c1669f47`.

JAX-AMG source distribution SHA256:
`b9b5186a4f47ba4a8534b255788166232df5ea2e624d6ec8e6725babf9c9dfaf`.
The CUDA toolkit used nvcc 12.9.86, cudart 12.9.79, and CCCL 12.9.27, obtained
from the [official NVIDIA 12.9.1 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)
and verified against its checksums. Existing CUDA 12.9 wheel libraries supplied
cuBLAS, cuSPARSE, cuSOLVER, and nvJitLink. Build helpers were CMake 3.31.10,
pybind11 3.1.0, and setuptools 84.0.0.

The run-time Python package versions are recorded separately in both result
JSON files. AMJax was 0.0.3. No custom JAX-AMG fork, patched native backend,
private cache manipulation, or fixed hierarchy-refresh schedule was used.

## Build and run configuration

Install a complete compatible CUDA 12.9 toolkit, the documented Python build
prerequisites, and a compiler. Adapt the paths below to local installations;
the benchmark environment used `/tmp/jaxscape-cuda-12.9` and
`/tmp/jaxscape-AMGX`. The shared virtual environment was
`/home/boussang/jaxscape/.venv`.

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/path/to/cuda-12.9
export AMGX_ROOT=/path/to/AMGX
export AMGX_BUILD="$AMGX_ROOT/build"
export JAXAMG_ENABLE_MPI=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$AMGX_BUILD:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

git clone --depth 1 --branch v2.5.0 https://github.com/NVIDIA/AMGX.git "$AMGX_ROOT"
cmake -S "$AMGX_ROOT" -B "$AMGX_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_NO_MPI=ON -DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON
cmake --build "$AMGX_BUILD" --target amgxsh -j 24
python -m pip install --no-deps --no-build-isolation 'jaxamg==0.1.4'
```

CUDA-enabled JAX and the project's benchmark, AMJax, and Cholmod extras must
already be installed in the chosen Python environment. The native library
loader must be able to find `libamgxsh.so` and the matching CUDA libraries.
Use the [upstream JAX-AMG installation guide](https://jx-wang-s-group.github.io/JAX-AMG/install/)
for supported platform-specific setup.

Before benchmark execution, the native backend was validated on GPU 0 and the
solver/inverse correctness tests passed. The final benchmark objective regression
checks both float32 and float64 values and implicit gradients against a dense
reference after parameter changes. These checks do not measure performance.
