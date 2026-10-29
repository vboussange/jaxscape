# Building spineax against CUDA 12

The spineax 0.0.5 wheel on PyPI targets CUDA 13 and Python 3.12+. This run kept
the CUDA 12 JAX stack of the other experiments, so spineax was built from source
at commit `97dae03f54c3d0698c9a714471cc64f471b5aedd` (23 July 2026).

1. Create a Python 3.12 environment and install `jax[cuda12]==0.10.0`, `lineax`,
   `equinox`, `cholespy` and `scipy`.
2. Install the build tools and the cuDSS runtime: `cmake>=3.26` (3.31.10 here),
   `nanobind` (3.1.0), `scikit-build-core` (1.1.1) and `nvidia-cudss-cu12==0.8.0.10`.
   spineax does not support cuDSS 1.0 yet (upstream issues #29 and #30).
3. Build and install spineax from the source checkout with `--no-deps`, so that
   its CUDA 13 requirements do not replace the CUDA 12 JAX stack. The C++ build
   takes about 45 s and needs no `nvcc`.
4. Check the build with spineax's own test suite: all 63 tests passed with
   jax 0.10.0 on an RTX 3090.
