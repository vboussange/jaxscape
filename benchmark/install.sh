#!/usr/bin/env bash
# Install the pinned toolchains of the benchmark into this directory. Run it
# from any directory.
set -euo pipefail
cd "$(dirname "$0")"

# Python: JAXScape and the Python tools, pinned by uv.lock.
uv sync

# R 4.1.2: the packages of renv.lock, such as gdistance, in the renv library of
# this directory. renv installs itself, and builds the packages from source;
# terra needs the GDAL, GEOS and PROJ development libraries.
MAKEFLAGS="-j$(nproc)" Rscript -e 'renv::restore(prompt = FALSE)'
