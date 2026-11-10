#!/usr/bin/env bash
# Install the toolchains of the external tools, each pinned. It runs again
# without harm.
set -euo pipefail
cd "$(dirname "$0")"

# Julia, from juliaup: the version that resolved julia/Manifest.toml, for every
# `julia` started in this directory, and the packages of the Manifest
# (Circuitscape.jl, ConScape.jl).
julia_version=$(sed -n 's/^julia_version = "\(.*\)"$/\1/p' julia/Manifest.toml)
juliaup add "$julia_version"
juliaup override set "$julia_version"
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
