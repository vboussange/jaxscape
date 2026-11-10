#!/usr/bin/env bash
# Install the toolchains of the external tools and the data, each pinned. It
# runs again without harm.
set -euo pipefail
cd "$(dirname "$0")"

# Julia, from juliaup: the version that resolved julia/Manifest.toml, for every
# `julia` started in this directory, and the packages of the Manifest
# (Circuitscape.jl, ConScape.jl).
julia_version=$(sed -n 's/^julia_version = "\(.*\)"$/\1/p' julia/Manifest.toml)
juliaup add "$julia_version"
juliaup override set "$julia_version"
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# The land cover of Frederick, Maryland, of Omniscape's example, from
# Circuitscape/datasets at a fixed commit, checked against its checksum.
mkdir -p data
curl -fsSL -o data/nlcd_2016_frederick_md.tif https://raw.githubusercontent.com/Circuitscape/datasets/659535620b0fcddb888d618f8bd962927c1422bf/data/nlcd_2016_frederick_md.tif
echo "d76bbfea7d1482364debaf63c9843e4ff0e622c1c779eaa97df0a6fa7597a9fe  data/nlcd_2016_frederick_md.tif" | sha256sum --check
