#!/usr/bin/env bash
# Install the toolchains of the external tools and the data, each pinned. It
# runs again without harm.
set -euo pipefail
cd "$(dirname "$0")"

# Julia, from juliaup: the version that resolved julia/Manifest.toml, for every
# `julia` started in this directory, and the packages of the Manifest
# (Circuitscape.jl, ConScape.jl, Omniscape.jl).
julia_version=$(sed -n 's/^julia_version = "\(.*\)"$/\1/p' julia/Manifest.toml)
juliaup add "$julia_version"
juliaup override set "$julia_version"
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# R 4.1.2: the packages of renv.lock (gdistance, samc), in the renv library of
# this directory, for every `Rscript` started in it. renv installs itself, and
# builds the packages from source; terra needs the GDAL, GEOS and PROJ
# development libraries.
MAKEFLAGS="-j$(nproc)" Rscript -e 'renv::restore(prompt = FALSE)'

# Graphab 3.0.8, and the Java 21 runtime (Temurin) that it needs, in java/,
# each checked against its checksum.
mkdir -p java/jre
curl -fsSL -o java/graphab-3.0.8.jar https://gitlab-mshe.univ-fcomte.fr/api/v4/projects/206/packages/generic/graphab/3.0.8/graphab-3.0.8.jar
echo "9b38cdffc11794f3cebbf1e3846ffc14a1e46acb9102746a446501b1897eca43  java/graphab-3.0.8.jar" | sha256sum --check
curl -fsSL -o java/jre.tar.gz https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.13%2B9/OpenJDK21U-jre_x64_linux_hotspot_21.0.13_9.tar.gz
echo "a535a1df9c6284d9618872984e026ae0352d1f7e9e0605425c0e6114f47351dd  java/jre.tar.gz" | sha256sum --check
tar -xzf java/jre.tar.gz -C java/jre --strip-components=1
rm java/jre.tar.gz

# The land cover of Frederick, Maryland, of Omniscape's example, from
# Circuitscape/datasets at a fixed commit, checked against its checksum.
mkdir -p data
curl -fsSL -o data/nlcd_2016_frederick_md.tif https://raw.githubusercontent.com/Circuitscape/datasets/659535620b0fcddb888d618f8bd962927c1422bf/data/nlcd_2016_frederick_md.tif
echo "d76bbfea7d1482364debaf63c9843e4ff0e622c1c779eaa97df0a6fa7597a9fe  data/nlcd_2016_frederick_md.tif" | sha256sum --check
