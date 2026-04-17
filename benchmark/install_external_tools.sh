#!/usr/bin/env bash
set -euo pipefail

julia -e 'using Pkg; Pkg.add("Circuitscape")'

echo "Install R plus the CRAN packages 'samc' and 'ResistanceGA' to enable those adapters."
echo "Set CONEFOR_BIN to the Conefor executable to enable the Conefor adapter."
