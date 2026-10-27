# SPDX-FileCopyrightText: 2024-present Victor Boussange <vic.boussange@gmail.com>
#
# SPDX-License-Identifier: MIT
import importlib.metadata

from .connectivity_analysis import (
    ConnectivityAnalysis as ConnectivityAnalysis,
    SensitivityAnalysis as SensitivityAnalysis,
)
from .euclidean_distance import EuclideanDistance as EuclideanDistance
from .free_energy_distance import FreeEnergyDistance as FreeEnergyDistance
from .graph import (
    Graph as Graph,
    GridGraph as GridGraph,
    QUEEN_CONTIGUITY as QUEEN_CONTIGUITY,
    ROOK_CONTIGUITY as ROOK_CONTIGUITY,
)
from .lcp_distance import LCPDistance as LCPDistance
from .resistance_distance import (
    ExactResistance as ExactResistance,
    ResistanceDistance as ResistanceDistance,
    SpielmanApproximation as SpielmanApproximation,
)
from .utils import padding as padding
from .window_operation import WindowOperation as WindowOperation
from .windowed_analysis import WindowedAnalysis as WindowedAnalysis


__version__ = importlib.metadata.version("jaxscape")


def __getattr__(name: str):
    if name == "RSPDistance":
        # ImportError keeps this message for `from jaxscape import RSPDistance`.
        raise ImportError(
            "RSPDistance was removed in JAXScape 0.0.14. FreeEnergyDistance "
            "replaces it: it uses edge costs 1/a_ij and returns symmetric "
            "first-hitting free-energy distances, so its values differ from the "
            "expected path costs that RSPDistance returned. See "
            "https://vboussange.github.io/jaxscape/api/distances/.",
            name=__name__,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
