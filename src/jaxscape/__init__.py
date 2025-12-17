# SPDX-FileCopyrightText: 2024-present Victor Boussange <vic.boussange@gmail.com>
#
# SPDX-License-Identifier: MIT
import importlib.metadata

from .connectivity_analysis import (
    ConnectivityAnalysis as ConnectivityAnalysis,
    SensitivityAnalysis as SensitivityAnalysis,
)
from .euclidean_distance import EuclideanDistance as EuclideanDistance
from .graph import (
    Graph as Graph,
    GridGraph as GridGraph,
    QUEEN_CONTIGUITY as QUEEN_CONTIGUITY,
    ROOK_CONTIGUITY as ROOK_CONTIGUITY,
)
from .lcp_distance import LCPDistance as LCPDistance
from .resistance_distance import ResistanceDistance as ResistanceDistance
from .rsp_distance import RSPDistance as RSPDistance
from .utils import padding as padding
from .window_operation import WindowOperation as WindowOperation
from .windowed_analysis import WindowedAnalysis as WindowedAnalysis


__version__ = importlib.metadata.version("jaxscape")
