import json
import os
import subprocess
import sys

import numpy as np
from conftest import BENCHMARK


def run_adapter(tmp_path, permeability, settings):
    """Run JAXScape's adapter on E3 under resistance on the CPU; return its map.

    The case has habitat of quality 1 on 30 % of its 32 x 32 pixels, drawn at
    random, the permeability given, and windows of `dependency_range` 10 in
    blocks.
    """
    quality = (np.random.default_rng(1).random((32, 32)) > 0.7).astype(float)
    case = tmp_path / "case"
    case.mkdir()
    np.save(case / "quality.npy", quality)
    np.save(case / "permeability.npy", np.broadcast_to(permeability, quality.shape))
    cell = {
        "question": "E3",
        "distance": "resistance",
        "device": "cpu",
        "settings": settings,
        "parameters": {},
        "windows": {"dependency_range": 10, "coarsening_factor": 0.2},
        "case": str(case),
        "repeats": 0,
        "single_call_after": 300,
        "progress": str(tmp_path / "progress"),
        "output": str(tmp_path / "output.npy"),
        "result": str(tmp_path / "result.json"),
    }
    (tmp_path / "cell.json").write_text(json.dumps(cell))
    subprocess.run(
        [sys.executable, "-m", "adapters.jaxscape", tmp_path / "cell.json"],
        cwd=BENCHMARK,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"},
        check=True,
    )
    return np.load(tmp_path / "output.npy")


def test_normalized_sensitivity_to_resistance_is_one_on_a_uniform_landscape(
    tmp_path,
):
    settings = {
        "dtype": "float32",
        "solver": "GridMultigridSolver(rtol=3e-4)",
        "method": "SpielmanApproximation(epsilon=0.15)",
        "batch_size": 2601,
        "proximity": "lambda d: -d",
        "normalized": True,
    }

    # With K(R) = -R, connectivity on a landscape of permeability c is that of
    # the flat landscape divided by c, so that p^2 dC/dp at c equals dC/dp on
    # the flat landscape at every pixel.
    normalized = run_adapter(tmp_path, 0.25, settings)

    assert normalized.shape == (32, 32)
    np.testing.assert_allclose(normalized, 1, rtol=1e-4)
