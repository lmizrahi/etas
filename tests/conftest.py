"""Shared fixtures and paths for the test suite."""
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "input_data"
DATA = Path(__file__).resolve().parent / "data"

# Short version of config/invert_etas_config.json, used for fast golden tests.
SHORT_INVERSION_CONFIG = {
    "fn_catalog": str(INPUT / "example_catalog.csv"),
    "auxiliary_start": "1971-01-01 00:00:00",
    "timewindow_start": "1981-01-01 00:00:00",
    "timewindow_end": "1991-01-01 00:00:00",
    "theta_0": {
        "log10_mu": -5.8, "log10_k0": -2.6, "a": 1.8, "log10_c": -2.5,
        "omega": -0.02, "log10_tau": 3.5, "log10_d": -0.85, "gamma": 1.3,
        "rho": 0.66,
    },
    "mc": 3.6,
    "delta_m": 0.1,
    "coppersmith_multiplier": 100,
    "shape_coords": str(INPUT / "california_shape.npy"),
    "id": "short",
}

FULL_INVERSION_CONFIG = dict(
    SHORT_INVERSION_CONFIG,
    timewindow_end="2007-01-01 00:00:00",
    testwindow_end="2021-01-01 00:00:00",
    id="full",
)

# Inversion on 1981-1984, evaluation on 1984-1986 (etas.evaluation).
EVALUATION_CONFIG = dict(
    SHORT_INVERSION_CONFIG,
    timewindow_end="1984-01-01 00:00:00",
    testwindow_end="1986-01-01 00:00:00",
    id="evaluation",
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running test (deselect with -m 'not slow')")


@pytest.fixture(autouse=True)
def _seed():
    """Make tests using the global numpy RNG reproducible."""
    np.random.seed(42)
    yield


@pytest.fixture(scope="session")
def short_inversion():
    """Prepared and inverted calculation on the short example catalog."""
    from etas.inversion import ETASParameterCalculation
    calc = ETASParameterCalculation(dict(SHORT_INVERSION_CONFIG))
    calc.prepare()
    calc.invert()
    return calc


@pytest.fixture
def tmp_cwd(tmp_path):
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)
