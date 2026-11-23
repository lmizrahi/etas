"""Seeded simulations are reproduced exactly, and parallel forecasts have
the same form as serial ones.

The golden synthetic catalog was generated with the simulation code before
it was sped up, so it pins the old behaviour: same random numbers, same
catalog. The golden continuation was regenerated when the aftershocks of
observed events started to be drawn only within the forecast window (same
distribution, different random numbers; see CHANGELOG.md). Regenerate with
``python -m tests.generate_snapshots simulation`` only for intended
changes.
"""
import io

import pandas as pd
import pytest

from etas.simulation import ETASSimulation
from tests.conftest import DATA
from tests.generate_snapshots import GOLDEN_SIMULATION_FILES, simulation_golden


def _read(source):
    return pd.read_csv(source, index_col=0,
                       parse_dates=["time", "parent_time"])


def _roundtrip(df):
    buffer = io.StringIO()
    df.to_csv(buffer, float_format="%.17g")
    buffer.seek(0)
    return _read(buffer)


@pytest.fixture(scope="module")
def simulated():
    return simulation_golden()


@pytest.mark.parametrize("name", sorted(GOLDEN_SIMULATION_FILES))
def test_seeded_simulation_matches_golden(simulated, name):
    expected = _read(DATA / GOLDEN_SIMULATION_FILES[name])
    pd.testing.assert_frame_equal(
        _roundtrip(simulated[name]), expected, check_exact=True)


@pytest.fixture(scope="module")
def simulation(short_inversion):
    sim = ETASSimulation(short_inversion)
    sim.prepare()
    return sim


def test_parallel_forecast_has_same_form(simulation):
    kwargs = dict(forecast_n_days=30, n_simulations=7, chunksize=3)
    serial = list(simulation.simulate(**kwargs))
    parallel = list(simulation.simulate(**kwargs, n_jobs=2))
    # chunks after simulations 0, 3, 6
    assert len(serial) == len(parallel) == 3
    for s, p in zip(serial, parallel):
        assert list(s.columns) == list(p.columns)
        assert list(s.dtypes) == list(p.dtypes)
        assert p.index.name == "id"
    ids = [set(chunk["catalog_id"]) for chunk in parallel]
    assert ids[0] <= {0} and ids[1] <= {1, 2, 3} and ids[2] <= {4, 5, 6}
    # the catalogs are within the forecast window
    all_parallel = pd.concat(parallel)
    assert (all_parallel["time"] >= simulation.forecast_start_date).all()
    assert (all_parallel["time"] <= simulation.forecast_end_date).all()


def test_simulate_to_csv_parallel(simulation, tmp_path):
    fn = tmp_path / "forecast.csv"
    simulation.simulate_to_csv(str(fn), forecast_n_days=30,
                               n_simulations=5, chunksize=2, n_jobs=2)
    stored = pd.read_csv(fn, index_col=0)
    assert set(stored["catalog_id"]) <= set(range(5))
    assert {"latitude", "longitude", "magnitude", "time"} <= set(
        stored.columns)


def test_forecast_with_induced_term(short_inversion):
    """Simulation with an additional (induced) background term on a grid,
    as used for models with ``bg_term``."""
    import numpy as np

    lats, lons = np.meshgrid(np.arange(38.6, 39.0, 0.1) + 0.05,
                             np.arange(-123.0, -122.6, 0.1) + 0.05)
    lats, lons = pd.Series(lats.ravel()), pd.Series(lons.ravel())
    term = pd.Series(np.linspace(0.2, 1.0, len(lats)))
    n_induced = 30.0
    sim = ETASSimulation(short_inversion,
                         induced_info=[lats, lons, term, 0.1, 0.1, n_induced])
    sim.prepare()
    forecast = pd.concat(list(sim.simulate(
        30, 20, chunksize=10, info_cols=["is_background", "generation"])))
    induced = forecast[forecast["is_background"].astype(str) == "induced"]
    # about n_induced induced events per catalog, inside the grid cells
    assert 15 < len(induced) / 20 < 45
    # within the grid cells (centers +- half a cell)
    assert induced["latitude"].between(lats.min() - 0.05,
                                       lats.max() + 0.05).all()
    assert induced["longitude"].between(lons.min() - 0.05,
                                        lons.max() + 0.05).all()
    assert (induced["generation"] == 0).all()
    # induced events trigger aftershocks too
    assert (forecast["generation"] > 0).any()
