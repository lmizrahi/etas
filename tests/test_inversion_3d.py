"""3D inversion (``three_dim``): Euclidean coordinates x, y, z, region given
as points whose convex hull is the volume.

The catalog is the example catalog with x, y in km (from latitude and
longitude) and a random depth z, in a box around the events.
"""
import numpy as np
import pandas as pd
import pytest

from etas.inversion import ETASParameterCalculation, coppersmith
from tests.conftest import INPUT


@pytest.fixture(scope="module")
def config_3d():
    catalog = pd.read_csv(INPUT / "example_catalog.csv", index_col=0,
                          parse_dates=["time"])
    catalog = catalog[(catalog["time"] >= "1978-01-01")
                      & (catalog["time"] < "1984-01-01")].copy()
    lat0, lon0 = 36.0, -119.0
    catalog["x"] = ((catalog["longitude"] - lon0) * 111.32
                    * np.cos(np.radians(lat0)))
    catalog["y"] = (catalog["latitude"] - lat0) * 110.57
    catalog["z"] = np.random.default_rng(0).uniform(0, 20, len(catalog))
    low = catalog[["x", "y", "z"]].min().to_numpy() - 1
    high = catalog[["x", "y", "z"]].max().to_numpy() + 1
    box = np.array([[x, y, z] for x in (low[0], high[0])
                    for y in (low[1], high[1]) for z in (low[2], high[2])])
    return dict(
        catalog=catalog, shape_coords=box, three_dim=True,
        space_unit_in_meters=1000,
        auxiliary_start="1978-01-01 00:00:00",
        timewindow_start="1981-01-01 00:00:00",
        timewindow_end="1984-01-01 00:00:00",
        mc=3.6, delta_m=0.1, coppersmith_multiplier=100,
        theta_0={"log10_mu": -6.5, "log10_k0": -2.6, "a": 1.8,
                 "log10_c": -2.5, "omega": -0.02, "log10_tau": 3.5,
                 "log10_d": -0.85, "gamma": 1.3, "rho": 0.66},
        id="3d")


@pytest.fixture(scope="module")
def calc_3d(config_3d):
    calc = ETASParameterCalculation(dict(config_3d))
    calc.prepare()
    return calc


def test_volume_of_the_box(calc_3d, config_3d):
    box = config_3d["shape_coords"]
    expected = np.prod(box.max(axis=0) - box.min(axis=0))
    assert calc_3d.area == pytest.approx(expected, rel=1e-9)


def test_pairs_match_brute_force(calc_3d):
    """Every source-target pair with the target later than the source and
    within the source's maximum distance, with 3D Euclidean distances."""
    catalog = calc_3d.catalog
    events = catalog[catalog["magnitude"] >= catalog["mc_current"]]
    sources = events
    targets = events[events["time"] >= calc_3d.timewindow_start]
    max_distance_sq = np.square(
        coppersmith(sources["magnitude"], 4)["SSRL"]
        * calc_3d.coppersmith_multiplier).to_numpy()

    s_xyz = sources[["x", "y", "z"]].to_numpy()
    t_xyz = targets[["x", "y", "z"]].to_numpy()
    dist_sq = np.square(s_xyz[:, None, :] - t_xyz[None, :, :]).sum(axis=2)
    later = (targets["time"].to_numpy()[None, :]
             > sources["time"].to_numpy()[:, None])
    i, j = np.nonzero(later & (dist_sq <= max_distance_sq[:, None]))
    expected = pd.Series(
        dist_sq[i, j],
        index=pd.MultiIndex.from_arrays(
            [sources.index[i], targets.index[j]],
            names=["source_id", "target_id"])).sort_index()

    actual = calc_3d.distances["spatial_distance_squared"].sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(),
                               rtol=1e-12)


def test_inversion_runs(config_3d):
    calc = ETASParameterCalculation(dict(config_3d))
    calc.prepare()
    theta = calc.invert()
    for name in ["log10_k0", "a", "log10_c", "omega", "log10_tau",
                 "log10_d", "gamma", "rho"]:
        assert np.isfinite(theta[name]), name
    assert calc.i < calc.max_iterations
    te = calc.target_events
    np.testing.assert_allclose(te["P_triggered"] + te["P_background"], 1,
                               atol=1e-8)
