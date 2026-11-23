"""Regenerate the stored reference data used by the regression tests.

Run deliberately, and only when a change in behaviour is intended:

    python -m tests.generate_snapshots api      # public API snapshot
    python -m tests.generate_snapshots golden   # inversion golden values
    python -m tests.generate_snapshots evaluation  # evaluation golden values
    python -m tests.generate_snapshots simulation  # simulation golden values

Every regeneration of golden values must be explained in CHANGELOG.md.
"""
import contextlib
import json
import sys

from tests.api_snapshot import collect
from tests.conftest import DATA, SHORT_INVERSION_CONFIG


def golden_from_calc(calc):
    pij = calc.pij
    return {
        "theta": {k: (None if v is None else float(v))
                  for k, v in calc.theta.items()},
        "beta": float(calc.beta),
        "n_hat": float(calc.n_hat),
        "area": float(calc.area),
        "n_iterations": int(calc.i),
        "n_pairs": int(len(calc.distances)),
        "n_targets": int(len(calc.target_events)),
        "n_sources": int(len(calc.source_events)),
        "sum_spatial_distance_squared":
            float(calc.distances["spatial_distance_squared"].sum()),
        "sum_time_distance": float(calc.distances["time_distance"].sum()),
        "sum_pij": float(pij["Pij"].sum()),
        "sum_p_triggered": float(calc.target_events["P_triggered"].sum()),
        "sum_l_hat": float(calc.source_events["l_hat"].sum()),
    }


def evaluation_scores(tmp_dir):
    """Invert three years, evaluate on the following two (seeded)."""
    import numpy as np
    from etas.evaluation import ETASLikelihoodCalculation
    from etas.inversion import ETASParameterCalculation
    from tests.conftest import EVALUATION_CONFIG

    np.random.seed(42)
    calc = ETASParameterCalculation(dict(EVALUATION_CONFIG))
    calc.prepare()
    calc.invert()
    calc.store_results(str(tmp_dir) + "/")
    meta = json.loads(
        (tmp_dir / "parameters_{}.json".format(calc.id)).read_text())
    ev = ETASLikelihoodCalculation(meta)
    ev.prepare(n=20000)
    ev.evaluate_baseline_poisson_model()
    scores = ev.evaluate()
    return ev, {
        "ETAS": {k: float(v) for k, v in scores.items()},
        "Poisson": {k: float(v) for k, v in ev.Poisson_scores.items()},
        "n_test_events": len(ev.indexes_in_test_window),
    }


GOLDEN_SIMULATION_FILES = {
    "catalog": "golden_simulation_catalog.csv",
    "continuation": "golden_simulation_continuation.csv",
}


@contextlib.contextmanager
def fixed_random_state(seed):
    """Seed numpy's global generator and make the internal reseeding
    (``np.random.seed()`` without argument) a no-op, so that a simulation
    is reproducible."""
    import numpy as np
    original = np.random.seed

    def seed_only_with_argument(*args, **kwargs):
        if args or kwargs:
            original(*args, **kwargs)

    original(seed)
    np.random.seed = seed_only_with_argument
    try:
        yield
    finally:
        np.random.seed = original


def simulation_golden():
    """A one-year synthetic catalog and a 90-day continuation, seeded."""
    import datetime as dt

    import numpy as np
    from shapely.geometry import Polygon

    from etas.simulation import (generate_catalog,
                                 simulate_catalog_continuation)
    from tests.conftest import INPUT

    params = {
        "log10_mu": -6.5, "log10_k0": -2.49, "a": 1.69, "log10_c": -2.95,
        "omega": -0.03, "log10_tau": 3.99, "log10_d": -0.35, "gamma": 1.22,
        "rho": 0.51,
    }
    region = Polygon(np.load(INPUT / "california_shape.npy"))
    start, end = dt.datetime(1999, 1, 1), dt.datetime(2000, 1, 1)
    with fixed_random_state(1):
        catalog = generate_catalog(
            polygon=region, timewindow_start=start, timewindow_end=end,
            parameters=params, mc=3.6, beta_main=np.log(10), delta_m=0.1)
    auxiliary = catalog[["latitude", "longitude", "time", "magnitude"]].copy()
    auxiliary["xi_plus_1"] = 1
    with fixed_random_state(2):
        continuation = simulate_catalog_continuation(
            auxiliary, auxiliary_start=start, auxiliary_end=end,
            polygon=region, simulation_end=end + dt.timedelta(days=90),
            parameters=params, mc=3.55, beta_main=np.log(10),
            background_lats=auxiliary["latitude"],
            background_lons=auxiliary["longitude"],
            background_probs=np.full(len(auxiliary), 0.5),
            gaussian_scale=0.1)
    return {"catalog": catalog, "continuation": continuation}


def main(what):
    if what in ("api", "all"):
        with open(DATA / "public_api.json", "w") as f:
            json.dump(collect(), f, indent=1, sort_keys=True)
    if what in ("golden", "all"):
        from etas.inversion import ETASParameterCalculation
        calc = ETASParameterCalculation(dict(SHORT_INVERSION_CONFIG))
        calc.prepare()
        calc.invert()
        with open(DATA / "golden_inversion_short.json", "w") as f:
            json.dump(golden_from_calc(calc), f, indent=1, sort_keys=True)

    if what in ("evaluation", "all"):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            _, scores = evaluation_scores(Path(d))
        with open(DATA / "golden_evaluation.json", "w") as f:
            json.dump(scores, f, indent=1, sort_keys=True)

    if what in ("simulation", "all"):
        for name, df in simulation_golden().items():
            df.to_csv(DATA / GOLDEN_SIMULATION_FILES[name],
                      float_format="%.17g")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
