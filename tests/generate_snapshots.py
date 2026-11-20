"""Regenerate the stored reference data used by the regression tests.

Run deliberately, and only when a change in behaviour is intended:

    python -m tests.generate_snapshots api      # public API snapshot
    python -m tests.generate_snapshots golden   # inversion golden values
    python -m tests.generate_snapshots evaluation  # evaluation golden values

Every regeneration of golden values must be explained in CHANGELOG.md.
"""
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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
