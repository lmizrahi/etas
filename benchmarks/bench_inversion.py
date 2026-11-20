"""Time data preparation and inversion on the example catalog.

    python benchmarks/bench_inversion.py [short|full]

Baseline on main @ 88562dd (Apple M-series laptop):
    full:  prepare 5.2 s, invert 44.0 s, 10 iterations, 267655 pairs
"""
import logging
import sys
import time

from etas.inversion import ETASParameterCalculation
from tests.conftest import FULL_INVERSION_CONFIG, SHORT_INVERSION_CONFIG

logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "full"
    cfg = FULL_INVERSION_CONFIG if which == "full" else \
        SHORT_INVERSION_CONFIG
    t0 = time.perf_counter()
    calc = ETASParameterCalculation(dict(cfg))
    calc.prepare()
    t1 = time.perf_counter()
    calc.invert()
    t2 = time.perf_counter()
    print(f"{which}: prepare {t1 - t0:.1f} s, invert {t2 - t1:.1f} s, "
          f"{calc.i} iterations, {len(calc.distances)} pairs")
