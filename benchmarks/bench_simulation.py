"""Time a forecast (catalog continuations) after an inversion.

    PYTHONPATH=. python benchmarks/bench_simulation.py [n] [days] [n_jobs]

(``n`` simulations of ``days`` days each, with ``n_jobs`` processes.)

Runs the inversion on the full example catalog first (about 6 s).

Before the speed-up (main @ 00229cb), on a laptop: 52 ms per catalog for
30 days, 83 ms for 365 days. After: 16 ms and 27 ms; 4 ms for 365 days
with n_jobs=8.
"""
import logging
import sys
import time

from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation
from tests.conftest import FULL_INVERSION_CONFIG

logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    calc = ETASParameterCalculation(dict(FULL_INVERSION_CONFIG))
    calc.prepare()
    calc.invert()
    simulation = ETASSimulation(calc)
    simulation.prepare()
    t0 = time.perf_counter()
    n_events = sum(len(chunk) for chunk in simulation.simulate(
        days, n, n_jobs=n_jobs))
    t1 = time.perf_counter()
    print(f"{n} catalogs, {days} days, n_jobs={n_jobs}: {t1 - t0:.1f} s "
          f"({(t1 - t0) / n * 1e3:.1f} ms per catalog), {n_events} events")
