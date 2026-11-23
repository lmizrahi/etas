"""Example figures of the documentation (docs/figures/*.png).

Run from the repository root, after changes to `etas.plots`:

    python docs/figures/make_figures.py

Same inversions as in docs/quickstart.md: the example catalog
(California, m_ref 3.6) for 1981-2007, compared with the Swiss parameters,
and for 1981-1994 and 1994-2007.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from etas.inversion import ETASParameterCalculation  # noqa: E402
from etas.plots import ETASFitVisualisation  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUT = REPO / "input_data"
DPI = 110

CONFIG = {
    "fn_catalog": str(INPUT / "example_catalog.csv"),
    "shape_coords": str(INPUT / "california_shape.npy"),
    "auxiliary_start": "1971-01-01 00:00:00",
    "timewindow_start": "1981-01-01 00:00:00",
    "timewindow_end": "2007-01-01 00:00:00",
    "mc": 3.6,
    "delta_m": 0.1,
    "coppersmith_multiplier": 100,
    "theta_0": {
        "log10_mu": -5.8, "log10_k0": -2.6, "a": 1.8, "log10_c": -2.5,
        "omega": -0.02, "log10_tau": 3.5, "log10_d": -0.85,
        "gamma": 1.3, "rho": 0.66,
    },
    "id": "example",
}

SWISS = {"log10_mu": -6.19, "log10_k0": -2.71, "a": 1.42, "log10_c": -2.68,
         "omega": -0.1, "log10_tau": 3.6, "log10_d": -0.55, "gamma": 0.43,
         "rho": 0.72, "mc": 2.3, "delta_m": 0.1, "beta": 2.74}


def fit(start=None, end=None):
    config = dict(CONFIG)
    if start is not None:
        config.update(id=f"{start}-{end}",
                      timewindow_start=f"{start}-01-01 00:00:00",
                      timewindow_end=f"{end}-01-01 00:00:00")
    calculation = ETASParameterCalculation(config)
    calculation.prepare()
    calculation.invert()
    visualisation = ETASFitVisualisation.from_calculation(calculation)
    visualisation.calculation = calculation
    return visualisation


def main():
    whole = fit()
    with_parameters = ETASFitVisualisation.from_calculation(
        whole.calculation, comparison_parameters={"Swiss ETAS": SWISS})
    halves = [fit("1981", "1994"), fit("1994", "2007")]
    figures = {
        "fit_overview": whole.overview()[0],
        "space_kernel": whole.space_kernel_plot(fn_store=None)[0],
        "fit_with_parameters": with_parameters.overview()[0],
        "fit_comparison": halves[0].overview(others=halves[1:])[0],
    }
    for name, figure in figures.items():
        figure.savefig(OUT / f"{name}.png", dpi=DPI)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
