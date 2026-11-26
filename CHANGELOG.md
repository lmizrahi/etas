# Changelog

## 4.0 (2026-11-23)

A reworked release: faster inversion (about 8 times) and simulation (about
3 times per catalog, plus parallel forecasts), fixes, tests, a
documentation site and new plots. The public API is backwards compatible;
results change only in the cases listed under "Changed behaviour". It also
contains the changes made since 3.2 (January 2023), summarized
[at the end](#since-32-before-the-rework).

New requirements: Python ≥ 3.12 (since February 2025) and Shapely ≥ 2.0.
The package is now on PyPI (`pip install etas`); the `hermes` extra
installs hermes-model and seismostats from PyPI.

### Changed behaviour

These fixes change results in the cases they cover; inversions without fixed
parameters and with a non-vanishing background rate are unaffected.

- **The inversion no longer runs forever when the background rate goes to
  zero.** For log10_mu and log10_iota, the stopping rule now also accepts a
  change of less than 0.001 in the expected *number* of background events.
  The rule is unchanged unless fewer than about 0.43 background events are
  expected. New setting `max_iterations` (default 1000; `None` for no limit)
  stops with a warning as a safety net.
- **Fixed parameters are now exact.** They are removed from the optimization
  and inserted with the given value, instead of being imposed as equality
  constraints. Previously:
  - a fixed value outside the default parameter range was silently replaced
    by the range limit (e.g. `log10_tau = 13` became 12.26);
  - fixed `log10_mu` / `log10_iota` were silently ignored.
  A value outside the default range is now used as given, with a warning.
  The initial values are set to the fixed values, so the first expectation
  step is consistent with them. Inversions with fixed parameters (but no
  alpha constraint) now use the L-BFGS-B optimizer instead of SLSQP, and are
  often faster.
- **Fixed parameters work with free productivity.** Previously the indices
  were shifted (wrong parameters fixed, or `IndexError` for alpha). Fixing
  `log10_k0`, `a` or alpha in free-productivity mode raises a clear
  `ValueError`, since these are derived from the event productivities.
- Fixing alpha together with all of `a`, `rho` and `gamma` to inconsistent
  values raises a `ValueError`.
- **Fixing `log10_tau` to infinity (no taper) now works.** Previously the
  likelihood was NaN for every pair (silently skipped by the sum), so the
  space-time distribution did not enter the fit: on the original code the
  parameters stayed at their initial values. The normalization now uses its
  limit for tau -> infinity, which requires `omega > 0` (enforced). This
  also holds in free-productivity mode, where the expected numbers of
  aftershocks were NaN and the productivities collapsed to zero.
- `max_iterations` must be a positive integer or None (`0` used to return
  no parameters while marking the inversion as done).
- Fixing `log10_mu` together with `free_background=True` raises a
  `ValueError`: the free background replaces the rate per event, so the
  fixed value was reported but not used.

### Performance

Results are unchanged: the inversion agrees with the previous version to
about 1e-5 in every parameter; seeded synthetic catalogs
(`generate_catalog`) are exactly the same, and forecasts have the same
distribution (see below). Timings on the
full example catalog (4.4k events, 268k pairs), on a laptop:

| | Before | After |
|---|---|---|
| Data preparation (`prepare`) | 5.6 s | 0.3 s |
| Inversion (`invert`) | 44.5 s | 5.5 s |
| Forecast, 30 days, per catalog | 52 ms | 16 ms |
| Forecast, 365 days, per catalog | 83 ms | 27 ms |
| Forecast, 365 days, per catalog, 8 processes (`n_jobs=8`) | | 4 ms |

- Maximization step: the likelihood is evaluated on NumPy arrays, with its
  analytic gradient, instead of on DataFrames with a finite-difference
  gradient (about 9 times fewer evaluations).
- Distances: one vectorized calculation instead of one pandas query per
  source event; the table of pairs is identical.
- Simulation: aftershocks are generated on NumPy arrays, each generation
  takes its sources from the previous one instead of querying the whole
  catalog, and the generations are concatenated once.
- Forecasts (`simulate_catalog_continuation`, `ETASSimulation`): for the
  observed events, only the aftershocks that fall in the forecast window
  are drawn, with times from the time kernel restricted to the window.
  Before, all their aftershocks over infinite time were drawn and those
  outside the window discarded (in the example, about 2,900 drawn per
  catalog, a handful kept). The distribution of the simulated catalogs is
  the same, but seeded forecasts now draw different random numbers; the
  column `expected_n_aftershocks` of the observed events in the output of
  `simulate_catalog_continuation` is now the expected number in the
  window. Checked against the previous version on thousands of catalogs:
  number of events per catalog, numbers per generation, times and
  magnitudes of first-order aftershocks, also with an induced term.
- The map projection used for areas is cached.
- Areas are projected with `shapely.transform` instead of the deprecated
  `shapely.ops.transform` (same areas, to the last digit).

### Fixed

- `ETASFitVisualisation`: comparison models with another reference
  magnitude were converted with the wrong signs (and the wrong exponent
  for `k0`), so their lines in the productivity, time and space plots were
  off (in the example configuration, the Swiss model by a factor of about
  8). The conversion is now `convert_reference_magnitude`, tested to keep
  the expected number of aftershocks above the new reference magnitude and
  the size of the aftershock zone; areas also convert with the square of
  the space unit.
- `ETASFitVisualisation` evaluated the kernels at `m_ref` instead of
  `m_ref - delta_m / 2`, the reference magnitude used by the inversion
  (lines about 9% too high in the productivity plot of the example).
- `simulate_catalog_continuation` with an induced term (`induced_lats`
  etc., `ETASSimulation(..., induced_info=...)`): when the forecast window
  had no background events, the induced events got the id NaN, and if
  they had aftershocks, the simulation crashed. They are now numbered after
  the observed events.
- `ETASLikelihoodCalculation.store_results()` crashed with `NameError`
  (missing `import os`) when called without `data_path`.
- Integer initial values (e.g. `"log10_c": -3` in a JSON config) crashed
  the inversion ("Integers to negative integer powers are not allowed");
  integer fixed values above 18 overflowed.
- `runnable_code/predict_etas.py` unpacked the scores dict into its keys.

- **Free-productivity inversion (`free_productivity=True`) crashed** with
  `UnboundLocalError` in `expected_aftershocks_free_prod`. A formatting change
  in February 2025 split a tuple assignment across lines without brackets.
- Free productivity combined with `bg_term` crashed with a `TypeError` in
  `optimize_parameters`.
- `ETASParameterCalculation.prepare()` logged "Preparation already done,
  aborting..." but then repeated the preparation. It now returns, as the
  message says and as `invert()` already did.
- `store_results` wrote the parameter ranges under the wrong names (shifted
  by one, e.g. `log10_k0_range` held the range of `log10_iota`). The ranges
  are now labelled correctly, and `log10_iota_range` was added.
- `download_catalog_sed`: the default `endtime` was the time the module was
  imported, not the time of the call. The default is now `None`, meaning
  "now".
- `etas.simulation` can now be imported without `seismostats`, which is only
  needed by `ETASSimulation.simulate_to_df` and is imported there.
- `etas.evaluation` can now be imported without `matplotlib` (unused imports
  removed).

### Changed

- The likelihood functions `neg_log_likelihood` and
  `neg_log_likelihood_free_prod` no longer add columns (`G`,
  `likelihood_term`) to the DataFrames passed to them. The `G` column of the
  source events is now set once after each optimization step. Stored results
  are unchanged.
- `etas.evaluation`: reformatted and documented; results and printed
  output are unchanged.
- `runnable_code/predict_etas.py` is renamed to `evaluate_etas.py`
  (Sam Stockman).
- Plots (`etas.plots`) in one consistent style (Avenir where installed,
  thin lines, the encoding is the same in every figure: colour identifies
  a fit, whose dots (observed) and line (model) share it, the first fit
  dark gray; comparison models have one colour each; line style never
  carries meaning). Thin vertical lines with vertical labels mark $c$,
  $\tau$ and the size $D$ of the aftershock zone,
  $D^2 = d \, e^{\gamma (m - m_{\mathrm{ref}})}$. The style is applied
  only inside the plotting functions and does not change the user's
  matplotlib settings. Every plot function takes an optional `ax`
  (`axes`), returns the figure and axes, and saves only when a file name
  is given (the default file names are unchanged, so existing calls still
  write the same files).
- `spatial_decay_plot` / `ETASFitVisualisation.space_kernel_plot` draw one
  figure with one panel per magnitude, instead of one file per magnitude
  (whose names ended in `.pdf_mag_<m>.pdf`).
- The whole package passes flake8, which is now enforced in CI.
- Declared dependencies that were used but missing: `joblib`, `pyproj`. New
  extras: `plot` (matplotlib, needed by `etas.plots`) and `test`.
- Replaced deprecated pandas usage (`inplace=True`, the `"d"` time unit). This
  removes about 1,800 deprecation warnings per inversion and keeps the code
  working with pandas 4.

### Added

- `ETASFitVisualisation.from_calculation` (plots directly after an
  inversion, without storing and reading files) and
  `ETASFitVisualisation.overview` (time kernel, productivity law and space
  kernel in one figure).
- `others` argument of the `ETASFitVisualisation` plots: several
  inversions in one figure, each with its dots and line in its own colour.
- Example figures in the documentation (quickstart and `etas.plots`),
  made with `docs/figures/make_figures.py`.
- `n_jobs` argument of `ETASSimulation.simulate`, `simulate_to_csv` and
  `simulate_to_df`: simulates the catalogs of a forecast in parallel
  processes (default 1, as before). The output has the same form and order.
- `etas.__version__`.
- Documentation site (mkdocs, `docs/`): quickstart, model, configuration
  reference, API reference generated from the docstrings. Build with
  `pip install -e ".[docs]"` and `mkdocs serve`. All public functions and
  classes now have docstrings, with units.
- Test suite (`pytest`), inversion benchmark (`benchmarks/`) and continuous
  integration (`.github/workflows/tests.yml`).

### Since 3.2, before the rework

Developed between January 2023 and November 2026 by Leila Mizrahi, Marta
Han, Nicolas Schmid and Sam Stockman; summarized by topic.

- **Inversion**: b-positive and ETAS-positive (as a special case of
  varying completeness); zone-wise b-values and fixing beta to a given
  value; an inner polygon for the target events; fixed parameters and an
  alpha constraint (`constraints`); infinite taper time; 3D inversion with
  a convex-hull region and space units other than km; an additional
  background term for induced seismicity (`bg_term`, iota); reloading an
  inversion without the stored catalogs. Fixes: `m_ref` larger than the
  smallest current completeness (probabilities above 1); sources without
  aftershocks; the likelihood with varying completeness (factor
  `zeta_plus_1`); `P_background` and `P_induced` without sources.
- **Simulation**: background locations from a grid, an induced term,
  `filter_polygon`, continuation of `simulate_to_csv` from where it
  stopped (`i_start`), simulated event ids and info columns, untapered
  times for infinite tau, an exact rejection sampler for the time kernel
  (omega = 0, taper switch), storing only events above `m_thr`.
- **Parameters**: conversion between reference magnitudes and from the
  standard ETAS formulation; branching ratio for infinite tau and for
  alpha ≥ beta.
- **Evaluation**: likelihood of a test window (`etas.evaluation`, Sam
  Stockman).
- **Operational forecasting** (`etas.oef`): hermes entrypoints for
  Switzerland (`suiETAS`), a deep and shallow mix (`genETAS`) and Europe,
  returning `ForecastCatalog` objects (Nicolas Schmid, Marta Han).
- **Packaging**: `pyproject.toml`, Python ≥ 3.12, package data via
  `importlib.resources`.
