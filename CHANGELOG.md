# Changelog

## Unreleased

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

### Fixed

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
- The whole package passes flake8, which is now enforced in CI.
- Declared dependencies that were used but missing: `joblib`, `pyproj`. New
  extras: `plot` (matplotlib, needed by `etas.plots`) and `test`.
- Replaced deprecated pandas usage (`inplace=True`, the `"d"` time unit). This
  removes about 1,800 deprecation warnings per inversion and keeps the code
  working with pandas 4.

### Added

- Documentation site (mkdocs, `docs/`): quickstart, model, configuration
  reference, API reference generated from the docstrings. Build with
  `pip install -e ".[docs]"` and `mkdocs serve`. All public functions and
  classes now have docstrings, with units.

- Test suite (`pytest`), inversion benchmark (`benchmarks/`) and continuous
  integration (`.github/workflows/tests.yml`).
