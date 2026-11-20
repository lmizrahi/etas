# Configuration

An inversion is configured with a dict (or a JSON file loaded into one)
passed to `ETASParameterCalculation`. Examples are in `config/`.

## Data and time windows

| Key | Required | Description |
|---|---|---|
| `fn_catalog` | one of the two | Path to a csv file with columns `id`, `latitude`, `longitude`, `time`, `magnitude` (and `mc_current` for `mc = "var"`). |
| `catalog` | one of the two | The catalog as a pandas DataFrame, with a parsed `time` column. |
| `auxiliary_start` | yes | Start of the auxiliary window: events from here on can trigger, but are not modeled. |
| `timewindow_start` | yes | Start of the primary window: events from here on can trigger and be triggered. |
| `timewindow_end` | yes | End of the primary window (training period). |
| `testwindow_end` | no | End of the test window, used by `etas.evaluation`. |

## Region

| Key | Default | Description |
|---|---|---|
| `shape_coords` | None (whole globe) | Region as a list of `[lat, lon]` pairs, or a path to a `.npy` file with such an array. |
| `inner_shape_coords` | None | Smaller region inside `shape_coords`: only events inside it are targets. |
| `three_dim` | False | Use 3D Euclidean coordinates (columns `x`, `y`, `z`); `shape_coords` is then a set of points whose convex hull is the region. |
| `space_unit_in_meters` | 1000 | Unit of `x`, `y`, `z` for 3D catalogs. |
| `earth_radius` | 6378.1 | In km. |
| `coppersmith_multiplier` | required | Pairs further apart than this multiple of the subsurface rupture length of the source (Wells and Coppersmith, 1994) are treated as unrelated, to keep the problem small. 100 is typical. |

## Magnitudes

| Key | Default | Description |
|---|---|---|
| `mc` | required | Completeness magnitude (a number), `"var"` (per-event completeness in the column `mc_current`), or `"positive"` (ETAS-positive: each event's completeness is the previous magnitude + `delta_m`). |
| `m_ref` | `mc` | Reference magnitude; required for `"var"` and `"positive"`, at most the smallest `mc_current`. |
| `delta_m` | required | Magnitude bin size; magnitudes are rounded to it. |
| `beta` | estimated | Fix the Gutenberg-Richter `beta` (natural-log units), or `"positive"` to estimate it with b-positive. |

## Parameters and the algorithm

| Key | Default | Description |
|---|---|---|
| `theta_0` | random | Initial parameter values, as a dict (see [The model](model.md)). A good guess speeds up convergence. |
| `fixed_parameters` | None | Parameters that are not estimated, e.g. `{"omega": -0.1, "log10_tau": 4}`. Any parameter can be fixed, including `log10_mu`; also `"alpha"` (= a - rho * gamma), to a value or to `"beta"`. Values outside the default range are used as given, with a warning. In free-productivity mode, `log10_k0`, `a` and `alpha` cannot be fixed. |
| `max_iterations` | 1000 | Maximum number of EM iterations; `None` for no limit. |
| `free_background` | False | Spatially variable background estimated from the background probabilities (flETAS). |
| `free_productivity` | False | Individual productivity per event (flETAS); `a` and `log10_k0` are fitted to them afterwards. |
| `bw_sq` | 1 | Squared bandwidth of the Gaussian kernel for `free_background`. |
| `bg_term` | None | Name of a catalog column with a covariate driving an additional background term with rate `log10_iota` (needs `log10_iota` in `theta_0`). |

## Bookkeeping

| Key | Default | Description |
|---|---|---|
| `name` | "NoName ETAS Model" | Name of the model. |
| `id` | random UUID | Used in the names of the output files. |

## Output files

`ETASParameterCalculation.store_results(data_path, store_pij=False,
store_distances=False)` writes, with `<id>` the model id:

| File | Content |
|---|---|
| `parameters_<id>.json` | All settings, `final_parameters`, `beta`, `n_hat`, area, number of iterations, and the names of the other files. Input to `load_calculation`, the evaluation and the plots. |
| `trig_and_bg_probs_<id>.csv` | Target events with `P_background` and `P_triggered`. |
| `sources_<id>.csv` | Source events with the expected (`G`) and estimated (`l_hat`) number of aftershocks. |
| `pij_<id>.csv` | Triggering probability of each source-target pair (`store_pij=True`; needed for plots). |
| `distances_<id>.csv` | Distances of each pair (`store_distances=True`). |
