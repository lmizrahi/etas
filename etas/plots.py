r"""Plots of the fit of an ETAS model.

Requires the ``plot`` extra (matplotlib).

The plots compare the fitted kernels with the "observed" branching
structure of the inversion: the pairs of a possible triggering event $e_i$
and a possible aftershock $e_j$, weighted by the probability $p_{ij}$ that
$e_j$ was triggered by $e_i$ (times the `etas.inversion.observation_factor`
$1 + \zeta_j$ for varying completeness).

- `temporal_decay_plot`: distribution of the time between $e_i$ and its
  aftershocks, and the time kernel.
- `productivity_plot`: number of direct aftershocks against the magnitude
  of $e_i$, and the productivity law.
- `spatial_decay_plot`: distribution of the distance between $e_i$ and its
  aftershocks, and the space kernel, one panel per magnitude.
- `ETASFitVisualisation`: all of these for one inversion, from stored
  results or (`ETASFitVisualisation.from_calculation`) directly from an
  `etas.inversion.ETASParameterCalculation`, optionally together with
  other inversions (``others``) and other models (comparison parameters).

Every function takes an optional ``ax`` (or ``axes``) to draw into,
returns the figure and axes, and saves only when ``file_name`` is given.

The encoding is the same in all figures. Colour identifies a fit: its dots
(observed) and its line (model) have the same colour; the first fit is
dark gray, further fits and comparison models have one colour each. Dots
are observations, lines are models, and all lines are solid. Thin pale
vertical lines mark the Omori offset $c$, the taper time $\tau$ and the
size $D$ of the aftershock zone of an event of magnitude $m$, with
$D^2 = d \, e^{\gamma (m - m_{\mathrm{ref}})}$.
"""
import os

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "etas.plots needs matplotlib; install the 'plot' extra: "
        "pip install etas[plot]") from e

from etas import _figstyle as fs
from etas.inversion import (expected_aftershocks, parameter_dict2array,
                            upper_gamma_ext)

SINGLE = (6.4, 4.8)


class _Fit:
    """One model to draw: a line from its parameters and, if ``pij`` is
    given, dots from the observed branching structure.

    ``area_factor`` converts its squared distances (and ``d``) to the space
    unit of the figure.
    """

    def __init__(self, label, color, params, pij=None, ref=None,
                 catalog=None, delta_m=None, area_factor=1.0):
        self.label, self.color, self.params = label, color, params
        self.pij, self.ref, self.catalog = pij, ref, catalog
        self.delta_m, self.area_factor = delta_m, area_factor

    @property
    def weights(self):
        return self.pij["Pij"] * self.pij["zeta_plus_1"]

    @property
    def c(self):
        return np.power(10, self.params["log10_c"])

    @property
    def tau(self):
        return np.power(10, self.params["log10_tau"])

    @property
    def omega(self):
        return self.params["omega"]

    @property
    def d(self):
        return np.power(10, self.params["log10_d"]) * self.area_factor

    @property
    def gamma(self):
        return self.params["gamma"]

    @property
    def rho(self):
        return self.params["rho"]


def _comparisons(comparison_params, keys):
    """Comparison models of the low-level functions (natural parameters)
    as `_Fit` objects, coloured after the fitted model."""
    colors = fs.colors(len(comparison_params) + 1)[1:]
    out = []
    for color, (label, p) in zip(colors, comparison_params.items()):
        params = dict(p)
        for key in keys:
            if key in params:
                params["log10_" + key] = np.log10(params.pop(key))
        out.append(_Fit(label, color, params))
    return out


def _new_axes(ax, figsize=SINGLE):
    """Use ``ax`` or create a figure with one axes."""
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    return fig, ax, True


def _finish(fig, file_name, own_figure):
    """Save if ``file_name`` is given; close figures created and saved here
    (as earlier versions did), so that loops do not accumulate figures."""
    if file_name is not None:
        fig.savefig(file_name, dpi=200)
        if own_figure:
            plt.close(fig)


def _legend_entries(fits, comparisons):
    """Handles and labels: a dot on a line for each fit (observed and
    model in its colour), a line for each comparison model."""
    handles = [Line2D([], [], color=f.color, lw=fs.LINE, marker="o",
                      markersize=np.sqrt(fs.DOT), markeredgewidth=0)
               for f in fits]
    handles += [Line2D([], [], color=f.color, lw=fs.LINE)
                for f in comparisons]
    labels = [f.label for f in fits] + [f.label for f in comparisons]
    return handles, labels


LEGEND_TITLE = "dots: observed, lines: model"


def _legend(ax, fits, comparisons):
    handles, labels = _legend_entries(fits, comparisons)
    legend = ax.legend(handles, labels, title=LEGEND_TITLE, loc="best",
                       alignment="left")
    legend.get_title().set_color(fs.MUTED)


def _side(i):
    """Label side of the reference lines of the i-th fit (alternating, so
    that close values of two fits do not overlap)."""
    return "left" if i % 2 == 0 else "right"


def time_scaling_factor(
        c: float,
        tau: float,
        omega: float,
        t0: float = None,
        t1: float = None):
    r"""Integral of the time kernel between two time delays.

    $$
    \int_{10^{t_0}}^{10^{t_1}} \frac{e^{-t / \tau}}{(t + c)^{1 + \omega}}
    \, dt,
    $$

    used to normalize the time kernel over the range of the plot.

    Parameters
    ----------
    c, tau : float
        Omori offset $c$ and taper time $\tau$, in days.
    omega : float
        Omori exponent minus 1.
    t0, t1 : float, optional
        log10 of the smallest and largest time delay (days); default 0
        and infinity.

    Returns
    -------
    float
    """
    factor = - np.exp(c / tau) * np.power(tau, -omega)
    if t1 is not None:
        factor_gamma = upper_gamma_ext(-omega,
                                       (c + np.power(10, t1)) / tau)
    else:
        factor_gamma = upper_gamma_ext(-omega, c / tau)

    if t0 is not None:
        factor_gamma -= upper_gamma_ext(-omega,
                                        (c + np.power(10, t0)) / tau)
    return factor * factor_gamma


def _time_plot(fits, comparisons, file_name, ax):
    with fs.style():
        fig, ax, own = _new_axes(ax)
        time_deltas = np.concatenate(
            [f.pij["time_distance"].to_numpy() for f in fits])
        min_t_dist, max_t_dist = np.log10(
            np.min(time_deltas)), np.log10(np.max(time_deltas))
        time_bins = np.logspace(np.floor(min_t_dist * 100)
                                / 100, np.ceil(max_t_dist * 100) / 100)
        tmid = (time_bins[:-1] + time_bins[1:]) / 2

        def kernel(f):
            decay = np.exp(-tmid / f.tau) / (tmid + f.c) ** (1 + f.omega)
            return decay / time_scaling_factor(
                f.c, f.tau, f.omega, min_t_dist, max_t_dist)

        for f in comparisons:
            ax.plot(tmid, kernel(f), color=f.color, label=f.label, zorder=3)
        for f in fits:
            ax.plot(tmid, kernel(f), color=f.color, label=f.label, zorder=4)
            counts, _ = np.histogram(f.pij["time_distance"], bins=time_bins,
                                     weights=f.weights, density=True)
            ax.scatter(tmid[counts > 0], counts[counts > 0], s=fs.DOT,
                       color=f.color, label=f"observed: {f.label}",
                       zorder=5, linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("time since the triggering event [days]")
        ax.set_ylabel("probability density [1/day]")
        for i, f in enumerate(fits):
            fs.reference_line(
                ax, f.c, f"c = 10$^{{{np.log10(f.c):.2f}}}$ days",
                f.color, _side(i), where="bottom")
            if np.isfinite(f.tau):
                fs.reference_line(
                    ax, f.tau, r"$\tau$ = "
                    f"10$^{{{np.log10(f.tau):.2f}}}$ days", f.color,
                    _side(i))
        _legend(ax, fits, comparisons)
        _finish(fig, file_name, own)
    return fig, ax


def temporal_decay_plot(
        p_mat: pd.DataFrame,
        tau: float,
        c: float,
        omega: float,
        label: str = None,
        comparison_params: dict = {},
        file_name: str = "time_kernel_fit.pdf",
        ax=None):
    r"""Fit of the time kernel to the observed branching structure.

    Dots: distribution of the time delays $t_j - t_i$ of all pairs,
    weighted by $p_{ij} (1 + \zeta_j)$, in logarithmic bins. Line: the time
    kernel $e^{-t/\tau} / (t + c)^{1 + \omega}$, normalized over the same
    range. Thin vertical lines mark $c$ and $\tau$.

    To show several inversions in one figure, use
    `ETASFitVisualisation.time_kernel_plot` with ``others``.

    Parameters
    ----------
    p_mat : pd.DataFrame
        Pairs, with columns ``time_distance`` (days), ``Pij`` and
        ``zeta_plus_1``.
    tau, c : float
        Taper time and Omori offset, in days.
    omega : float
        Omori exponent minus 1.
    label : str, optional
        Legend label of the fitted model.
    comparison_params : dict, optional
        ``{label: {"tau": ..., "c": ..., "omega": ...}}``, other kernels to
        draw for comparison.
    file_name : str or None, default "time_kernel_fit.pdf"
        Where to save the figure; ``None`` to not save.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; by default a new figure.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fit = _Fit(label or "fitted model", fs.DARK,
               {"log10_c": np.log10(c), "log10_tau": np.log10(tau),
                "omega": omega}, pij=p_mat)
    return _time_plot([fit], _comparisons(comparison_params, ("c", "tau")),
                      file_name, ax)


def _productivity_plot(fits, comparisons, ref, file_name, ax):
    with fs.style():
        fig, ax, own = _new_axes(ax)
        min_mag = min(np.min(f.catalog["magnitude"]) for f in fits)
        max_mag = max(np.max(f.catalog["magnitude"]) for f in fits)
        magnitudes = np.linspace(min_mag, max_mag, 100)

        def law(params, reference, m):
            return expected_aftershocks(
                m, (parameter_dict2array(params)[2:], reference), True, True)

        for f in comparisons:
            ax.plot(magnitudes, law(f.params, ref, magnitudes),
                    color=f.color, label=f.label, zorder=3)
        for f in fits:
            ax.plot(magnitudes, law(f.params, f.ref, magnitudes),
                    color=f.color, label=f.label, zorder=4)
            f_min = np.min(f.catalog["magnitude"])
            f_max = np.max(f.catalog["magnitude"])
            bins = np.arange(f_min - f.delta_m / 2,
                             f_max + 3 * f.delta_m / 2, f.delta_m)
            centres = (bins[1:] + bins[:-1]) / 2
            counts, _ = np.histogram(f.pij["source_magnitude"], bins=bins,
                                     weights=f.weights)
            how_many, _ = np.histogram(f.catalog["magnitude"], bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                counts_scaled = np.array(counts) / how_many
            shown = how_many > 0
            ax.scatter(centres[shown], counts_scaled[shown], s=fs.DOT,
                       color=f.color, label=f"observed: {f.label}",
                       zorder=5, linewidths=0)

        ax.set_yscale("log")
        ax.set_xlabel("magnitude of the triggering event")
        ax.set_ylabel("number of direct aftershocks")
        _legend(ax, fits, comparisons)
        _finish(fig, file_name, own)
    return fig, ax


def productivity_plot(
        p_mat: pd.DataFrame,
        catalog: pd.DataFrame,
        params: dict,
        mc: float,
        delta_m: float,
        label: str = None,
        comparison_params: dict = {},
        file_name: str = "productivity_fit.pdf",
        ax=None):
    r"""Fit of the productivity law to the observed branching structure.

    Dots: per magnitude bin, the sum of $p_{ij} (1 + \zeta_j)$ over the
    pairs whose triggering event $e_i$ is in the bin, divided by the number
    of events in the bin. Line: the expected number of direct aftershocks
    (above the reference magnitude) of an event of magnitude $m$,
    `etas.inversion.expected_aftershocks` over all space and time.

    To show several inversions in one figure, use
    `ETASFitVisualisation.productivity_law_plot` with ``others``.

    Parameters
    ----------
    p_mat : pd.DataFrame
        Pairs, with columns ``source_magnitude``, ``Pij`` and
        ``zeta_plus_1``.
    catalog : pd.DataFrame
        Events, with column ``magnitude`` (used to count the events per
        bin).
    params : dict
        Fitted ETAS parameters.
    mc : float
        Reference magnitude of the kernel ($m_{\mathrm{ref}}$, the lower
        edge of the lowest magnitude bin, ``m_ref - delta_m / 2``).
    delta_m : float
        Magnitude bin width.
    label : str, optional
    comparison_params : dict, optional
        ``{label: parameter dict}``, other productivity laws to draw, with
        the same reference magnitude ``mc`` (see
        `ETASFitVisualisation`, which converts them).
    file_name : str or None, default "productivity_fit.pdf"
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fit = _Fit(label or "fitted model", fs.DARK, params, pij=p_mat, ref=mc,
               catalog=catalog, delta_m=delta_m)
    return _productivity_plot([fit], _comparisons(comparison_params, ()),
                              mc, file_name, ax)


def spatial_kernel(
        dist: float,
        d: float,
        gamma: float,
        rho: float,
        m: float,
        mc: float):
    r"""Probability density of the squared distance $r^2$ between an event
    of magnitude $m$ and its aftershocks.

    $$
    f(r^2) = \frac{\rho \, d_m^{\rho}}{(r^2 + d_m)^{1 + \rho}},
    \quad d_m = d \, e^{\gamma (m - m_c)}.
    $$

    Parameters
    ----------
    dist : float or np.ndarray
        Squared distance $r^2$.
    d, gamma, rho : float
        Space-kernel parameters.
    m : float
        Magnitude of the triggering event.
    mc : float
        Reference magnitude of the kernel.

    Returns
    -------
    float or np.ndarray
    """
    d_m = d * np.exp(gamma * (m - mc))
    return rho * np.power(d_m, rho) / (dist + d_m) ** (1 + rho)


def _space_unit(space_unit_in_meters):
    return {0.001: "mm", 0.01: "cm", 1: "m", 1000: "km"}.get(
        space_unit_in_meters, f"{space_unit_in_meters} m")


def _squared_distances(f, moi):
    """Squared distances (in the unit of the figure) and weights of the
    pairs of fit ``f`` whose triggering event has magnitude ``moi``."""
    sub = f.pij[np.round(f.pij["source_magnitude"], 1) == np.round(moi, 1)]
    sq = np.array(sub["spatial_distance_squared"], dtype=float) \
        * f.area_factor
    if len(sq) and np.min(sq) == 0 and np.any(sq > 0):
        sq = sq + np.min(sq[np.nonzero(sq)]) / 2
    return sq, (sub["Pij"] * sub["zeta_plus_1"]).to_numpy()


def _space_plot(fits, comparisons, ref, magnitudes, unit, file_name, axes):
    selections = []
    for moi in magnitudes:
        data = [_squared_distances(f, moi) for f in fits]
        if any(np.any(sq > 0) for sq, _ in data):
            selections.append((moi, data))

    with fs.style():
        if axes is None:
            n = max(len(selections), 1)
            ncols = min(n, 3)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows),
                layout="constrained", squeeze=False, sharey=True)
            for extra in axes.ravel()[len(selections):]:
                extra.set_visible(False)
            axes = list(axes.ravel()[:len(selections)])
            own = True
        else:
            axes = list(axes)
            fig = axes[0].figure
            own = False

        for k, ((moi, data), ax) in enumerate(zip(selections, axes)):
            all_sq = np.concatenate([sq for sq, _ in data])
            all_sq = all_sq[all_sq > 0]
            distance_bins = np.logspace(np.log10(np.min(all_sq) / 2),
                                        np.log10(np.max(all_sq) + 1))
            distances = (distance_bins[1:] + distance_bins[:-1]) / 2

            for f in comparisons:
                ax.plot(np.sqrt(distances), spatial_kernel(
                    distances, f.d, f.gamma, f.rho, moi, ref),
                    color=f.color, label=f.label, zorder=3)
            for f, (sq, weights) in zip(fits, data):
                ax.plot(np.sqrt(distances), spatial_kernel(
                    distances, f.d, f.gamma, f.rho, moi, f.ref),
                    color=f.color, label=f.label, zorder=4)
                if len(sq) == 0:
                    continue
                dist_emp, _ = np.histogram(sq, bins=distance_bins,
                                           weights=weights, density=True)
                shown = dist_emp > 0
                ax.scatter(np.sqrt(distances)[shown], dist_emp[shown],
                           s=fs.DOT, color=f.color,
                           label=f"observed: {f.label}", zorder=5,
                           linewidths=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"magnitude {moi:.1f}")
            ax.set_xlabel(f"distance [{unit}]")
            for i, f in enumerate(fits):
                d_m = f.d * np.exp(f.gamma * (moi - f.ref))
                fs.reference_line(
                    ax, np.sqrt(d_m),
                    f"D$^2$ = {d_m:.2g} {unit}$^2$", f.color, _side(i),
                    where="bottom")
            if k == 0:
                ax.set_ylabel(f"probability density [1/{unit}²]")
                _legend(ax, fits, comparisons)

        if file_name is not None and not os.path.splitext(file_name)[1]:
            file_name = file_name + ".pdf"
        _finish(fig, file_name, own)
    return fig, axes


def spatial_decay_plot(
        p_mat: pd.DataFrame,
        magnitudes: list,
        d: float,
        gamma: float,
        rho: float,
        mc: float,
        space_unit_in_meters: float = 1000,
        label: str = None,
        comparison_params: dict = {},
        file_name: str = "space_kernel_fit",
        axes=None):
    r"""Fit of the space kernel to the observed branching structure, one
    panel per magnitude of the triggering event.

    Dots: distribution of the squared distances $r_{ij}^2$ of the pairs
    whose triggering event has the given magnitude $m$, weighted by
    $p_{ij} (1 + \zeta_j)$, in logarithmic bins. Line: `spatial_kernel`.
    A thin vertical line marks the size $D$ of the aftershock zone, with
    $D^2 = d \, e^{\gamma (m - m_{\mathrm{ref}})}$: up to about $D$, the
    density is nearly flat, beyond it, it decays. Magnitudes without
    pairs are left out.

    To show several inversions in one figure, use
    `ETASFitVisualisation.space_kernel_plot` with ``others``.

    Parameters
    ----------
    p_mat : pd.DataFrame
        Pairs, with columns ``spatial_distance_squared``,
        ``source_magnitude``, ``Pij`` and ``zeta_plus_1``.
    magnitudes : list of float
        Magnitudes of the triggering events, one panel each.
    d, gamma, rho : float
        Space-kernel parameters.
    mc : float
        Reference magnitude of the kernel.
    space_unit_in_meters : float, default 1000
    label : str, optional
    comparison_params : dict, optional
        ``{label: {"d": ..., "gamma": ..., "rho": ...}}``.
    file_name : str or None, default "space_kernel_fit"
        Where to save the figure (``.pdf`` is added if there is no
        extension); ``None`` to not save. Earlier versions wrote one file
        per magnitude; all magnitudes are now panels of one figure.
    axes : sequence of matplotlib.axes.Axes, optional
        One axes per magnitude (with pairs) to draw into.

    Returns
    -------
    (matplotlib.figure.Figure, list of matplotlib.axes.Axes)
    """
    fit = _Fit(label or "fitted model", fs.DARK,
               {"log10_d": np.log10(d), "gamma": gamma, "rho": rho},
               pij=p_mat, ref=mc)
    return _space_plot([fit], _comparisons(comparison_params, ("d",)), mc,
                       magnitudes, _space_unit(space_unit_in_meters),
                       file_name, axes)


def convert_reference_magnitude(params, from_ref, to_ref, beta,
                                space_unit_factor=1.0):
    r"""Parameters of an ETAS model with another reference magnitude.

    A model with reference magnitude $m_1$ (``from_ref``) describes the
    events above $m_1$; for comparison with a model with reference
    magnitude $m_2$ (``to_ref``), it is rewritten for the events above
    $m_2$, with $\Delta = m_2 - m_1$:

    - $d \to d \, e^{\gamma \Delta}$: the size of the aftershock zone of
      an event of magnitude $m_2$;
    - $k_0 \to k_0 \, e^{(a - \beta) \Delta}$: the productivity of an
      event of magnitude $m_2$, counting only aftershocks above $m_2$;
    - $\mu \to \mu \, e^{-\beta \Delta}$: the background events above
      $m_2$.

    The other parameters do not change. ``space_unit_factor`` is the length
    of the model's space unit in the new unit (e.g. 0.001 from m to km);
    areas scale with its square.

    Parameters
    ----------
    params : dict
        ETAS parameters (``log10_mu``, ``log10_k0``, ``a``, ``log10_d``,
        ``gamma``, ``rho``, ...).
    from_ref, to_ref : float
        Reference magnitudes of the kernel (lower edges of the lowest
        magnitude bin).
    beta : float
        Gutenberg-Richter parameter of the model (natural-log units).
    space_unit_factor : float, default 1

    Returns
    -------
    dict
        New parameters; the input is not modified.
    """
    params = dict(params)
    delta = to_ref - from_ref
    u2 = space_unit_factor ** 2
    params["log10_d"] = (params["log10_d"] + params["gamma"] * delta
                         / np.log(10) + np.log10(u2))
    params["log10_k0"] = (params["log10_k0"]
                          + (params["a"] - beta) * delta / np.log(10)
                          + params["rho"] * np.log10(u2))
    if params.get("log10_mu") is not None:
        params["log10_mu"] = (params["log10_mu"] - beta * delta / np.log(10)
                              - np.log10(u2))
    return params


class ETASFitVisualisation:
    """Plots of the fit of one inversion (time kernel, productivity law,
    space kernel), optionally with other inversions and other models for
    comparison.

    Usage, from stored results (`ETASParameterCalculation.store_results`
    with ``store_pij=True``):

        visualisation = ETASFitVisualisation({
            "fn_catalog": out["fn_ip"], "fn_pij": out["fn_pij"],
            "delta_m": out["delta_m"], "mc": out["m_ref"],
            "parameters": out["final_parameters"],
            "magnitude_list": [4.0, 4.5, 5.0], "store_path": "plots/"})
        visualisation.all_plots()

    or directly after an inversion:

        visualisation = ETASFitVisualisation.from_calculation(calculation)
        fig, axes = visualisation.overview()

    Several inversions in one figure, each with its dots and line in its
    own colour (this one dark gray):

        fig, axes = visualisation.overview(others=[other_visualisation])

    Parameters
    ----------
    metadata : dict
        - ``fn_catalog`` or ``catalog``: events (csv file or DataFrame)
          with column ``magnitude``, usually the target events
          (``fn_ip``).
        - ``fn_pij`` or ``pij``: pairs (csv file or DataFrame).
        - ``mc``: reference magnitude ``m_ref`` of the inversion;
          ``delta_m``: magnitude bin width. The kernels use
          ``mc - delta_m / 2``, as the inversion does.
        - ``parameters``: fitted parameters.
        - ``label`` (optional): legend label of the fitted model.
        - ``comparison_parameters`` (optional): ``{label: parameters}``,
          models without data (lines only), each with its own ``mc``,
          ``delta_m``, ``beta`` and optionally ``space_unit_in_meters``;
          converted to the reference magnitude and space unit of the fit
          (`convert_reference_magnitude`).
        - ``space_unit_in_meters`` (default 1000), ``magnitude_list``
          (magnitudes for the space kernel), ``store_path`` (directory for
          the files written by the ``*_plot`` methods, default the current
          directory).
    """

    def __init__(self, metadata: dict):
        self.catalog = metadata.get("catalog")
        if self.catalog is None:
            self.catalog = pd.read_csv(metadata.get("fn_catalog", None))
        self.Pij = metadata.get("pij")
        if self.Pij is None:
            self.Pij = pd.read_csv(metadata.get("fn_pij", None))
        self.mc = metadata.get("mc", None)
        self.delta_m = metadata.get("delta_m", None)
        self.parameters = metadata.get("parameters", None)
        self.label = metadata.get("label", None)
        self.comparison_parameters = metadata.get("comparison_parameters", {})
        self.space_unit_in_meters = metadata.get("space_unit_in_meters", 1000)
        self.magnitude_list = metadata.get("magnitude_list", None)
        self.store_path = metadata.get("store_path", None) or ""

        self.a, self.gamma, self.c, self.d, self.k, self.mu, \
            self.tau, self.omega, self.rho = \
            self.parameters["a"], \
            self.parameters["gamma"], \
            np.power(10, self.parameters["log10_c"]), \
            np.power(10, self.parameters["log10_d"]), \
            np.power(10, self.parameters["log10_k0"]), \
            np.power(10, self.parameters["log10_mu"]), \
            np.power(10, self.parameters["log10_tau"]), \
            self.parameters["omega"], \
            self.parameters["rho"]

        comparison_params = {}
        for label, params in (self.comparison_parameters or {}).items():
            converted = convert_reference_magnitude(
                params,
                from_ref=params["mc"] - params["delta_m"] / 2,
                to_ref=self.reference_magnitude,
                beta=params["beta"],
                space_unit_factor=params.get("space_unit_in_meters", 1000)
                / self.space_unit_in_meters)
            converted["space_unit_in_meters"] = self.space_unit_in_meters
            comparison_params[label] = converted
        self.comparison_parameters = comparison_params

    @property
    def reference_magnitude(self):
        """Reference magnitude of the kernels, ``mc - delta_m / 2``."""
        return self.mc - self.delta_m / 2

    @classmethod
    def from_calculation(cls, calculation, label=None, magnitude_list=None,
                         comparison_parameters=None, store_path=None):
        """Visualisation of a finished inversion, without storing and
        reading files.

        Parameters
        ----------
        calculation : etas.inversion.ETASParameterCalculation
            After `invert`.
        label : str, optional
            Default: the ``name`` of the model, or its ``id`` if it has no
            name.
        magnitude_list : list of float, optional
            Magnitudes for the space-kernel plot; default: from ``m_ref``
            in steps of 0.5 up to the largest magnitude with at least 20
            triggering events (at most four).
        comparison_parameters : dict, optional
            As in the constructor.
        store_path : str, optional
        """
        pij = calculation.pij.reset_index()
        catalog = calculation.target_events.reset_index()
        if magnitude_list is None:
            magnitude_list = _default_magnitudes(
                pij, calculation.m_ref, calculation.delta_m)
        return cls({
            "catalog": catalog,
            "pij": pij,
            "mc": calculation.m_ref,
            "delta_m": calculation.delta_m,
            "parameters": calculation.theta,
            "label": label or _calculation_label(calculation),
            "comparison_parameters": comparison_parameters or {},
            "space_unit_in_meters": calculation.space_unit_in_meters,
            "magnitude_list": magnitude_list,
            "store_path": store_path,
        })

    def _fits(self, others=None):
        """This fit (dark gray), the ``others`` and the comparison models,
        in the colours, labels and space unit of the figure.

        If the reference magnitudes differ, it is added to the labels: the
        productivity law of each fit counts the aftershocks above its own
        reference magnitude.
        """
        visualisations = [self] + list(others or [])
        n = len(visualisations)
        colors = fs.colors(n + len(self.comparison_parameters))
        show_ref = len({round(v.mc, 6) for v in visualisations}) > 1
        fits = []
        for i, v in enumerate(visualisations):
            label = v.label or ("fitted model" if n == 1 else f"fit {i + 1}")
            if show_ref:
                label += f" (m$_{{ref}}$ = {v.mc:.1f})"
            fits.append(_Fit(
                label, colors[i], v.parameters, pij=v.Pij,
                ref=v.reference_magnitude, catalog=v.catalog,
                delta_m=v.delta_m,
                area_factor=(v.space_unit_in_meters
                             / self.space_unit_in_meters) ** 2))
        comparisons = [
            _Fit(label, color, params) for color, (label, params) in zip(
                colors[n:], self.comparison_parameters.items())]
        return fits, comparisons

    def _path(self, fn_store):
        if fn_store is None:
            return None
        return os.path.join(self.store_path, fn_store)

    def time_kernel_plot(self, fn_store: str = "time_kernel_fit.pdf",
                         ax=None, others=None):
        """Time kernel (see `temporal_decay_plot`); saved to
        ``store_path/fn_store`` unless ``fn_store`` is None.

        ``others``: further `ETASFitVisualisation` objects to draw in the
        same figure, each in its own colour (their comparison models are
        not drawn)."""
        fits, comparisons = self._fits(others)
        return _time_plot(fits, comparisons, self._path(fn_store), ax)

    def productivity_law_plot(
            self,
            fn_store: str = "productivity_law_fit.pdf",
            ax=None,
            others=None):
        """Productivity law (see `productivity_plot`). The comparison
        models use the reference magnitude of this fit, the ``others``
        their own."""
        fits, comparisons = self._fits(others)
        return _productivity_plot(fits, comparisons,
                                  self.reference_magnitude,
                                  self._path(fn_store), ax)

    def space_kernel_plot(self, fn_store: str = "space_kernel_fit.pdf",
                          axes=None, others=None):
        """Space kernel, one panel per magnitude in ``magnitude_list`` (see
        `spatial_decay_plot`), in the space unit of this fit."""
        fits, comparisons = self._fits(others)
        return _space_plot(fits, comparisons, self.reference_magnitude,
                           self.magnitude_list,
                           _space_unit(self.space_unit_in_meters),
                           self._path(fn_store), axes)

    def overview(self, file_name: str = None, magnitude: float = None,
                 others=None):
        """One figure with the time kernel (a), the productivity law (b)
        and the space kernel (c) for one magnitude (default: the first of
        ``magnitude_list``), with one legend on the right.

        ``others``: further `ETASFitVisualisation` objects to draw in the
        same figure, each in its own colour.

        Returns
        -------
        (matplotlib.figure.Figure, list of matplotlib.axes.Axes)
        """
        if magnitude is None:
            magnitude = (self.magnitude_list[0] if self.magnitude_list
                         else self.mc)
        with fs.style():
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.8),
                                     layout="constrained")
        self.time_kernel_plot(fn_store=None, ax=axes[0], others=others)
        self.productivity_law_plot(fn_store=None, ax=axes[1], others=others)
        original = self.magnitude_list
        self.magnitude_list = [magnitude]
        try:
            self.space_kernel_plot(fn_store=None, axes=[axes[2]],
                                   others=others)
        finally:
            self.magnitude_list = original
        with fs.style():
            for ax, letter in zip(axes, "abc"):
                fs.panel_letter(ax, letter)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            handles, labels = _legend_entries(*self._fits(others))
            legend = fig.legend(handles, labels, title=LEGEND_TITLE,
                                loc="outside right upper", alignment="left")
            legend.get_title().set_color(fs.MUTED)
            if file_name is not None:
                fig.savefig(file_name, dpi=200)
        return fig, list(axes)

    def all_plots(self, others=None):
        """Store the time, productivity and space plots in
        ``store_path``."""
        self.time_kernel_plot(others=others)
        self.productivity_law_plot(others=others)
        self.space_kernel_plot(others=others)


def _calculation_label(calculation):
    if calculation.name != "NoName ETAS Model":
        return calculation.name
    return str(calculation.id)


def _default_magnitudes(pij, m_ref, delta_m, min_sources=20, n_max=4):
    """Magnitudes from ``m_ref`` in steps of 0.5 with at least
    ``min_sources`` triggering events in their bin."""
    sources = pij.drop_duplicates("source_id")["source_magnitude"]
    rounded = np.round(sources, 1).value_counts()
    magnitudes = []
    for m in np.arange(m_ref, m_ref + 4, 0.5):
        if rounded.get(np.round(m, 1), 0) >= min_sources:
            magnitudes.append(float(np.round(m, 1)))
    return magnitudes[:n_max] or [float(np.round(m_ref, 1))]
