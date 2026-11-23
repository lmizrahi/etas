"""Plots render, follow the call conventions (ax, return values, saving
only with a file name), do not change the user's matplotlib settings, and
comparison models are converted to the fitted reference magnitude
correctly."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from etas.inversion import expected_aftershocks, parameter_dict2array  # noqa
from etas.plots import (ETASFitVisualisation,  # noqa: E402
                        convert_reference_magnitude, productivity_plot,
                        spatial_decay_plot, temporal_decay_plot)

SWISS = {"log10_mu": -6.19, "log10_k0": -2.71, "a": 1.42, "log10_c": -2.68,
         "omega": -0.1, "log10_tau": 3.6, "log10_d": -0.55, "gamma": 0.43,
         "rho": 0.72, "mc": 2.3, "delta_m": 0.1, "beta": 2.74}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def visualisation(short_inversion):
    return ETASFitVisualisation.from_calculation(
        short_inversion, comparison_parameters={"Swiss ETAS": dict(SWISS)})


def test_overview_has_three_panels(visualisation, tmp_path):
    fig, axes = visualisation.overview(file_name=tmp_path / "overview.png")
    assert len(axes) == 3
    assert (tmp_path / "overview.png").stat().st_size > 0
    for ax in axes:
        assert len(ax.lines) >= 2 and len(ax.collections) >= 1


def test_methods_save_only_with_a_file_name(visualisation, tmp_path):
    visualisation.store_path = str(tmp_path)
    fig, ax = visualisation.time_kernel_plot(fn_store=None)
    fig, ax = visualisation.productivity_law_plot(fn_store=None)
    fig, axes = visualisation.space_kernel_plot(fn_store=None)
    assert list(tmp_path.iterdir()) == []
    visualisation.all_plots()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "productivity_law_fit.pdf", "space_kernel_fit.pdf",
        "time_kernel_fit.pdf"]


def test_functions_draw_into_given_axes(visualisation):
    fig, axes = plt.subplots(1, 3)
    v = visualisation
    out = temporal_decay_plot(v.Pij, v.tau, v.c, v.omega, file_name=None,
                              ax=axes[0])
    assert out == (fig, axes[0])
    out = productivity_plot(v.Pij, v.catalog, v.parameters,
                            v.reference_magnitude, v.delta_m,
                            file_name=None, ax=axes[1])
    assert out == (fig, axes[1])
    out = spatial_decay_plot(v.Pij, v.magnitude_list[:1], v.d, v.gamma,
                             v.rho, v.reference_magnitude, file_name=None,
                             axes=[axes[2]])
    assert out == (fig, [axes[2]])
    # the fitted model and its observed points are both dark gray
    model = [line for line in axes[0].lines
             if line.get_label() == "fitted model"]
    assert model[0].get_color().lower() == "#3a3937"
    observed = [c for c in axes[0].collections
                if c.get_label() == "observed: fitted model"]
    assert matplotlib.colors.to_hex(
        observed[0].get_facecolor()[0]) == "#3a3937"


def _colors(ax):
    lines = {line.get_label(): matplotlib.colors.to_hex(line.get_color())
             for line in ax.lines if not line.get_label().startswith("_")}
    dots = {c.get_label(): matplotlib.colors.to_hex(c.get_facecolor()[0])
            for c in ax.collections}
    return lines, dots


def test_several_fits_one_colour_each(short_inversion):
    """Each fit has its dots and its line in one colour, the first dark
    gray; comparison models have their own colours and no dots."""
    first = ETASFitVisualisation.from_calculation(
        short_inversion, label="A",
        comparison_parameters={"Swiss ETAS": dict(SWISS)})
    second = ETASFitVisualisation.from_calculation(short_inversion,
                                                   label="B")
    fig, axes = first.overview(others=[second])
    for ax in axes:
        lines, dots = _colors(ax)
        assert lines["A"] == dots["observed: A"] == "#3a3937"
        assert lines["B"] == dots["observed: B"] != "#3a3937"
        assert lines["Swiss ETAS"] not in (lines["A"], lines["B"])
        assert "observed: Swiss ETAS" not in dots
    labels = [t.get_text() for t in fig.legends[0].texts]
    assert labels == ["A", "B", "Swiss ETAS"]


def test_reference_magnitude_in_labels_when_they_differ(short_inversion):
    first = ETASFitVisualisation.from_calculation(short_inversion,
                                                  label="A")
    second = ETASFitVisualisation.from_calculation(short_inversion,
                                                   label="B")
    second.mc = first.mc + 0.4
    fig, ax = first.productivity_law_plot(fn_store=None, others=[second])
    labels = [t.get_text() for t in ax.get_legend().texts]
    assert labels == ["A (m$_{ref}$ = 3.6)", "B (m$_{ref}$ = 4.0)"]


def test_reference_lines_have_vertical_labels(visualisation):
    fig, ax = visualisation.time_kernel_plot(fn_store=None)
    labels = [t for t in ax.texts if t.get_text().startswith(("c =",
                                                              "$\\tau$ ="))]
    assert len(labels) == 2
    assert all(t.get_rotation() == 90 for t in labels)
    fig, axes = visualisation.space_kernel_plot(fn_store=None)
    for ax in axes:
        zone = [t for t in ax.texts if t.get_text().startswith("D$^2$")]
        assert len(zone) == 1 and zone[0].get_rotation() == 90


def test_style_does_not_leak(visualisation):
    before = dict(matplotlib.rcParams)
    visualisation.overview()
    assert dict(matplotlib.rcParams) == before


@pytest.mark.parametrize("magnitude", [3.6, 4.5, 6.0])
def test_comparison_model_conversion(magnitude):
    """After the conversion, a comparison model predicts the same number of
    aftershocks above the new reference magnitude, and the same size of
    the aftershock zone, as the original model."""
    from_ref = SWISS["mc"] - SWISS["delta_m"] / 2
    to_ref = 3.55
    converted = convert_reference_magnitude(SWISS, from_ref, to_ref,
                                            SWISS["beta"])

    def n_above(params, ref, threshold):
        n = expected_aftershocks(
            magnitude, [parameter_dict2array(params)[2:], ref], True, True)
        return n * np.exp(-SWISS["beta"] * (threshold - ref))

    assert n_above(converted, to_ref, to_ref) == pytest.approx(
        n_above(SWISS, from_ref, to_ref), rel=1e-12)

    def zone(params, ref):
        return 10 ** params["log10_d"] * np.exp(
            params["gamma"] * (magnitude - ref))

    assert zone(converted, to_ref) == pytest.approx(zone(SWISS, from_ref))
    # background events above the new reference magnitude
    assert 10 ** converted["log10_mu"] == pytest.approx(
        10 ** SWISS["log10_mu"] * np.exp(-SWISS["beta"] * (to_ref
                                                           - from_ref)))


def test_comparison_model_unit_conversion():
    """A model in meters, converted to km, gives the same expected number of
    aftershocks and a zone size 1e6 times smaller in km^2."""
    in_m = dict(SWISS, log10_d=SWISS["log10_d"] + 6)
    converted = convert_reference_magnitude(in_m, 2.25, 2.25, 2.74,
                                            space_unit_factor=0.001)
    assert converted["log10_d"] == pytest.approx(SWISS["log10_d"])
    args = (4.0, True, True)
    n_m = expected_aftershocks(args[0], [parameter_dict2array(in_m)[2:],
                                         2.25], *args[1:])
    n_km = expected_aftershocks(args[0], [parameter_dict2array(
        converted)[2:], 2.25], *args[1:])
    assert n_km == pytest.approx(n_m, rel=1e-12)
