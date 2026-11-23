"""Figure style of the package (used by `etas.plots`).

One encoding throughout, so that a colour or a mark means the same thing in
every figure:

- colour identifies a fit: its dots (the observed branching structure of
  the inversion) and its line (the fitted model) have the same colour; the
  first fit is dark gray, further fits and comparison models get one
  categorical colour each (colour-blind safe);
- dots are observations, lines are models; all lines are solid, line style
  never carries meaning;
- reference values (c, tau, the size of the aftershock zone) are thin pale
  vertical lines in the colour of their fit, with a vertical label.

The style is applied only inside the plotting functions (`style`), so it
does not change the user's matplotlib settings. Avenir is used where
installed, otherwise DejaVu Sans.
"""
import contextlib

import matplotlib.pyplot as plt

TEAL = "#43AEBA"
DARK = "#3A3937"
GRID = "#E4E3DF"
MUTED = "#8C8B86"

# the first fit is dark gray; then teal and Okabe-Ito colours,
# distinguishable also for colour-blind readers
PALETTE = [DARK, TEAL, "#E69F00", "#CC79A7", "#0072B2", "#D55E00",
           "#56B4E9"]

L, M, S = 15, 13, 11  # labels and titles, tick labels, small notes
LINE = 1.1            # data lines
THIN = 0.7            # reference lines
DOT = 12              # marker area of observed points

RC = {
    "font.family": ["Avenir", "DejaVu Sans"],
    "font.weight": "light",
    "axes.labelweight": "light",
    "axes.titleweight": "light",
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "axes.edgecolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "lines.linewidth": LINE,
    "xtick.labelsize": M,
    "ytick.labelsize": M,
    "axes.labelsize": L,
    "axes.titlesize": L,
    "legend.fontsize": S,
    "legend.title_fontsize": S,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.formatter.use_mathtext": True,
    "mathtext.default": "regular",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}


@contextlib.contextmanager
def style():
    """Apply the package style within a ``with`` block."""
    with plt.rc_context(RC):
        yield


def colors(n):
    """``n`` colours: dark gray first, then the categorical colours."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def reference_line(ax, x, text, color=DARK, side="left", where="top"):
    """Thin pale vertical line at ``x`` with a vertical label along it, at
    the ``where`` ("top" or "bottom") end of the axes and on the given
    ``side`` of the line (switched if the label would leave the axes).
    Choose ``where`` so that the label is off the data. Call after the axis
    scales and limits are set."""
    ax.axvline(x, color=color, lw=THIN, alpha=0.45, zorder=1)
    ax.get_xlim()  # apply pending autoscaling before converting x
    x_axes = ax.transAxes.inverted().transform(
        ax.transData.transform((x, 1)))[0]
    if side == "left" and x_axes < 0.06:
        side = "right"
    elif side == "right" and x_axes > 0.94:
        side = "left"
    top = where == "top"
    ax.annotate(text, xy=(x, 0.98 if top else 0.02),
                xycoords=ax.get_xaxis_transform(),
                xytext=(-2 if side == "left" else 2, 0),
                textcoords="offset points", rotation=90, fontsize=S,
                color=color, va="top" if top else "bottom",
                ha="right" if side == "left" else "left")


def panel_letter(ax, letter):
    """Lowercase panel letter at the top left, outside the axes."""
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=L,
            fontweight="medium", color=DARK, ha="right", va="bottom")
