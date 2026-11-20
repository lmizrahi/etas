# The model

## Notation

A catalog is a list of earthquakes $e_1, \dots, e_N$, ordered by time. Each
event $e_i$ has a location $(x_i, y_i)$ (latitude and longitude, in
degrees), an occurrence time $t_i$ (in days), and a magnitude $m_i$. The
distance (in km) between a location $(x, y)$ and the epicenter of $e_i$ is
written $r_i(x, y)$, and the distance between two events $e_i$ and $e_j$ is
$r_{ij}$.

## Completeness and reference magnitude

Two magnitudes play a role in the model, and they need to be kept apart.

- The **magnitude of completeness** $m_c$ is the magnitude above which all
  events are recorded in the catalog.
- The **reference magnitude** $m_{\mathrm{ref}}$ is the magnitude above
  which the model describes events, both as triggering events and as
  aftershocks. Productivity and the size of the aftershock zone are given
  relative to an event of magnitude $m_{\mathrm{ref}}$.

In the default mode, the catalog is complete above a single magnitude $m_c$,
and $m_{\mathrm{ref}} = m_c$. The two differ only in the modes with varying
completeness (`mc = "var"` or `"positive"`, see
[Varying completeness](#varying-completeness)). There, each event $e_i$ has
its own completeness $m_{c,i}$, and $m_{\mathrm{ref}}$ is a single magnitude
at or below the smallest $m_{c,i}$.

Magnitudes are rounded to bins of width $\Delta m$ (`delta_m`). In all
formulas below, $m_{\mathrm{ref}}$ stands for the lower edge of the lowest
magnitude bin, in the code `m_ref - delta_m / 2`. For example, with
`m_ref = 3.6` and `delta_m = 0.1`, the formulas use
$m_{\mathrm{ref}} = 3.55$.

## Conditional intensity

The ETAS model describes the earthquake rate, given past earthquakes. Each
event can trigger aftershocks, which in turn can trigger their own
aftershocks. Events that are not triggered by any other event are called
background events. The conditional intensity, that is, the expected number
of events per km² and day at location $(x, y)$ and time $t$, is

$$
\lambda(x, y, t) = \mu + \sum_{i: t_i < t} g\big(m_i, r_i(x, y), t - t_i\big),
$$

where

- $\mu$ is the rate of background events, in events per km² and day,
  assumed constant in space and time unless noted otherwise;
- the sum runs over all events $e_i$ that occurred before $t$;
- $g$ is the aftershock triggering kernel (defined below), evaluated at the
  magnitude $m_i$ of $e_i$, the distance $r_i(x, y)$ from $e_i$, and the
  time $t - t_i$ since $e_i$.

## Triggering kernel

The triggering kernel gives the rate of direct aftershocks of an event of
magnitude $m$, at a distance $r$ (km) and a time delay $\Delta t$ (days). It
is the product of a productivity term, a space kernel and a time kernel,

$$
g(m, r, \Delta t) = k_0 \, e^{a (m - m_{\mathrm{ref}})}
    \cdot \frac{1}{\left(r^2 + d \, e^{\gamma (m - m_{\mathrm{ref}})}\right)^{1 + \rho}}
    \cdot \frac{e^{-\Delta t / \tau}}{(\Delta t + c)^{1 + \omega}},
$$

where

- $m_{\mathrm{ref}}$ is the reference magnitude (see above);
- $k_0$ is the productivity of an event of magnitude $m_{\mathrm{ref}}$,
  and $a$ defines how productivity increases with magnitude (per magnitude
  unit);
- $d$ (km²), $\gamma$ and $\rho$ are the parameters of the space kernel:
  the size of the aftershock zone of an event of magnitude
  $m_{\mathrm{ref}}$ is $d$, it grows with magnitude at rate $\gamma$ (per
  magnitude unit), and $\rho$ sets how fast the rate decays with distance
  beyond it;
- $c$ (days), $\omega$ and $\tau$ (days) are the parameters of the time
  kernel: an Omori-Utsu law with offset $c$ and exponent $1 + \omega$,
  tapered by an exponential with time scale $\tau$.

## Magnitudes

Magnitudes at or above $m_{\mathrm{ref}}$ are assumed to follow the
Gutenberg-Richter law, with probability density

$$
f(m) = \beta \, e^{-\beta (m - m_{\mathrm{ref}})}, \quad m \geq m_{\mathrm{ref}},
$$

where $\beta = b \ln 10$ is the Gutenberg-Richter parameter in natural-log
units and $b$ is the usual b-value. For $b = 1$, $\beta \approx 2.30$, and
each magnitude unit has ten times fewer events than the one below it. All
`beta` values in the code are in these natural-log units.

## Parameters

| Parameter | Symbol | Meaning | Unit | Default range |
|---|---|---|---|---|
| `log10_mu` | $\log_{10} \mu$ | background rate | per km² and day | (-10, 0) |
| `log10_iota` | $\log_{10} \iota$ | rate of the additional background term (`bg_term`) | per km² and day | (-10, 0) |
| `log10_k0` | $\log_{10} k_0$ | productivity of an event of magnitude $m_{\mathrm{ref}}$ | | (-20, 10) |
| `a` | $a$ | increase of productivity with magnitude | per magnitude unit | (0.01, 20) |
| `log10_c` | $\log_{10} c$ | offset of the Omori-Utsu law | days | (-8, 0) |
| `omega` | $\omega$ | exponent of the Omori-Utsu law minus 1 ($p = 1 + \omega$) | | (-0.99, 1) |
| `log10_tau` | $\log_{10} \tau$ | time scale of the taper | days | (0.01, 12.26) |
| `log10_d` | $\log_{10} d$ | size of the aftershock zone of an event of magnitude $m_{\mathrm{ref}}$ | km² | (-4, 3) |
| `gamma` | $\gamma$ | increase of the aftershock zone with magnitude | per magnitude unit | (-1, 5) |
| `rho` | $\rho$ | decay of the rate with distance | | (0.01, 5) |

## Branching ratio

The branching ratio $\eta$ is the expected number of direct aftershocks of
an event, averaged over the Gutenberg-Richter distribution of magnitudes and
integrated over all space and time. Integrating the space kernel over the
plane gives a factor $e^{-\rho \gamma (m - m_{\mathrm{ref}})}$, so that the
number of direct aftershocks grows with magnitude as
$e^{\alpha (m - m_{\mathrm{ref}})}$, with the effective productivity
exponent

$$
\alpha = a - \rho \gamma .
$$

The branching ratio is finite only if $\alpha < \beta$, that is, if the
number of aftershocks grows more slowly with magnitude than the number of
events decreases. For $\eta < 1$, every sequence eventually dies out, and
about a fraction $\eta$ of all events are aftershocks. With the example
parameters, $\alpha = 1.69 - 0.51 \cdot 1.22 \approx 1.07$, below
$\beta \approx 2.30$, and $\eta \approx 0.90$: about 90% of the events in
the example catalog are aftershocks. See `etas.inversion.branching_ratio`.

## Inversion

The parameters are estimated with the expectation-maximization (EM)
algorithm for ETAS models introduced by Veen and Schoenberg (2008), adapted for
the tapered temporal kernel as described by Mizrahi et al. (2021a). The algorithm
starts from a random initial guess for the ETAS parameters, and then iterates through
two steps repeatedly until the parameters converge.

1. **Expectation step.** Given the current set of ETAS parameters, we calculate,
   for each pair of a possible triggering event $e_i$
   and a possible aftershock $e_j$, the probability $p_{ij}$ that $e_j$ was
   triggered by $e_i$ 

    $$
    p_{ij} = \frac{g(m_i, r_{ij}, t_j - t_i)}{\lambda(x_j, y_j, t_j)},
    $$

    where $r_{ij}$ is the distance (km) between $e_i$ and $e_j$, and
    $\lambda(x_j, y_j, t_j)$ the conditional intensity at the location and
    time of $e_j$. The probability that $e_j$ is a background event is
    $p_j^{\mathrm{bg}} = \mu / \lambda(x_j, y_j, t_j)$, so that
    $p_j^{\mathrm{bg}} + \sum_i p_{ij} = 1$. 

2. **Maximization step.** Given the newly calculated branching structure,
    the ETAS parameters are updated to maximize the likelihood of observing
    this branching structure, yielding a new set of ETAS parameters for the next
    expectation step.

The algorithm stops when the parameters change by less than 0.001 in total
from one iteration to the next, or after `max_iterations` iterations.

## Model variants

Besides the default model above, the inversion has several variants. They
are only summarized here; the configuration options are listed in
[Configuration](configuration.md), and the details are in the cited papers.

### Varying completeness

When the magnitude of completeness varies in space and time (`mc = "var"`),
each event $e_i$ has its own completeness $m_{c,i}$ (the column
`mc_current`): events below $m_{c,i}$ at the location and time of $e_i$ may
be missing from the catalog. The reference magnitude $m_{\mathrm{ref}}$ is
at most the smallest completeness. Two factors account for the missing
events (Mizrahi et al., 2021b, Section 3.1).

1. For each observed event $e_j$,

    $$
    1 + \zeta_j = e^{\beta (m_{c,j} - m_{\mathrm{ref}})}
    $$

    is the expected number of events above $m_{\mathrm{ref}}$, including
    the unobserved ones.

2. The aftershock rate of an observed event $e_i$ is multiplied by

    $$
    1 + \xi_i = e^{(\beta - \alpha)(m_{c,i} - m_{\mathrm{ref}})},
    $$

    to include the triggering by the unobserved events with magnitudes
    between $m_{\mathrm{ref}}$ and $m_{c,i}$, where $\alpha = a - \rho
    \gamma$ as above.

See `etas.inversion.observation_factor` and
`etas.inversion.responsibility_factor`.

### ETAS-positive

With `mc = "positive"`, the catalog is filtered to the events that are at
least $\Delta m$ larger than the previous event, following the idea of b-positive (van der
Elst, 2021). The completeness $m_{c,i}$ of each event is the magnitude of
the previous event plus $\Delta m$, and the inversion then runs as with
varying completeness. This makes the inversion robust to the short-term
incompleteness after large events. With `beta = "positive"` (implied by
`mc = "positive"`), $\beta$ is estimated with b-positive.

### Free background and free productivity (flETAS)

With `free_background`, the background rate is not constant in space: it is
estimated from the background probabilities $p_j^{\mathrm{bg}}$ of the
events, smoothed with a Gaussian kernel of squared bandwidth `bw_sq`. With
`free_productivity`, each event $e_i$ has its own productivity, estimated
from its number of direct aftershocks, $\hat l_i = \sum_j p_{ij}$, instead of the
productivity term $k_0 \, e^{a (m_i - m_{\mathrm{ref}})}$; the parameters
$k_0$ and $a$ are fitted to the individual productivities afterwards. The
two options can be used separately or together.

### Additional background term

With `bg_term`, the background rate has an additional term with rate
$\iota$ (`log10_iota`), driven by a covariate given as a column of the
catalog.

### Three dimensions

With `three_dim`, the locations are 3D Euclidean coordinates (columns `x`,
`y`, `z`), the distance $r$ is the 3D distance, and the region is the convex
hull of the points in `shape_coords`. Distances are in units of
`space_unit_in_meters` instead of km.

## Coordinates and units

- Coordinates are **(latitude, longitude)** in degrees, in this order, for
  regions (`shape_coords`) and catalogs.
- Distances are in km, areas in km², and time in days. For 3D catalogs
  (`three_dim`), distances are in units of `space_unit_in_meters`.
- `beta` is in natural-log units, $\beta = b \ln 10$.

## References

- Mizrahi, L., Nandan, S., & Wiemer, S. (2021a). The effect of
  declustering on the size distribution of mainshocks. *Seismological
  Research Letters*, 92(4), 2333-2342.
  <https://doi.org/10.1785/0220200231>
- Mizrahi, L., Nandan, S., & Wiemer, S. (2021b). Embracing data
  incompleteness for better earthquake forecasting. *Journal of
  Geophysical Research: Solid Earth*, 126(12), e2021JB022379.
  <https://doi.org/10.1029/2021JB022379>
- van der Elst, N. J. (2021). B-positive: A robust estimator of aftershock
  magnitude distribution in transiently incomplete catalogs. *Journal of
  Geophysical Research: Solid Earth*, 126(2), e2020JB021027.
  <https://doi.org/10.1029/2020JB021027>
- Veen, A., & Schoenberg, F. P. (2008). Estimation of space-time branching
  process models in seismology using an EM-type algorithm. *Journal of the
  American Statistical Association*, 103(482), 614-624.
  <https://doi.org/10.1198/016214508000000148>
