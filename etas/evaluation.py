r"""Evaluation of a fitted ETAS model on a test window.

Given the output of an inversion (``ETASParameterCalculation.store_results``),
`ETASLikelihoodCalculation` evaluates the fitted model on the events
$e_i$ in the test window ``(timewindow_end, testwindow_end]``. For each of
them, with location $(x_i, y_i)$ and occurrence time $t_i$, the
event-based log-likelihood is

$$
\mathrm{LL}_i = \log \lambda^*(x_i, y_i, t_i)
    - \int_{t_{i-1}}^{t_i} \iint_R \lambda^*(x, y, s)\, dx\, dy\, ds,
$$

where $\lambda^*(x, y, t)$ is the conditional intensity of the model
(events per km² and day, given all events before $t$, see
`etas.inversion`), $t_{i-1}$ is the occurrence time of the previous event
$e_{i-1}$, and $R$ is the region. The first term rewards a high rate where
and when the event occurred; the second term penalizes the number of
events the model expected between $e_{i-1}$ and $e_i$. The temporal part
``TLL`` is the same with the space-integrated rate
$\lambda^*(t) = \iint_R \lambda^*(x, y, t)\, dx\, dy$ in place of
$\lambda^*(x_i, y_i, t_i)$, and the spatial part is ``SLL = LL - TLL``.
The averages over the events in the test window are reported as ``nll``
(negative ``LL``), ``tll`` and ``sll``, and compared to a homogeneous
Poisson model fitted to the training period.

Units: distances in km, time in days, rates per km² and day.
"""
import json
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import quad
from shapely.geometry import Polygon
from tabulate import tabulate

from etas.inversion import (ETASParameterCalculation, haversine,
                            parameter_dict2array, polygon_surface,
                            read_shape_coords, round_half_up)
from etas.simulation import simulate_aftershock_time

logger = logging.getLogger(__name__)


def compute_dist_squared_from_i(i, lat_rads: np.ndarray,
                                long_rads: np.ndarray,
                                earth_radius=6.3781e3):
    """Squared great-circle distances from event ``i`` to all earlier events.

    Parameters
    ----------
    i : int
        Position of the event in the (time-sorted) arrays.
    lat_rads, long_rads : np.ndarray
        Latitudes and longitudes of all events, in radians.
    earth_radius : float
        Earth radius in km.

    Returns
    -------
    np.ndarray
        Squared distances in km² to events ``0, ..., i - 1``.
    """
    return np.square(
        haversine(
            lat_rads[i],
            lat_rads[:i],
            long_rads[i],
            long_rads[:i],
            earth_radius,
        )
    )


def to_days(timediff):
    """Convert numpy ``timedelta64`` values to (fractional) days."""
    return timediff / np.timedelta64(1, 'D')


class ETASLikelihoodCalculation(ETASParameterCalculation):
    def __init__(self, metadata: dict):
        """Evaluate fitted ETAS parameters on a test window.

        Parameters
        ----------
        metadata : dict
            The content of the ``parameters_<id>.json`` file written by
            `ETASParameterCalculation.store_results`, which contains
            the catalog path, time windows, region, ``beta``, ``area`` and
            ``final_parameters``. In addition, ``testwindow_end`` (end of
            the test window, str or datetime) must be set; the test window
            starts at ``timewindow_end``, the end of the training period.

        Notes
        -----
        Usage:

            calc = ETASLikelihoodCalculation(metadata)
            calc.prepare(n=1_000_000)
            calc.evaluate_baseline_poisson_model()
            scores = calc.evaluate()   # {"nll": ..., "tll": ..., "sll": ...}
            calc.store_results(data_path)

        Known limitation: the kernels here use ``mc`` as reference
        magnitude, whereas the inversion uses ``m_ref - delta_m / 2``; with
        ``mc = 'var'`` or ``'positive'`` the evaluation does not work.
        Results are kept unchanged on purpose (used by earthquakeNPP); a
        correction would be a separate, reviewed change.
        """

        super().__init__(metadata)

        self.testwindow_end = pd.to_datetime(metadata["testwindow_end"])
        self.area = metadata["area"]
        self.beta = metadata["beta"]

        self.parameters = parameter_dict2array(metadata["final_parameters"])

        (
            self.log10_mu,
            self.log10_iota,
            self.log10_k0,
            self.a,
            self.log10_c,
            self.omega,
            self.log10_tau,
            self.log10_d,
            self.gamma,
            self.rho,
        ) = self.parameters

        self.mu = np.power(10, self.log10_mu)
        self.k0 = np.power(10, self.log10_k0)
        self.c = np.power(10, self.log10_c)
        self.d = np.power(10, self.log10_d)
        self.tau = np.power(10, self.log10_tau)

        self.alpha = self.a - self.rho * self.gamma

    def prepare(self, n):
        """Filter the catalog and precompute the time-kernel integral.

        Parameters
        ----------
        n : int
            Number of points at which the integral of the time kernel is
            precomputed (see `_precompute_integral`). More points give
            a more accurate integral; 10⁶ is a typical value.
        """
        if self.preparation_done:
            # unlike the inversion, a repeated call redoes the preparation
            # (kept as is: downstream code, e.g. earthquakeNPP, relies on
            # unchanged results)
            self.logger.warning("Preparation already done, repeating...")

        self.logger.info("PREPARING {}".format(self.name))
        self.logger.info("  filtering catalog...")
        self.catalog = self.filter_catalog(self.catalog)

        self.preparation_done = True

        self.catalog = self.catalog.sort_values(by='time')

        self.catalog['index_from_zero'] = range(len(self.catalog))

        self.times = self.catalog.time.to_numpy()
        self.magnitudes = self.catalog.magnitude.to_numpy()
        self.latitudes = self.catalog.latitude.to_numpy()
        self.longitudes = self.catalog.longitude.to_numpy()
        self.lat_rads = np.radians(self.latitudes)
        self.long_rads = np.radians(self.longitudes)

        self.time_mesh, self.integral_values = self._precompute_integral(n)

        in_test_window = (
            (self.catalog.time >= self.timewindow_end)
            & (self.catalog.time <= self.testwindow_end)
        )
        self.indexes_in_test_window = \
            self.catalog[in_test_window].index_from_zero.tolist()

    def filter_catalog(self, catalog):
        """Keep events between ``auxiliary_start`` and ``testwindow_end``,
        inside the region and above completeness.

        Same as `ETASParameterCalculation.filter_catalog`, but up to
        the end of the test window instead of the training window.
        """
        len_full_catalog = catalog.shape[0]

        filtered_catalog = catalog.copy()

        # filter for events in relevant timewindow
        filtered_catalog = filtered_catalog.query(
            "time >= @ self.auxiliary_start and time < @ self.testwindow_end",
        )
        self.logger.info(
            "  {} out of {} events are within time window.".format(
                filtered_catalog.shape[0], len_full_catalog
            )
        )

        # filter for events in region of interest
        if self.shape_coords is not None:
            self.shape_coords = read_shape_coords(self.shape_coords)

            self.logger.info(
                "  Coordinates of region: {}".format(list(self.shape_coords))
            )

            poly = Polygon(self.shape_coords)
            self.area = polygon_surface(poly)
            # coordinates are (lat, lon) throughout the package
            gdf = gpd.GeoDataFrame(
                filtered_catalog,
                geometry=gpd.points_from_xy(
                    filtered_catalog.latitude, filtered_catalog.longitude
                ),
            )
            filtered_catalog = gdf[gdf.intersects(poly)].copy()
            filtered_catalog = filtered_catalog.drop("geometry", axis=1)
        else:
            self.area = 6.3781e3**2 * 4 * np.pi
        self.logger.info("Region has {} square km".format(self.area))
        self.logger.info(
            "{} events lie within target region.".format(len(filtered_catalog))
        )

        # filter for events above cutoff magnitude - delta_m/2
        # first sort by time, in case ETAS-positive is used
        filtered_catalog = filtered_catalog.sort_values(by="time")
        if self.delta_m > 0:
            filtered_catalog["magnitude"] = (
                round_half_up(filtered_catalog["magnitude"] / self.delta_m)
                * self.delta_m
            )
        if self.mc == "var":
            assert "mc_current" in filtered_catalog.columns, self.logger.error(
                'Need column "mc_current" in catalog when mc is set to "var".'
            )
        elif self.mc == "positive":
            filtered_catalog["mc_current"] = (
                filtered_catalog["magnitude"].shift(1) + self.delta_m
            )
            if self.delta_m > 0:
                filtered_catalog["mc_current"] = (
                    round_half_up(
                        filtered_catalog["mc_current"] / self.delta_m)
                    * self.delta_m
                )
        else:
            filtered_catalog["mc_current"] = self.mc
        filtered_catalog = filtered_catalog.query("magnitude >= mc_current")
        self.logger.info(
            "{} events are above completeness.".format(len(filtered_catalog))
        )

        return filtered_catalog

    # --- kernel components (time in days, distances in km) ----------------

    def aftershock_number(self, m):
        """Productivity ``k0 * exp(a * (m - mc))`` of events of magnitude
        ``m``."""
        return self.k0 * np.exp(self.a * (m - self.mc))

    def time_decay(self, time_distance):
        """Tapered Omori kernel ``exp(-t / tau) / (t + c)^(1 + omega)``."""
        return (np.exp(-time_distance / self.tau)
                / np.power((time_distance + self.c), (1 + self.omega)))

    def aftershock_zone(self, m):
        """Size ``d * exp(gamma * (m - mc))`` (km²) of the aftershock zone of
        events of magnitude ``m``."""
        return self.d * np.exp(self.gamma * (m - self.mc))

    def space_decay(self, spatial_distance_squared, m):
        """Spatial kernel ``1 / (r² + d_m)^(1 + rho)``, with ``d_m`` the
        aftershock zone size."""
        return 1 / np.power(
            (spatial_distance_squared + self.aftershock_zone(m)),
            (1 + self.rho))

    def triggering_kernel(self, t, x, m):
        """Triggering rate density at time distance ``t`` (days) and squared
        distance ``x`` (km²) from a source of magnitude ``m``."""
        return (self.aftershock_number(m) * self.space_decay(x, m)
                * self.time_decay(t))

    def space_integral(self, m):
        """Integral of `space_decay` over the plane,
        ``pi / (rho * d_m^rho)``."""
        return np.pi / (self.rho * np.power(self.aftershock_zone(m),
                                            self.rho))

    def integral(self, x_values):
        """Cumulative numerical integral of `time_decay` from
        ``x_values[0]`` to each of ``x_values[1:]``."""
        integral_values = []
        cumulative_integral = 0.0
        for i in range(1, len(x_values)):
            result, _ = quad(self.time_decay, x_values[i - 1], x_values[i])
            cumulative_integral += result
            integral_values.append(cumulative_integral)
        return np.array(integral_values)

    def _precompute_integral(self, n):
        """Tabulate the integral of the time kernel from 0 at ``n`` points.

        The points are sampled from the time kernel itself, so that they are
        dense where the kernel changes quickly.
        """
        print('Precomputing Intgral')

        t_values = simulate_aftershock_time(
            self.log10_c, self.omega, self.log10_tau, size=n)
        t_values = np.append([0], np.sort(t_values))

        integral_values = self.integral(t_values)

        print('Done')

        return np.array(t_values[1:]), np.array(integral_values)

    def integral_time_decay(self, t_values):
        """Integral of `time_decay` from 0 to ``t_values``,
        interpolated from the precomputed table."""
        return np.interp(t_values, self.time_mesh, self.integral_values)

    # --- intensities -------------------------------------------------------

    def Lambda(self):
        """Integrated intensity between consecutive events.

        Returns
        -------
        np.ndarray
            For each event $e_i$ in the test window, the expected number of
            events between the previous event $e_{i-1}$ and $e_i$,
            $\\int_{t_{i-1}}^{t_i} \\int_R \\lambda^*(s, x)\\,dx\\,ds$,
            where $t_i$ is the occurrence time of $e_i$ and $R$ the region;
            zero for events outside the test window.
        """
        int_lambda_star = np.zeros_like(self.magnitudes)
        aux_start = self.auxiliary_start.to_numpy()

        def integral_up_to(i, t_end):
            # expected number of events between auxiliary_start and t_end,
            # triggered by the events before position i, plus background
            background_term = self.area * to_days(t_end - aux_start) * self.mu
            m = self.magnitudes[:i]
            triggering_term = (
                self.aftershock_number(m)
                * self.space_integral(m)
                * self.integral_time_decay(to_days(t_end - self.times[:i]))
            ).sum()
            return background_term + triggering_term

        def calculate_value(i):
            return integral_up_to(i, self.times[i])

        int_0_to_ti = np.array(Parallel(n_jobs=-1)(
            delayed(calculate_value)(i) for i in self.indexes_in_test_window))

        # integral from auxiliary_start to the start of the test window
        first = self.indexes_in_test_window[0]
        background_term = self.area * self.mu * to_days(
            self.timewindow_end.to_numpy() - aux_start)
        m = self.magnitudes[:first]
        triggering_term = (
            self.aftershock_number(m)
            * self.space_integral(m)
            * self.integral_time_decay(
                to_days(self.times[first] - self.times[:first]))
        ).sum()
        int_0_to_test_start = triggering_term + background_term

        int_lambda_star[self.indexes_in_test_window] = np.ediff1d(
            int_0_to_ti, to_begin=int_0_to_ti[0] - int_0_to_test_start)

        return int_lambda_star

    def lambd(self):
        """Conditional intensity $\\lambda^*(x_i, y_i, t_i)$ (per km² and
        day) at the location $(x_i, y_i)$ and occurrence time $t_i$ of each
        event $e_i$ in the test window; zero elsewhere.

        Note: `evaluate` replaces this method by its result array.
        """
        lam = np.zeros_like(self.magnitudes)

        def calculate_value(i):
            background_term = self.mu
            triggering_term = self.triggering_kernel(
                to_days(self.times[i] - self.times[:i]),
                compute_dist_squared_from_i(i, self.lat_rads, self.long_rads),
                self.magnitudes[:i],
            ).sum()
            return background_term + triggering_term

        lam[self.indexes_in_test_window] = Parallel(n_jobs=-1)(
            delayed(calculate_value)(i) for i in self.indexes_in_test_window)

        return lam

    def lambd_star(self):
        """Space-integrated intensity $\\lambda^*(t_i)$ (per day), that is,
        the expected number of events per day in the whole region, at the
        occurrence time $t_i$ of each event $e_i$ in the test window; zero
        elsewhere.

        Note: `evaluate` replaces this method by its result array.
        """
        lam = np.zeros_like(self.magnitudes)

        def calculate_value(i):
            background_term = self.mu * self.area
            m = self.magnitudes[:i]
            triggering_term = (
                self.aftershock_number(m)
                * self.space_integral(m)
                * self.time_decay(to_days(self.times[i] - self.times[:i]))
            ).sum()
            return background_term + triggering_term

        lam[self.indexes_in_test_window] = Parallel(n_jobs=-1)(
            delayed(calculate_value)(i) for i in self.indexes_in_test_window)

        return lam

    # --- scores ------------------------------------------------------------

    def find_poisson_mle(self):
        """Rate (per km² and day) of a homogeneous Poisson model fitted to
        the training period ``[auxiliary_start, timewindow_end]``."""
        aux_start = self.auxiliary_start.to_numpy()
        tw_end = self.timewindow_end.to_numpy()
        self.training_window_length = to_days(tw_end - aux_start)
        self.training_window_number_events = (
            (self.times >= aux_start) & (self.times <= tw_end)).sum()

        self.mu_poisson = self.training_window_number_events / (
            self.area * self.training_window_length)

    def evaluate_baseline_poisson_model(self):
        """Scores of the homogeneous Poisson model on the test window.

        Sets ``self.Poisson_scores = {"nll", "tll", "sll"}`` (per event).
        """
        self.find_poisson_mle()

        tw_end = self.timewindow_end.to_numpy()
        test_end = self.testwindow_end.to_numpy()
        self.testing_window_length = to_days(test_end - tw_end)
        self.testing_window_number_events = (
            (self.times >= tw_end) & (self.times <= test_end)).sum()

        expected_per_event = (
            self.area * self.mu_poisson * self.testing_window_length
            / self.testing_window_number_events)
        self.poisson_nll = expected_per_event - np.log(self.mu_poisson)
        self.poisson_tll = (np.log(self.mu_poisson * self.area)
                            - expected_per_event)
        self.poisson_sll = -self.poisson_nll - self.poisson_tll

        self.Poisson_scores = {
            "nll": self.poisson_nll,
            "tll": self.poisson_tll,
            "sll": self.poisson_sll,
        }

        print('+----------+---Poisson--+------------+')
        print(tabulate(
            [[str(self.poisson_nll), str(self.poisson_sll),
              str(self.poisson_tll)]],
            ["nll", "sll", "tll"], tablefmt="grid"))

    def evaluate(self):
        """Event-based log-likelihood scores of the ETAS model.

        Returns
        -------
        dict
            ``{"nll": ..., "tll": ..., "sll": ...}``: the negative total,
            the temporal and the spatial log-likelihood, averaged over the
            events in the test window (``-nll = tll + sll``). Also stored
            as ``self.ETAS_scores``; per-event values are in ``self.LL``,
            ``self.TLL``, ``self.SLL``.
        """
        self.int_lambd = self.Lambda()

        self.lambd = self.lambd()
        self.lambd_star = self.lambd_star()

        self.LL = np.log(self.lambd) - self.int_lambd
        self.TLL = np.log(self.lambd_star) - self.int_lambd
        self.SLL = self.LL - self.TLL

        self.nll = -self.LL[self.indexes_in_test_window].mean()
        self.tll = self.TLL[self.indexes_in_test_window].mean()
        self.sll = self.SLL[self.indexes_in_test_window].mean()
        assert -self.nll - (self.tll + self.sll) < 1e-5

        print('area: ', self.area)

        self.ETAS_scores = {
            "nll": self.nll,
            "tll": self.tll,
            "sll": self.sll,
        }

        print('+----------+----ETAS----+------------+')
        print(tabulate(
            [[str(self.nll), str(self.sll), str(self.tll)]],
            ["nll", "sll", "tll"], tablefmt="grid"))

        return self.ETAS_scores

    def store_results(self, data_path=""):
        """Write ``augmented_catalog.csv`` (catalog with per-event
        intensities and scores) and ``ll_scores.json`` (ETAS and Poisson
        scores) to ``data_path`` (default: current directory)."""
        if data_path == "":
            data_path = os.getcwd() + "/"

        self.logger.info("  Data will be stored in {}".format(data_path))

        aug_catalog_filepath = data_path + "augmented_catalog.csv"

        self.catalog['int_lambd'] = self.int_lambd
        self.catalog['lambd'] = self.lambd
        self.catalog['lambd_star'] = self.lambd_star
        self.catalog['LL'] = self.LL
        self.catalog['TLL'] = self.TLL
        self.catalog['SLL'] = self.SLL

        self.catalog.to_csv(aug_catalog_filepath)

        scores_filepath = data_path + "ll_scores.json"

        scores = {
            "ETAS": self.ETAS_scores,
            "Poisson": self.Poisson_scores,
        }

        with open(scores_filepath, "w") as f:
            f.write(json.dumps(scores))
