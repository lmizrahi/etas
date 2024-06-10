import sys
from etas.inversion import ETASParameterCalculation, read_shape_coords, polygon_surface, round_half_up, parameter_dict2array, haversine
from etas.simulation import simulate_aftershock_time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from tabulate import tabulate
import numpy as np
from scipy.special import expi
import json
from scipy.integrate  import quad
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt


def compute_dist_squared_from_i(i,lat_rads: np.ndarray,long_rads: np.ndarray,earth_radius=6.3781e3):

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
    return timediff / np.timedelta64(1, 'D')



class ETASLikelihoodCalculation(ETASParameterCalculation):
    def __init__(self, metadata: dict):
        """
        Class to invert ETAS parameters.


        Parameters
        ----------
        metadata : dict
            A dict with stored metadata.

            Necessary attributes are:

            - fn_catalog: Path to the catalog. Catalog is expected to be a csv
                    file with the following columns:
                        id, latitude, longitude, time, magnitude
                    id needs to contain a unique identifier for each event
                    time contains datetime of event occurrence
                    see example_catalog.csv for an example
                    Either 'fn_catalog' or 'catalog' need to be defined.
            - catalog: Dataframe with a catalog, same requirements as for the
                    csv above apply.
                    Either 'fn_catalog' or 'catalog' need to be defined.
            - auxiliary_start (str or datetime): Start date of the auxiliary
                    catalog. Events of the auxiliary catalog act as sources,
                    not as targets.
            - timewindow_start: Start date of the primary catalog , end date
                    of auxiliary catalog (str or datetime). Events of the
                    primary catalog act as sources and as targets.
            - timewindow_end: End date of the primary catalog and begining of the testing catalog (str or datetime)
            - testwindow_end: End date of the testing catalog (str or datetime)
            - mc: Cutoff magnitude. Catalog needs to be complete above mc.
                    if mc == 'var', m_ref is required, and the catalog needs to
                    contain a column named 'mc_current'.
                    if mc == 'positive', m_ref is required, and the catalog
                    will be automatically filtered to contain only events with
                    magnitudes greater than that of the previous event,
                    following the idea of Van der Elst 2021
                    (J Geophysical Research: Solid Earth, Vol 126, Issue 2).
            - m_ref: Reference magnitude when mc is variable. Not required
                    unless mc == 'var'.
            - delta_m: Size of magnitude bins
            - coppersmith_multiplier: Events further apart from each other than
                    coppersmith subsurface rupture length * this multiplier
                    are considered to be uncorrelated (to reduce size of
                    distance matrix).
            - shape_coords: Coordinates of the boundary of the region to
                    consider, or path to a .npy file containing the coordinates
                    (list of lists, i.e. ``[[lat1, lon1], [lat2, lon2],
                    [lat3, lon3]]``). Necessary unless globe=True when calling
                    invert_etas_params(), i.e. `invert_etas_params(
                    inversion_config, globe=True)`. In this case, the whole
                    globe is considered.
            - beta: optional. If provided, beta will be fixed to this value.
                    If set to 'positive', beta will be estimated using the
                    b-positive method. Default is None.
            - theta_0: optional, initial guess for parameters. Does not affect
                    final parameters, but with a good initial guess
                    the algorithm converges faster.
            - free_background: optional, allow free_background during
                    inversion (flETAS)
                default: False
            - free_productivity: optional, allow free_productivity during
                    inversion (flETAS)
                default: False
            - bw_sq: optional, squared bandwidth of Gaussian kernel used for
                    free_background/free_productivity mode
                default: 2
            - name: optional, give the model a name
            - id: optional, give the model an ID
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

    def prepare(self,n):
        if self.preparation_done:
            self.logger.warning("Preparation already done, aborting...")
            pass

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

        self.indexes_in_test_window = self.catalog[(self.catalog.time >= self.timewindow_end) & (self.catalog.time <= self.testwindow_end)].index_from_zero.tolist()

    def filter_catalog(self, catalog):
        len_full_catalog = catalog.shape[0]

        filtered_catalog = catalog.copy()

        # filter for events in relevant timewindow
        filtered_catalog.query(
            "time >= @ self.auxiliary_start and time < @ self.testwindow_end",
            # "time >= @ self.auxiliary_start",
            inplace=True,
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
            gdf = gpd.GeoDataFrame(
                filtered_catalog,
                geometry=gpd.points_from_xy(
                    filtered_catalog.latitude, filtered_catalog.longitude
                ),
            )
            filtered_catalog = gdf[gdf.intersects(poly)].copy()
            filtered_catalog.drop("geometry", axis=1, inplace=True)
        else:
            self.area = 6.3781e3**2 * 4 * np.pi
        self.logger.info("Region has {} square km".format(self.area))
        self.logger.info(
            "{} events lie within target region.".format(len(filtered_catalog))
        )

        # filter for events above cutoff magnitude - delta_m/2
        # first sort by time, in case ETAS-positive is used
        filtered_catalog.sort_values(by="time", inplace=True)
        if self.delta_m > 0:
            filtered_catalog["magnitude"] = (
                round_half_up(filtered_catalog["magnitude"] / self.delta_m)
                * self.delta_m
            )
        if self.mc == "var":
            assert "mc_current" in filtered_catalog.columns, self.logger.error(
                'Need column "mc_current" in ' 'catalog when mc is set to "var".'
            )
        elif self.mc == "positive":
            filtered_catalog["mc_current"] = (
                filtered_catalog["magnitude"].shift(1) + self.delta_m
            )
            if self.delta_m > 0:
                filtered_catalog["mc_current"] = (
                    round_half_up(filtered_catalog["mc_current"] / self.delta_m)
                    * self.delta_m
                )
        else:
            filtered_catalog["mc_current"] = self.mc
        filtered_catalog.query("magnitude >= mc_current", inplace=True)
        self.logger.info(
            "{} events are above completeness.".format(len(filtered_catalog))
        )

        return filtered_catalog

    def aftershock_number(self, m):
        return self.k0 * np.exp(self.a * (m - self.mc))

    def time_decay(self, time_distance):
        return np.exp(-time_distance / self.tau) / np.power((time_distance + self.c), (1 + self.omega))

    def aftershock_zone(self, m):
        return self.d * np.exp(self.gamma * (m - self.mc))

    def space_decay(self, spatial_distance_squared, m):
        return 1 / np.power((spatial_distance_squared + self.aftershock_zone(m)), (1 + self.rho))

    def triggering_kernel(self, t, x, m):
        return self.aftershock_number(m) * self.space_decay(x, m) * self.time_decay(t)

    def space_integral(self, m):
        return np.pi/(self.rho * np.power(self.aftershock_zone(m), self.rho))


    def integral(self, x_values):
        integral_values = []
        cumulative_integral = 0.0
        for i in range(1,len(x_values)):
            result, _ = quad(self.time_decay, x_values[i-1], x_values[i])
            cumulative_integral += result
            integral_values.append(cumulative_integral)
        return np.array(integral_values)


    def _precompute_integral(self, n):
        print('Precomputing Intgral')

        t_values = simulate_aftershock_time(self.log10_c, self.omega, self.log10_tau, size=n)
        t_values = np.append([0], np.sort(t_values))

        integral_values = self.integral(t_values)

        print('Done')

        return np.array(t_values[1:]),np.array(integral_values)

    def integral_time_decay(self, t_values):
        # Interpolate the precomputed integral values
        return np.interp(t_values, self.time_mesh, self.integral_values)


    
    def Lambda(self): ### returns vecotr \int_{t_{i-1}}^{t_i} \lambda*(s)ds for each i in the test sequence

        int_lambda_star = np.zeros_like(self.magnitudes)

        def calculate_value(i):

            background_term = self.area*(to_days(self.times[i]-self.auxiliary_start.to_numpy()))*self.mu

            triggering_term = (self.aftershock_number(self.magnitudes[:i])
            *self.space_integral(self.magnitudes[:i])
            *(self.integral_time_decay(to_days(self.times[i]-self.times[:i])))).sum()


            return background_term+triggering_term

        # Parallelize 
        int_0_to_ti = np.array(Parallel(n_jobs=-1)(delayed(calculate_value)(i) for i in self.indexes_in_test_window))
        

        #### int_0_to_test_start

        index_of_first_in_test_window = self.indexes_in_test_window[0]

        background_term = self.area*(to_days(self.timewindow_end.to_numpy()-self.auxiliary_start.to_numpy()))*self.mu

        triggering_term = (self.aftershock_number(self.magnitudes[:index_of_first_in_test_window])
            *self.space_integral(self.magnitudes[:index_of_first_in_test_window])
            *(self.integral_time_decay(to_days(self.times[index_of_first_in_test_window]-self.times[:index_of_first_in_test_window])))).sum()

        int_0_to_test_start = triggering_term+ background_term

        int_lambda_star[self.indexes_in_test_window]  = np.ediff1d(int_0_to_ti,to_begin=int_0_to_ti[0]-int_0_to_test_start)

        return int_lambda_star


    def lambd(self): ## returns \lambda*(t,x) for each i in test interval

        lam = np.zeros_like(self.magnitudes)

        def calculate_value(i):

            background_term = self.mu

            triggering_term = self.triggering_kernel(to_days(self.times[i]-self.times[:i]),
                compute_dist_squared_from_i(i,self.lat_rads,self.long_rads),
                self.magnitudes[:i]).sum()

            return background_term+triggering_term

        lam[self.indexes_in_test_window] = Parallel(n_jobs=-1)(delayed(calculate_value)(i) for i in self.indexes_in_test_window)

        return lam


    def lambd_star(self): ##returns \lambda*(t) for each i in test interval

        lam = np.zeros_like(self.magnitudes)

        def calculate_value(i):

            background_term = self.mu*self.area

            triggering_term = (self.aftershock_number(self.magnitudes[:i])
                *self.space_integral(self.magnitudes[:i])
                *self.time_decay(to_days(self.times[i]-self.times[:i]))).sum()

            return background_term+triggering_term

        lam[self.indexes_in_test_window] = Parallel(n_jobs=-1)(delayed(calculate_value)(i) for i in self.indexes_in_test_window)

        return lam


    def find_poisson_mle(self):

        self.training_window_length = to_days(self.timewindow_end.to_numpy()-self.auxiliary_start.to_numpy())
        self.training_window_number_events = ((self.times >= self.auxiliary_start.to_numpy()) & (self.times <= self.timewindow_end.to_numpy())).sum()

        self.mu_poisson = self.training_window_number_events/(self.area *self.training_window_length)

    def evaluate_baseline_poisson_model(self):

        self.find_poisson_mle()

        self.testing_window_length = to_days(self.testwindow_end.to_numpy()-self.timewindow_end.to_numpy())
        self.testing_window_number_events = ((self.times >= self.timewindow_end.to_numpy()) & (self.times <= self.testwindow_end.to_numpy())).sum()

        self.poisson_nll = self.area*self.mu_poisson*self.testing_window_length/self.testing_window_number_events - np.log(self.mu_poisson)
        self.poisson_tll = np.log(self.mu_poisson*self.area) - self.area*self.mu_poisson*self.testing_window_length/self.testing_window_number_events
        self.poisson_sll = -self.poisson_nll-self.poisson_tll

        self.Poisson_scores = {
        "nll":self.poisson_nll,
        "tll":self.poisson_tll,
        "sll":self.poisson_sll,
        }

        print('+----------+---Poisson--+------------+')
        print(tabulate([[str(self.poisson_nll),str(self.poisson_sll), str(self.poisson_tll)]], ["nll", "sll","tll"], tablefmt="grid"))

    def evaluate(self):

        self.int_lambd = self.Lambda()

        self.lambd = self.lambd()
        self.lambd_star = self.lambd_star()

        self.LL= np.log(self.lambd) - self.int_lambd
        self.TLL = np.log(self.lambd_star) - self.int_lambd
        self.SLL = self.LL - self.TLL

        self.nll = -self.LL[self.indexes_in_test_window].mean()
        self.tll = self.TLL[self.indexes_in_test_window].mean()
        self.sll = self.SLL[self.indexes_in_test_window].mean()
        assert -self.nll -(self.tll+self.sll)<1e-5
        
        print('area: ',self.area )

        self.ETAS_scores = {
        "nll":self.nll,
        "tll":self.tll,
        "sll":self.sll,
        }

        print('+----------+----ETAS----+------------+')
        print(tabulate([[str(self.nll),str(self.sll), str(self.tll)]], ["nll", "sll","tll"], tablefmt="grid"))

        return self.ETAS_scores



    def store_results(self, data_path=""):
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
