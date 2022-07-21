import json
import datetime as dt
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import pprint

from etas.simulation import simulate_catalog_continuation
from etas.inversion import (
    parameter_dict2array,
    read_shape_coords,
    round_half_up)
from etas.inversion import invert_etas_params

if __name__ == '__main__':

    # reads configuration for example ETAS parameter inversion
    with open('../config/ch_forecast_config.json', 'r') as f:
        forecast_config = json.load(f)

    forecast_config['timewindow_end'] = dt.datetime.now()

    parameters = invert_etas_params(forecast_config)

    aux_start = pd.to_datetime(forecast_config['auxiliary_start'])
    prim_start = pd.to_datetime(forecast_config['timewindow_start'])

    # end of training period is start of forecasting period
    forecast_start_date = pd.to_datetime(forecast_config['timewindow_end'])
    forecast_end_date = forecast_start_date + \
        dt.timedelta(days=int(forecast_config['forecast_duration']))

    coordinates = read_shape_coords(forecast_config['shape_coords'])
    poly = Polygon(coordinates)

    fn_train_catalog = forecast_config['fn_catalog']
    delta_m = forecast_config['delta_m']
    m_ref = forecast_config.get('m_ref', forecast_config['mc'])

    beta = parameters['beta']

    # read in correct ETAS parameters to be used for simulation
    theta = parameter_dict2array(parameters)
    theta_without_mu = theta[1:]
    pprint.pprint(parameters)

    # read training catalog and source info (contains current rate needed for
    # inflation factor calculation)
    catalog = pd.read_csv(fn_train_catalog,
                          index_col=0,
                          parse_dates=['time'],
                          dtype={'url': str, 'alert': str})

    sources = pd.read_csv(forecast_config['fn_src'], index_col=0)

    # xi_plus_1 is aftershock productivity inflation factor. not used here.
    sources['xi_plus_1'] = 1

    catalog = pd.merge(
        sources,
        catalog[['latitude', 'longitude', 'time', 'magnitude']],
        left_index=True,
        right_index=True,
        how='left',
    )

    assert len(catalog) == len(sources), \
        'lost/found some sources in the merge! ' + \
        str(len(catalog)) + ' -- ' + str(len(sources))
    assert catalog.magnitude.min() == m_ref, \
        'smallest magnitude in sources is ' + str(catalog.magnitude.min()) \
        + ' but I am supposed to simulate above ' + str(m_ref)

    # background rates
    ip = pd.read_csv(forecast_config['fn_ip'], index_col=0)
    ip.query('magnitude>=@m_ref -@delta_m/2', inplace=True)
    ip = gpd.GeoDataFrame(
        ip, geometry=gpd.points_from_xy(ip.latitude, ip.longitude))
    ip = ip[ip.intersects(poly)]

    # other constants
    coppersmith_multiplier = forecast_config['coppersmith_multiplier']
    earth_radius = forecast_config.get('earth_radius', 6.3781e3)

    print(f'm ref: {m_ref} min magnitude in training '
          f'catalog: {catalog["magnitude"].min()}')

    start = dt.datetime.now()

    # using 100 only for testing purposes
    # for simulation_i in range(100000):
    for simulation_i in range(100):
        continuation = simulate_catalog_continuation(
            catalog,
            auxiliary_start=aux_start,
            auxiliary_end=forecast_start_date,
            polygon=poly,
            simulation_end=forecast_end_date,
            parameters=parameters,
            mc=m_ref - delta_m / 2,
            beta_main=beta,
            verbose=False,
            background_lats=ip['latitude'],
            background_lons=ip['longitude'],
            background_probs=ip['P_background'],
            gaussian_scale=0.1
        )
        continuation.query(
            'time>=@forecast_start_date and '
            'time<=@forecast_end_date and magnitude >= @m_ref-@delta_m/2',
            inplace=True)

        print(
            f'took {dt.datetime.now() - start} to simulate {str(simulation_i)}'
            f' catalogs.\n The latest one contains {len(continuation)} events.')

        continuation.magnitude = round_half_up(continuation.magnitude, 1)
        continuation.index.name = 'id'

        print('store catalog..')
        os.makedirs(os.path.dirname(
            forecast_config['fn_store_simulation']), exist_ok=True)

        output_cols = [
            'latitude',
            'longitude',
            'time',
            'magnitude',
            'is_background']

        continuation[output_cols].sort_values(by='time').to_csv(
            f'{forecast_config["fn_store_simulation"]+str(simulation_i)}.csv')

    print('\nDONE!')
