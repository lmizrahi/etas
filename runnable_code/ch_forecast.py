import json
import datetime as dt
import logging
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from etas import set_up_logger

from etas.simulation import simulate_catalog_continuation
from etas.inversion import read_shape_coords, round_half_up
from etas.inversion import ETASParameterCalculation

set_up_logger(level=logging.DEBUG)

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    # reads configuration for example ETAS parameter inversion
    with open('../config/ch_forecast_config.json', 'r') as f:
        forecast_config = json.load(f)

    # sets training period end to today
    # disabled because the catalog provided is not automatically updated
    # forecast_config['timewindow_end'] = dt.datetime.now()

    etas_invert = ETASParameterCalculation(forecast_config)

    etas_invert.prepare()

    theta = etas_invert.invert()

    etas_invert.store_results(forecast_config['data_path'])

    aux_start = pd.to_datetime(forecast_config['auxiliary_start'])
    prim_start = pd.to_datetime(forecast_config['timewindow_start'])

    # # end of training period is start of forecasting period
    forecast_start_date = pd.to_datetime(forecast_config['timewindow_end'])
    forecast_end_date = forecast_start_date + \
        dt.timedelta(days=int(forecast_config['forecast_duration']))

    coordinates = read_shape_coords(forecast_config['shape_coords'])
    poly = Polygon(coordinates)

    fn_train_catalog = forecast_config['fn_catalog']
    delta_m = forecast_config['delta_m']
    m_ref = forecast_config.get('m_ref', forecast_config['mc'])
    beta = etas_invert.beta

    # read training catalog and source info (contains current rate needed for
    # inflation factor calculation)
    catalog = pd.read_csv(fn_train_catalog,
                          index_col=0,
                          parse_dates=['time'],
                          dtype={'url': str, 'alert': str})

    catalog = pd.merge(
        etas_invert.source_events,
        catalog[['latitude', 'longitude', 'time', 'magnitude']],
        left_index=True,
        right_index=True,
        how='left',
    )

    assert len(catalog) == len(etas_invert.source_events), \
        f'lost/found some sources in the merge! {len(catalog)}' \
        f' -- {len(etas_invert.source_events)}'
    assert catalog.magnitude.min() == m_ref, \
        f'smallest magnitude in sources is {catalog.magnitude.min()}' \
        f' but I am supposed to simulate above {m_ref}'

    # background rates
    ip = etas_invert.target_events
    ip.query('magnitude>=@m_ref -@delta_m/2', inplace=True)
    ip = gpd.GeoDataFrame(
        ip, geometry=gpd.points_from_xy(ip.latitude, ip.longitude))
    ip = ip[ip.intersects(poly)]

    logger.info(f'm ref: {m_ref} min magnitude in training '
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
            parameters=theta,
            mc=m_ref - delta_m / 2,
            beta_main=beta,
            background_lats=ip['latitude'],
            background_lons=ip['longitude'],
            background_probs=ip['P_background'],
            gaussian_scale=0.1
        )
        continuation.query(
            'time>=@forecast_start_date and '
            'time<=@forecast_end_date and magnitude >= @m_ref-@delta_m/2',
            inplace=True)

        logger.debug(
            f'took {dt.datetime.now() - start} to simulate {simulation_i}'
            f' catalogs. The latest one contains {len(continuation)} '
            'events.')

        continuation.magnitude = round_half_up(continuation.magnitude, 1)
        continuation.index.name = 'id'

        logger.info(f'store catalog {simulation_i}..')
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

    logger.info('DONE!')
