import datetime as dt
import urllib.request
import urllib.parse
from io import BytesIO

import pandas as pd


def download_catalog_sed(
        starttime=dt.datetime(1970, 1, 1),
        endtime=dt.datetime.now(),
        minmagnitude=0.01,
        delta_m=0.1,
):
    print('downloading data..\n')

    base_url = 'http://arclink.ethz.ch/fdsnws/event/1/query'
    params = {
        'starttime': starttime.strftime("%Y-%m-%dT%H:%M:%S"),
        'endtime': endtime.strftime("%Y-%m-%dT%H:%M:%S"),
        'minmagnitude': str(minmagnitude - delta_m / 2),
        'format': 'text'
    }
    query = urllib.parse.urlencode(params)
    url = f'{base_url}?{query}'
    response = urllib.request.urlopen(url)
    data = response.read()

    df = pd.read_csv(BytesIO(data), delimiter="|", parse_dates=['Time'])

    df.rename(
        {
            "Magnitude": "magnitude",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Time": "time",
            "Depth/km": "depth"
        }, axis=1, inplace=True)

    df.sort_values(by="time", inplace=True)

    return df
