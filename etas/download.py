"""Download of earthquake catalogs."""
import datetime as dt
import urllib.request
import urllib.parse
from io import BytesIO

import pandas as pd


def download_catalog_sed(
    starttime=dt.datetime(1970, 1, 1),
    endtime=None,
    minmagnitude=0.01,
    delta_m=0.1,
):
    """Download the Swiss Seismological Service (SED) earthquake catalog.

    Queries the FDSN event web service of ETH Zurich.

    Parameters
    ----------
    starttime : datetime, default 1970-01-01
    endtime : datetime, optional
        Default is the time of the call.
    minmagnitude : float, default 0.01
        Minimum (binned) magnitude; events from ``minmagnitude - delta_m / 2``
        are requested.
    delta_m : float, default 0.1
        Magnitude bin size.

    Returns
    -------
    pd.DataFrame
        With columns time, latitude, longitude, depth, magnitude (and the
        other columns of the service), sorted by time.
    """
    if endtime is None:
        endtime = dt.datetime.now()
    print("downloading data..\n")

    base_url = "http://eida.ethz.ch/fdsnws/event/1/query"
    params = {
        "starttime": starttime.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": endtime.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": str(minmagnitude - delta_m / 2),
        "format": "text",
    }
    query = urllib.parse.urlencode(params)
    url = f"{base_url}?{query}"
    response = urllib.request.urlopen(url)
    data = response.read()

    df = pd.read_csv(BytesIO(data), delimiter="|", parse_dates=["Time"])

    df = df.rename(
        {
            "Magnitude": "magnitude",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Time": "time",
            "Depth/km": "depth",
        },
        axis=1,
    )

    df = df.sort_values(by="time")

    return df
