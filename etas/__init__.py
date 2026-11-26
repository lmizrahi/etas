"""ETAS: Epidemic-Type Aftershock Sequence model.

Parameter inversion (`etas.inversion`), catalog simulation and
forecasting (`etas.simulation`), evaluation on a test window
(`etas.evaluation`), completeness and b-value estimation
(`etas.mc_b_est`), and plots of the model fit (`etas.plots`).
"""
import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("etas")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "unknown"


def set_up_logger(level=logging.INFO):
    """Configure logging for scripts: timestamped messages at ``level``
    (e.g. ``logging.INFO``) on the console."""
    date_strftime_format = "%d-%m-%y %H:%M:%S"
    logging.basicConfig(
        format='%(asctime)s : %(levelname)-8s : %(name)-15s - %(message)s',
        datefmt=date_strftime_format,
        level=level)
