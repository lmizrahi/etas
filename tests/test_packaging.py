"""Every module must be importable with only the declared dependencies.

Optional extras are hidden by putting ``None`` into ``sys.modules`` in a
fresh interpreter, which makes ``import <name>`` raise ImportError.
"""
import subprocess
import sys

import pytest

MODULES = ["etas", "etas.inversion", "etas.simulation", "etas.evaluation",
           "etas.mc_b_est", "etas.download"]
OPTIONAL = ["seismostats", "hermes", "matplotlib", "cartopy"]


@pytest.mark.parametrize("module", MODULES)
def test_import_without_optional_extras(module):
    code = (
        "import sys\n"
        + "".join(f"sys.modules[{m!r}] = None\n" for m in OPTIONAL)
        + f"import {module}\n"
    )
    res = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_download_default_endtime_is_call_time():
    import inspect
    from etas.download import download_catalog_sed
    assert inspect.signature(download_catalog_sed).parameters[
        "endtime"].default is None
