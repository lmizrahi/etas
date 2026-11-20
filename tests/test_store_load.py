"""``store_results`` output must remain loadable by ``load_calculation``."""
import json

import pytest

from etas.inversion import ETASParameterCalculation


def test_store_and_reload(short_inversion, tmp_path):
    calc = short_inversion
    calc.store_results(str(tmp_path) + "/", store_pij=True)
    meta = json.loads((tmp_path / "parameters_short.json").read_text())
    for key in ["final_parameters", "beta", "n_hat", "area",
                "fn_ip", "fn_src", "fn_pij", "shape_coords"]:
        assert key in meta

    loaded = ETASParameterCalculation.load_calculation(meta)
    assert loaded.theta == pytest.approx(calc.theta)
    assert loaded.beta == calc.beta
    assert len(loaded.pij) == len(calc.pij)
    assert len(loaded.source_events) == len(calc.source_events)
    assert len(loaded.target_events) == len(calc.target_events)


def test_stored_parameter_ranges_are_labelled_correctly(short_inversion,
                                                        tmp_path):
    from etas import inversion
    short_inversion.store_results(str(tmp_path) + "/")
    meta = json.loads((tmp_path / "parameters_short.json").read_text())
    for name in ["log10_mu", "log10_iota", "log10_k0", "a", "log10_c",
                 "omega", "log10_tau", "log10_d", "gamma", "rho"]:
        expected = getattr(inversion, name.upper() + "_RANGE")
        assert tuple(meta[name + "_range"]) == expected, name
