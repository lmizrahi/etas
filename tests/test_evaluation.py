"""Characterization tests for ``etas.evaluation`` (log-likelihood scores on
a test window). Golden values in ``tests/data/golden_evaluation.json``."""
import json

import numpy as np
import pytest

from tests.conftest import DATA
from tests.generate_snapshots import evaluation_scores

GOLDEN = json.loads((DATA / "golden_evaluation.json").read_text())


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    return evaluation_scores(tmp_path_factory.mktemp("evaluation"))


@pytest.mark.parametrize("model", ["ETAS", "Poisson"])
@pytest.mark.parametrize("score", ["nll", "tll", "sll"])
def test_scores_match_golden(evaluated, model, score):
    _, scores = evaluated
    assert scores[model][score] == pytest.approx(
        GOLDEN[model][score], rel=1e-6)


def test_score_decomposition(evaluated):
    """Total = temporal + spatial log-likelihood."""
    _, scores = evaluated
    s = scores["ETAS"]
    assert -s["nll"] == pytest.approx(s["tll"] + s["sll"], abs=1e-5)
    assert scores["n_test_events"] == GOLDEN["n_test_events"]


def test_store_results_default_path(evaluated, tmp_cwd):
    ev, _ = evaluated
    ev.store_results()
    stored = json.loads((tmp_cwd / "ll_scores.json").read_text())
    assert set(stored) == {"ETAS", "Poisson"}
    assert (tmp_cwd / "augmented_catalog.csv").exists()
    assert np.isfinite(stored["ETAS"]["nll"])
