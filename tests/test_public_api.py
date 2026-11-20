"""Backwards-compatibility guard for the public API.

Fails if a public function, class, method or constant that existed when
``tests/data/public_api.json`` was generated disappears, or if an existing
parameter is renamed, reordered, changes kind, or changes its default.
Adding new names, or new parameters *with defaults* at the end of a
signature, is allowed.
"""
import json

import pytest

from tests.api_snapshot import collect
from tests.conftest import DATA

SNAPSHOT = json.loads((DATA / "public_api.json").read_text())
LIVE = collect()

CASES = [(m, n) for m, entries in SNAPSHOT.items() for n in entries]


def _check_signature(old, new, where):
    assert new is not None, f"{where}: signature no longer inspectable"
    assert len(new) >= len(old), f"{where}: parameters were removed"
    for o, n in zip(old, new):
        assert o == n, f"{where}: parameter changed from {o} to {n}"
    for extra in new[len(old):]:
        name, kind, default = extra
        assert default is not None or kind in ("VAR_POSITIONAL",
                                               "VAR_KEYWORD"), \
            f"{where}: new parameter {name!r} needs a default"


@pytest.mark.parametrize("module,name", CASES)
def test_public_name_still_compatible(module, name):
    if module not in LIVE:
        pytest.skip(f"{module} needs optional extras")
    old = SNAPSHOT[module][name]
    assert name in LIVE[module], f"{module}.{name} was removed"
    new = LIVE[module][name]
    assert new["kind"] == old["kind"]
    where = f"{module}.{name}"
    if old["kind"] == "function":
        _check_signature(old["signature"], new["signature"], where)
    elif old["kind"] == "class":
        for meth, sig in old["methods"].items():
            assert meth in new["methods"], f"{where}.{meth} was removed"
            _check_signature(sig, new["methods"][meth], f"{where}.{meth}")
        assert set(old["properties"]) <= set(new["properties"])
    elif old["kind"] == "constant":
        assert new["value"] == old["value"], f"{where} changed value"
