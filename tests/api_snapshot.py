"""Collect the public API of the ``etas`` package as plain data.

The snapshot records, for every public module-level function and class
(and public methods of classes) defined in the package, the ordered
parameter names, kinds and defaults. ``tests/test_public_api.py`` compares
the live package against the stored snapshot so that refactoring cannot
silently break code that depends on ``etas``.
"""
import datetime
import importlib
import inspect

import numpy as np

MODULES = [
    "etas",
    "etas.download",
    "etas.evaluation",
    "etas.inversion",
    "etas.mc_b_est",
    "etas.plots",
    "etas.simulation",
    "etas.oef",
    "etas.oef.entrypoint",
]

# modules that need optional extras; skipped when those are missing
OPTIONAL = {"etas.oef", "etas.oef.entrypoint"}

# names re-exported from another module that downstream code imports
REEXPORTS = {
    "etas.inversion": ["round_half_up"],
    "etas.oef": ["entrypoint_suiETAS", "entrypoint_genETAS",
                 "entrypoint_europe"],
}


def _default(value):
    """repr of a default; time-dependent defaults are normalized."""
    if value is inspect.Parameter.empty:
        return None
    if isinstance(value, datetime.datetime):
        return "<datetime>"
    if isinstance(value, np.generic):
        # numpy scalars: repr differs between numpy 1 and 2
        value = value.item()
    return repr(value)


def _sig(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [
        [p.name, p.kind.name,
         _default(p.default)]
        for p in sig.parameters.values()
    ]


def collect():
    api = {}
    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            # optional extras (e.g. hermes for etas.oef) not installed
            if modname in OPTIONAL:
                continue
            raise
        entries = {}
        for name, obj in vars(mod).items():
            if name.startswith("_"):
                continue
            own = getattr(obj, "__module__", None) == modname
            if not (own or name in REEXPORTS.get(modname, [])):
                continue
            if inspect.isclass(obj):
                methods = {
                    m: _sig(f) for m, f in vars(obj).items()
                    if (not m.startswith("_") or m == "__init__")
                    and (inspect.isfunction(f)
                         or isinstance(f, (classmethod, staticmethod)))
                }
                methods = {
                    m: _sig(getattr(obj, m)) for m in methods}
                props = sorted(m for m, f in vars(obj).items()
                               if isinstance(f, property)
                               and not m.startswith("_"))
                entries[name] = {"kind": "class", "methods": methods,
                                 "properties": props}
            elif inspect.isfunction(obj):
                entries[name] = {"kind": "function", "signature": _sig(obj)}
        # public module-level constants (e.g. RANGES)
        for name, obj in vars(mod).items():
            if name.isupper() and not name.startswith("_") and \
                    isinstance(obj, (int, float, tuple, str)):
                entries[name] = {"kind": "constant", "value": repr(obj)}
        api[modname] = entries
    return api
