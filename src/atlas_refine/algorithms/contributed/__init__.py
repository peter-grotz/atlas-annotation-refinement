"""Correction families authored during the loop.

Modules here are admitted by :func:`atlas_refine.algorithms.contribute.contribute`,
which installs them only after they conform to the interface, behave correctly
under adversarial inputs, and outperform every admissible incumbent on the
available annotations. Each has a rationale record under ``docs/algorithms/``
and a generated regression test under ``tests/contributed/``.

Every module in this package is imported at load time so that its registry
entry is available without an explicit import.
"""

from __future__ import annotations

import importlib
import pkgutil

__all__: list[str] = []

for _module in pkgutil.iter_modules(__path__):
    if not _module.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module.name}")
        __all__.append(_module.name)
