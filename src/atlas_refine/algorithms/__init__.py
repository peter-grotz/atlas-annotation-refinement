"""Correction families and the registry that exposes them.

Built-in families are registered on import. Families authored during the loop
live in :mod:`atlas_refine.algorithms.contributed` and are discovered
automatically; see :mod:`atlas_refine.algorithms.contribute` for how one is
admitted.
"""

from .base import REGISTRY, AlgorithmRegistry, Parameter, RefinementAlgorithm
from .contribute import (
    BenchmarkReport,
    Contribution,
    ContributionRejected,
    Rationale,
    ValidationReport,
    benchmark,
    contribute,
    scaffold,
    validate,
)
from . import intensity  # noqa: F401  registers the built-in families
from . import regional  # noqa: F401  registers the region-wise families
from . import contributed  # noqa: F401  discovers authored families

__all__ = [
    "REGISTRY",
    "AlgorithmRegistry",
    "Parameter",
    "RefinementAlgorithm",
    "BenchmarkReport",
    "Contribution",
    "ContributionRejected",
    "Rationale",
    "ValidationReport",
    "benchmark",
    "contribute",
    "scaffold",
    "validate",
]
