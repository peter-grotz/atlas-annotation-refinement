"""Refinement algorithms and the registry that exposes them."""

from .base import REGISTRY, AlgorithmRegistry, Parameter, RefinementAlgorithm
from . import intensity  # noqa: F401  (registers the built-in algorithms)

__all__ = ["REGISTRY", "AlgorithmRegistry", "Parameter", "RefinementAlgorithm"]
