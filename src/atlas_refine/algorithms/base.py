"""Refinement algorithm interface and registry.

An algorithm transforms a propagated label into a corrected one using the
specimen image. Algorithms declare their free parameters and the search space
for each, so parameter fitting and the constraints on it are handled uniformly
rather than per algorithm.

Parameters are expressed in units derived from each specimen's own intensity
statistics rather than as absolute values. A threshold stated as an absolute
intensity does not transfer between acquisitions; the same threshold stated as a
position between that specimen's background and its structure level does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ..io.volumes import Volume


@dataclass(frozen=True)
class Parameter:
    """A free parameter and the values to search over."""

    name: str
    values: Sequence[float]
    description: str = ""


class RefinementAlgorithm(ABC):
    """Base class for label refinement.

    Subclasses implement :meth:`apply` and declare :attr:`parameters`. The
    number of free parameters is compared against the number of independent
    annotations available before fitting, since a family with more parameters
    than examples cannot be validated.
    """

    name: str = "unnamed"
    description: str = ""

    @property
    @abstractmethod
    def parameters(self) -> Sequence[Parameter]:
        """Free parameters, in a fixed order."""

    @abstractmethod
    def apply(self, image: Volume, label: Volume, **params: float) -> Volume:
        """Return a refined label on the same grid as ``label``."""

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    def grid(self) -> Iterator[dict[str, float]]:
        """Enumerate the Cartesian product of the declared search spaces."""
        import itertools

        names = [p.name for p in self.parameters]
        for combination in itertools.product(*(p.values for p in self.parameters)):
            yield dict(zip(names, combination))

    def defaults(self) -> dict[str, float]:
        """Midpoint of each search space, used when no fit has been performed."""
        return {p.name: float(np.median(list(p.values))) for p in self.parameters}


class AlgorithmRegistry:
    """Name-to-class lookup for available algorithms."""

    def __init__(self) -> None:
        self._entries: dict[str, type[RefinementAlgorithm]] = {}

    def register(self, cls: type[RefinementAlgorithm]) -> type[RefinementAlgorithm]:
        if cls.name in self._entries:
            raise ValueError(f"algorithm '{cls.name}' is already registered")
        self._entries[cls.name] = cls
        return cls

    def create(self, name: str, **kwargs: Any) -> RefinementAlgorithm:
        if name not in self._entries:
            raise KeyError(f"unknown algorithm '{name}'. Available: {', '.join(self.names())}")
        return self._entries[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def describe(self) -> Mapping[str, str]:
        return {name: cls.description for name, cls in sorted(self._entries.items())}


REGISTRY = AlgorithmRegistry()
