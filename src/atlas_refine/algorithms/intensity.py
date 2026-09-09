"""Intensity-based refinement algorithms.

Thresholds are placed relative to two levels measured in each specimen: the
modal background, and the structure's own interior intensity sampled where the
boundary transition cannot reach. Normalising to both ends rather than to
background alone accounts for variation in contrast between acquisitions, not
only variation in offset.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from ..io.volumes import Volume, background_level
from .base import REGISTRY, Parameter, RefinementAlgorithm


class _SpanNormalised(RefinementAlgorithm):
    """Shared base for families whose thresholds are placed on the span between
    the specimen's background and the label's interior level."""

    def __init__(self, min_depth: int = 8) -> None:
        self.min_depth = min_depth

    def prepare(self, image: Volume, label: Volume) -> dict:
        mask = label.data > 0
        return {
            "background": background_level(image),
            "interior": interior_level(image, mask, self.min_depth),
        }

    def _levels(self, image: Volume, label: Volume) -> tuple[float, float]:
        cached = self.context
        if cached:
            return cached["background"], cached["interior"]
        return (
            background_level(image),
            interior_level(image, label.data > 0, self.min_depth),
        )


def interior_level(image: Volume, mask: np.ndarray, min_depth: int = 8) -> float:
    """Median intensity of a label's core.

    Sampled at least ``min_depth`` voxels beneath the surface so that boundary
    transition voxels do not bias the estimate. Falls back to the whole label
    when it is too thin to have a core at that depth.
    """
    if not mask.any():
        return float("nan")
    core = mask & (ndi.distance_transform_edt(mask) >= min_depth)
    if int(core.sum()) < 1000:
        core = mask
    return float(np.median(image.data[core]))


@REGISTRY.register
class ContrastNormalisedThreshold(_SpanNormalised):
    """Retain label voxels above a threshold set by the specimen's own contrast.

    The threshold is placed a fixed fraction of the way from the modal
    background to the label's interior level::

        threshold = background + fraction * (interior - background)

    Suited to structures whose propagated label over-covers a lower-intensity
    margin, where the excess separates from the structure in intensity.
    """

    name = "contrast_threshold"
    description = "Threshold at a fraction of the background-to-interior span."

    @property
    def parameters(self) -> Sequence[Parameter]:
        return (
            Parameter(
                "fraction",
                np.round(np.arange(0.20, 0.66, 0.03), 3).tolist(),
                "Position of the threshold between background and interior.",
            ),
        )

    def apply(self, image: Volume, label: Volume, **params: float) -> Volume:
        image.assert_compatible(label)
        mask = label.data > 0
        background, interior = self._levels(image, label)
        if not np.isfinite(interior):
            return label.with_data(mask.astype("uint8"))
        threshold = background + float(params["fraction"]) * (interior - background)
        return label.with_data((mask & (image.data > threshold)).astype("uint8"))


@REGISTRY.register
class BandThreshold(_SpanNormalised):
    """Retain label voxels whose intensity falls inside a band.

    Both bounds are expressed relative to the background-to-interior span.
    Applicable where a structure is bounded on both sides in intensity, for
    example a region flanked by both lower- and higher-intensity neighbours.
    """

    name = "band_threshold"
    description = "Retain voxels between a lower and upper normalised bound."

    @property
    def parameters(self) -> Sequence[Parameter]:
        return (
            Parameter("lower", np.round(np.arange(0.20, 0.56, 0.05), 3).tolist(),
                      "Lower bound as a fraction of the span."),
            Parameter("upper", np.round(np.arange(1.20, 2.61, 0.20), 3).tolist(),
                      "Upper bound as a fraction of the span."),
        )

    def apply(self, image: Volume, label: Volume, **params: float) -> Volume:
        image.assert_compatible(label)
        mask = label.data > 0
        background, interior = self._levels(image, label)
        if not np.isfinite(interior):
            return label.with_data(mask.astype("uint8"))
        span = interior - background
        low = background + float(params["lower"]) * span
        high = background + float(params["upper"]) * span
        keep = mask & (image.data > low) & (image.data < high)
        return label.with_data(keep.astype("uint8"))


@REGISTRY.register
class HysteresisThreshold(_SpanNormalised):
    """Retain regions above a low bound that connect to a high-bound seed.

    Recovers parts of a structure whose intensity falls below a single threshold
    but which remain contiguous with clearly-included material. Ineffective when
    the excess is itself contiguous with the structure.
    """

    name = "hysteresis_threshold"
    description = "Seed at a high bound, grow to a low bound within the label."

    @property
    def parameters(self) -> Sequence[Parameter]:
        return (
            Parameter("low", np.round(np.arange(0.20, 0.51, 0.05), 3).tolist(),
                      "Growth bound as a fraction of the span."),
            Parameter("high", np.round(np.arange(0.55, 0.96, 0.05), 3).tolist(),
                      "Seed bound as a fraction of the span."),
        )

    def apply(self, image: Volume, label: Volume, **params: float) -> Volume:
        image.assert_compatible(label)
        mask = label.data > 0
        background, interior = self._levels(image, label)
        if not np.isfinite(interior):
            return label.with_data(mask.astype("uint8"))
        span = interior - background
        low = mask & (image.data > background + float(params["low"]) * span)
        seed = mask & (image.data > background + float(params["high"]) * span)
        if not seed.any():
            return label.with_data(low.astype("uint8"))
        components, _ = ndi.label(low)
        keep = np.isin(components, np.setdiff1d(np.unique(components[seed]), 0))
        return label.with_data(keep.astype("uint8"))


@REGISTRY.register
class MorphologicalCleanup(RefinementAlgorithm):
    """Erode the label and discard components below a size threshold.

    A blunt operation that trades recall for precision uniformly. Useful when
    the disagreement is a thin shell of roughly constant thickness and no
    intensity criterion separates it.
    """

    name = "morphological_cleanup"
    description = "Erode by a fixed radius, then remove small components."

    @property
    def parameters(self) -> Sequence[Parameter]:
        return (
            Parameter("erosion", [0, 1, 2, 3], "Erosion iterations."),
            Parameter("min_component", [0, 100, 500, 2000], "Smallest component to keep."),
        )

    def apply(self, image: Volume, label: Volume, **params: float) -> Volume:
        mask = label.data > 0
        iterations = int(params["erosion"])
        if iterations:
            mask = ndi.binary_erosion(mask, iterations=iterations)
        minimum = int(params["min_component"])
        if minimum and mask.any():
            components, _ = ndi.label(mask)
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            mask = np.isin(components, np.flatnonzero(sizes >= minimum))
        return label.with_data(mask.astype("uint8"))
