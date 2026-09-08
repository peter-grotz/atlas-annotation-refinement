"""Quantitative description of the disagreement between a label and a reference.

The refinement strategy for a structure follows from how the propagated label
fails, not from the identity of the structure. A label that over-covers a dim
margin and one that under-covers a dark interior call for opposite operations,
and both are distinguishable before any algorithm is chosen.

This module produces that description deterministically. It reports where the
disagreement sits in intensity, how deep it lies relative to the label surface,
whether it forms one coherent region or many, and the best recall any
subtractive method could reach.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..io.volumes import Volume, background_level


@dataclass(frozen=True)
class IntensityBand:
    """Intensity distribution of a voxel population, normalised to background."""

    count: int
    median: float
    p10: float
    p90: float
    median_over_background: float

    @classmethod
    def measure(cls, image: np.ndarray, mask: np.ndarray, background: float) -> "IntensityBand":
        if not mask.any():
            return cls(0, float("nan"), float("nan"), float("nan"), float("nan"))
        values = image[mask]
        p10, median, p90 = (float(v) for v in np.percentile(values, (10, 50, 90)))
        return cls(int(mask.sum()), median, p10, p90, median / background)


@dataclass(frozen=True)
class ErrorSignature:
    """How a propagated label differs from a reference annotation."""

    label_voxels: int
    reference_voxels: int
    dice: float
    recall: float
    precision: float
    volume_ratio: float

    matched: IntensityBand
    excess: IntensityBand
    missed: IntensityBand

    #: Separation between the excess and matched intensity distributions, as a
    #: fraction of their combined spread. Values near 1 indicate the two
    #: populations barely overlap and an intensity criterion can separate them.
    intensity_separation: float

    #: Median depth of excess voxels beneath the label surface, in voxels.
    excess_median_depth: float
    #: Fraction of excess voxels within three voxels of the label surface.
    excess_surface_fraction: float
    #: Fraction of excess voxels belonging to its largest connected component.
    excess_largest_component: float

    #: Highest recall a purely SUBTRACTIVE correction could reach. Missed voxels
    #: lie outside the propagated label, so no rule that only removes material
    #: can recover them. It bounds thresholding and erosion; it says nothing
    #: about corrections that move or grow the label, and is not a bound on the
    #: task.
    subtractive_recall_ceiling: float

    #: Which correction direction the disagreement calls for. See
    #: :func:`classify_error`.
    dominant_error: str

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"dice {self.dice:.4f}  recall {self.recall:.4f}  precision {self.precision:.4f}\n"
            f"dominant error: {self.dominant_error}\n"
            f"excess {self.excess.count:,} vox at {self.excess.median_over_background:.2f}x "
            f"background, depth {self.excess_median_depth:.1f}, "
            f"{self.excess_largest_component:.1%} in one component\n"
            f"matched at {self.matched.median_over_background:.2f}x background, "
            f"intensity separation {self.intensity_separation:.2f}\n"
            f"subtractive recall ceiling {self.subtractive_recall_ceiling:.4f}"
        )


#: Volume ratios within this band of 1.0 are treated as neither over- nor
#: under-covering; the label holds about the right amount of material.
_VOLUME_DEADBAND = 0.10

#: Error populations above this fraction of the reference are substantial enough
#: that a label with the right volume is displaced rather than well placed.
_DISPLACEMENT_ERROR_FRACTION = 0.15


def classify_error(volume_ratio: float, excess: int, missed: int, reference: int) -> str:
    """Name the correction direction the disagreement calls for.

    Comparing raw voxel counts is not enough. A label holding almost exactly the
    right amount of material can still disagree with the reference over most of
    its extent, and the sign of a 0.2% count difference is then arbitrary. Such a
    label is displaced, and neither adding nor removing material will fix it: the
    correction has to move it.

    Returns one of ``over-coverage``, ``under-coverage``, ``displacement``, or
    ``close-agreement``.
    """
    if reference <= 0:
        return "undefined"
    balanced = abs(volume_ratio - 1.0) <= _VOLUME_DEADBAND
    substantial = (
        excess / reference >= _DISPLACEMENT_ERROR_FRACTION
        and missed / reference >= _DISPLACEMENT_ERROR_FRACTION
    )
    if balanced:
        return "displacement" if substantial else "close-agreement"
    return "over-coverage" if volume_ratio > 1.0 else "under-coverage"


def characterize(image: Volume, label: Volume, reference: Volume) -> ErrorSignature:
    """Compare a propagated label against a manual reference on the same grid.

    Args:
        image: Specimen intensities.
        label: Propagated label to assess.
        reference: Manual annotation treated as correct.
    """
    image.assert_compatible(label)
    image.assert_compatible(reference)

    intensities = image.data
    background = background_level(image)
    predicted = label.data > 0
    truth = reference.data > 0

    matched = predicted & truth
    excess = predicted & ~truth
    missed = truth & ~predicted

    n_pred, n_true, n_match = int(predicted.sum()), int(truth.sum()), int(matched.sum())
    dice = 2 * n_match / (n_pred + n_true) if n_pred + n_true else 0.0
    recall = n_match / n_true if n_true else 0.0
    precision = n_match / n_pred if n_pred else 0.0

    band_matched = IntensityBand.measure(intensities, matched, background)
    band_excess = IntensityBand.measure(intensities, excess, background)
    band_missed = IntensityBand.measure(intensities, missed, background)

    separation = _distribution_separation(band_matched, band_excess)
    depth = ndi.distance_transform_edt(predicted) if predicted.any() else np.zeros_like(intensities)

    if excess.any():
        excess_depths = depth[excess]
        median_depth = float(np.median(excess_depths))
        surface_fraction = float((excess_depths <= 3).mean())
        components, _ = ndi.label(excess)
        sizes = np.bincount(components.ravel())[1:]
        largest = float(sizes.max() / sizes.sum()) if sizes.size else 0.0
    else:
        median_depth, surface_fraction, largest = 0.0, 0.0, 0.0

    return ErrorSignature(
        label_voxels=n_pred,
        reference_voxels=n_true,
        dice=round(dice, 6),
        recall=round(recall, 6),
        precision=round(precision, 6),
        volume_ratio=round(n_pred / n_true, 6) if n_true else float("nan"),
        matched=band_matched,
        excess=band_excess,
        missed=band_missed,
        intensity_separation=round(separation, 6),
        excess_median_depth=round(median_depth, 3),
        excess_surface_fraction=round(surface_fraction, 4),
        excess_largest_component=round(largest, 4),
        subtractive_recall_ceiling=round(recall, 6),
        dominant_error=classify_error(
            n_pred / n_true if n_true else float("nan"),
            int(excess.sum()),
            int(missed.sum()),
            n_true,
        ),
    )


def _distribution_separation(a: IntensityBand, b: IntensityBand) -> float:
    """How cleanly two intensity populations separate, on a bounded scale.

    The signed gap between the near edges of the two interquantile ranges,
    normalised by that gap plus their combined spread. The result lies in
    (-1, 1): approaching +1 when the ranges are disjoint and the populations can
    be told apart by intensity alone, and approaching -1 when they overlap so
    heavily that no threshold can separate them.

    Normalising by spread alone is undefined for a uniform population and
    unbounded for a nearly uniform one, which is the common case in synthetic
    data and in regions of saturated signal. Including the gap in the
    denominator keeps the measure finite and comparable across volumes.
    """
    if a.count == 0 or b.count == 0:
        return float("nan")
    spread = max((a.p90 - a.p10), 0.0) + max((b.p90 - b.p10), 0.0)
    gap = a.p10 - b.p90 if a.median > b.median else b.p10 - a.p90
    scale = spread + abs(gap)
    if scale <= 0:
        return 0.0  # identical degenerate distributions: no separation either way
    return float(gap / scale)
