"""Corrections that fit a separate threshold in each part of a structure.

A single threshold applied to a whole structure is a compromise between its
boundaries, and for a structure whose boundaries differ it is the wrong
compromise everywhere. Cortex is the clear case: its outer surface meets dark
background, where a threshold removes overshoot, and its inner surface meets
brighter white matter, where the same threshold removes nothing useful and a
higher one eats the structure. One number cannot serve both.

Measured on real corrections, the disagreement is also regionally concentrated
rather than spread along the boundary - the ten largest connected components of
the manual correction held over 90% of its voxels. That is the signature of a
correction that should vary by place.

The partition must be derivable from the image and the propagated label alone.
If regions were defined using the manual annotation, the region assignment would
carry information about the answer and every score computed against it would be
inflated. Everything here is therefore computed from the propagated label's own
geometry and the specimen's own intensities, exactly as it would be at inference
time on a specimen with no annotation.

Each region costs one free parameter, so the parameter budget applies with full
force: a two-region family needs two independent annotations, a four-region
family needs four. This is a real constraint on how fine a partition can be
justified, and it is enforced rather than advised.

Choosing the partition is the whole design decision, and it is per structure.
Measured on a cortical cohort, the shell/interior split gained exactly nothing:
the fit chose the same value for both regions when free to differ, and the
second parameter spanned its entire range for a 0.0005 change in agreement.
That is a result about cortex and that partition, not about region-wise fitting.
Cortex is in fact the worst case for it, because one of its two boundaries faces
brighter tissue that no threshold separates at any value, so the parameter
covering that side has nothing to find.

The lesson generalises even though the result does not. The shell/interior split
is an excellent *localiser* of error - most of the residual disagreement sits on
the side facing brighter tissue - yet error concentrating somewhere does not
mean a different threshold helps there. Partition by where the correction should
*differ*, not by where it is *largest*, and expect a partition that suits one
structure to be degenerate on another: measured on a ventricular system, where
every neighbour is brighter, shell/interior put 97-100% of the label in a single
region while a caliber split divided it roughly in half.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from ..io.volumes import Volume
from .base import REGISTRY, Parameter
from .intensity import _SpanNormalised

#: Fraction of the background-to-interior span used to decide whether material
#: outside the label counts as dark or bright when partitioning. Fixed rather
#: than fitted: it defines where the regions are, and letting the search move it
#: would make the regions themselves a function of the parameters.
PARTITION_REFERENCE = 0.5

#: Search space shared by every region. It starts at zero so the fit can
#: discover that a region should not be thresholded at all, which is the
#: expected answer wherever the neighbouring tissue is brighter than the
#: structure.
FRACTIONS = np.round(np.arange(0.0, 0.61, 0.05), 3).tolist()


def shell_interior_partition(
    image: Volume,
    label: Volume,
    interior: float,
    background: float,
) -> tuple[np.ndarray, list[str]]:
    """Split a label by which kind of neighbour each voxel is nearest.

    Every voxel outside the label is classed as dark or bright relative to the
    midpoint of the specimen's background-to-interior span. Each labelled voxel
    then belongs to whichever it is closer to: the ``shell`` faces darker
    material, the ``interior`` faces brighter material.

    This is a relative test, so it needs no thickness constant and does not
    assume a structure is any particular size. A structure with only one kind of
    neighbour yields one populated region, and the other is simply empty.

    Returns:
        An integer array - 0 outside the label, 1 and 2 for the regions - and
        the region names in index order.
    """
    mask = label.data > 0
    names = ["shell", "interior"]
    regions = np.zeros(mask.shape, dtype="uint8")
    if not mask.any() or not np.isfinite(interior):
        return regions, names

    reference = background + PARTITION_REFERENCE * (interior - background)
    exterior = ~mask
    darker = exterior & (image.data < reference)
    brighter = exterior & ~darker

    # distance_transform_edt measures distance to the nearest zero, so each
    # class is inverted before transforming.
    to_dark = (
        ndi.distance_transform_edt(~darker) if darker.any()
        else np.full(mask.shape, np.inf, dtype="float32")
    )
    to_bright = (
        ndi.distance_transform_edt(~brighter) if brighter.any()
        else np.full(mask.shape, np.inf, dtype="float32")
    )

    regions[mask & (to_dark <= to_bright)] = 1
    regions[mask & (to_dark > to_bright)] = 2
    return regions, names


#: Half-width, in voxels, below which a part of a structure is treated as thin.
#: Fixed rather than fitted, for the same reason as PARTITION_REFERENCE: it
#: defines where the regions are. Two voxels of half-width is where partial
#: volume starts to dominate a lumen at typical working resolutions.
THIN_HALF_WIDTH = 2.0


def caliber_partition(
    image: Volume,
    label: Volume,
    interior: float,
    background: float,
) -> tuple[np.ndarray, list[str]]:
    """Split a label by local caliber: how thick the structure is at each voxel.

    Intended for structures that vary in width along their length - a
    ventricular system runs from wide lateral chambers to a narrow aqueduct.
    A wide part has a clean core whose intensity reaches the structure's true
    level; a narrow part is partial-volume mixed with its surroundings
    throughout, so its intensity never gets there and a threshold fitted on the
    wide part removes it entirely.

    Caliber is taken as distance to the label surface, which is half-width
    rather than width. The ``image`` and level arguments are unused here and
    accepted so that partitions are interchangeable.

    Returns:
        An integer array - 0 outside the label, 1 thin, 2 thick - and the region
        names in index order.
    """
    del image, interior, background      # signature kept uniform across partitions
    mask = label.data > 0
    names = ["thin", "thick"]
    regions = np.zeros(mask.shape, dtype="uint8")
    if not mask.any():
        return regions, names
    depth = ndi.distance_transform_edt(mask)
    regions[mask & (depth <= THIN_HALF_WIDTH)] = 1
    regions[mask & (depth > THIN_HALF_WIDTH)] = 2
    return regions, names


@REGISTRY.register
class RegionalThreshold(_SpanNormalised):
    """Threshold each part of a structure at its own fraction of the span.

    The label is split by the kind of tissue each part borders, and a separate
    threshold is fitted for each part. Where the neighbour is darker a threshold
    removes overshoot; where it is brighter the fit is free to choose zero,
    leaving that part untouched, which is the correct answer rather than a
    failure to find one.

    Two free parameters, so two independent annotations are required before it
    may be fitted.
    """

    name = "regional_threshold"
    description = (
        "Separate thresholds for the parts of a label facing darker and "
        "brighter tissue."
    )

    #: Partition used to define the regions. Subclasses override it to split a
    #: structure a different way; the fitting machinery is unchanged.
    partition = staticmethod(shell_interior_partition)
    region_names: tuple[str, str] = ("shell", "interior")

    @property
    def parameters(self) -> Sequence[Parameter]:
        return (
            Parameter(
                "fraction_shell",
                FRACTIONS,
                "Threshold for the part of the label facing darker material.",
            ),
            Parameter(
                "fraction_interior",
                FRACTIONS,
                "Threshold for the part facing brighter material. Zero leaves "
                "it untouched, which is expected where intensity cannot "
                "separate the structure from its neighbour.",
            ),
        )

    def prepare(self, image: Volume, label: Volume) -> dict:
        context = super().prepare(image, label)
        regions, names = self.partition(
            image, label, context["interior"], context["background"]
        )
        context["regions"] = regions
        context["region_names"] = names
        return context

    def _regions(self, image: Volume, label: Volume) -> tuple[np.ndarray, list[str]]:
        cached = self.context
        if cached and "regions" in cached:
            return cached["regions"], cached["region_names"]
        background, interior = self._levels(image, label)
        return self.partition(image, label, interior, background)

    def apply(self, image: Volume, label: Volume, **params: float) -> Volume:
        image.assert_compatible(label)
        mask = label.data > 0
        background, interior = self._levels(image, label)
        if not np.isfinite(interior):
            return label.with_data(mask.astype("uint8"))

        regions, names = self._regions(image, label)
        out = np.zeros(mask.shape, dtype=bool)
        for index, region in enumerate(names, start=1):
            fraction = float(params[f"fraction_{region}"])
            selected = regions == index
            if not selected.any():
                continue
            if fraction <= 0.0:
                out |= selected
                continue
            threshold = background + fraction * (interior - background)
            out |= selected & (image.data > threshold)
        return label.with_data(out.astype("uint8"))

    def region_sizes(self, image: Volume, label: Volume) -> dict[str, int]:
        """Voxels in each region, for reporting which part a fit acted on."""
        regions, names = self._regions(image, label)
        return {
            name: int((regions == index).sum())
            for index, name in enumerate(names, start=1)
        }


@REGISTRY.register
class CaliberThreshold(RegionalThreshold):
    """Threshold the thin and thick parts of a structure separately.

    For a structure whose width varies along its length, one threshold is a
    compromise between parts with different intensity statistics. A wide part
    reaches the structure's true level in its core; a narrow part is
    partial-volume mixed throughout and never does, so a threshold fitted on the
    wide part deletes the narrow one.

    Two free parameters, so two independent annotations are required.
    """

    name = "caliber_threshold"
    description = "Separate thresholds for the thin and thick parts of a label."

    partition = staticmethod(caliber_partition)
    region_names = ("thin", "thick")

    @property
    def parameters(self) -> Sequence[Parameter]:
        return (
            Parameter(
                "fraction_thin",
                FRACTIONS,
                "Threshold for parts within THIN_HALF_WIDTH of the surface, "
                "where partial volume keeps intensity away from the "
                "structure's true level.",
            ),
            Parameter(
                "fraction_thick",
                FRACTIONS,
                "Threshold for parts with a genuine core.",
            ),
        )
