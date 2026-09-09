"""Intensity profile across a label boundary.

A correction places a boundary, so the question that decides whether intensity
can place it at all is what the intensity does as the boundary is crossed. The
profile answers it directly: mean intensity binned by signed distance to the
label surface, negative inside.

Three properties follow, and each rules something in or out.

How sharp the transition is. A steep profile means a small error in the
threshold produces a small positional error, so an intensity criterion can place
the boundary precisely. A shallow one means the opposite, and the same threshold
lands at different depths in different specimens.

Where the true boundary sits on the ramp. Expressed in normalised intensity,
this is directly comparable to a threshold expressed the same way, so a fitted
parameter can be read against the anatomy it is meant to follow.

Whether the two sides differ at all. A profile that is flat across the boundary
says no threshold can place it, however the parameter is chosen, and the
correction must come from elsewhere.

Normalisation uses the profile's own plateaus rather than an external level, so
it is stable for structures darker than their surroundings as well as brighter,
and for structures too thin to have an interior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

from ..io.volumes import Volume

#: Range either side of the boundary, in voxels.
EXTENT = 16

#: Bins with fewer voxels than this are not reported; a mean over a handful of
#: voxels is noise rather than a measurement.
MIN_VOXELS_PER_BIN = 200

#: Bins at least this far from the boundary define the plateau on each side.
PLATEAU_DISTANCE = 8


@dataclass(frozen=True)
class BoundaryProfile:
    """Intensity as a function of signed distance to a label surface."""

    #: Bin centres, in voxels. Negative is inside the label.
    distances: list[int]
    #: Mean normalised intensity per bin: 0 on the inside plateau, 1 outside.
    #: None where a bin held too few voxels to report.
    intensity: list[float | None]
    #: Voxels contributing to each bin.
    counts: list[int]

    #: Normalised intensity interpolated at the boundary itself.
    intensity_at_boundary: float
    #: Rate of change of normalised intensity per voxel at the boundary.
    #: Normalisation maps the inside plateau to 0 and the outside plateau to 1,
    #: so for a boundary lying on a monotonic ramp between them this is a
    #: positive sharpness magnitude, and larger means an edge that can be placed
    #: more precisely. It is negative when the profile is not monotonic between
    #: the plateaus - a label extending past the structure into darker material,
    #: with brighter tissue beyond, puts the surface in a trough and the local
    #: gradient runs opposite to the plateau-to-plateau trend. A negative value
    #: is therefore a positive finding: the surface does not lie on the ramp at
    #: all, and the label is misplaced rather than merely imprecise. The
    #: direction of the overall step is carried by ``contrast_across_boundary``.
    sharpness_at_boundary: float
    #: Distance over which the profile crosses from 0.2 to 0.8, in voxels.
    #: None when the two sides are too similar for the crossing to be defined.
    transition_width: float | None
    #: Intensity difference between the plateaus, before normalisation. Near
    #: zero means the sides are indistinguishable and no threshold can separate
    #: them.
    contrast_across_boundary: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_placeable_by_intensity(self) -> bool:
        """Whether intensity distinguishes the two sides enough to place a
        boundary at all."""
        return bool(
            np.isfinite(self.sharpness_at_boundary)
            and abs(self.sharpness_at_boundary) > 0.05
            and abs(self.contrast_across_boundary) > 0
        )

    def summary(self) -> str:
        width = f"{self.transition_width:.1f} vox" if self.transition_width else "undefined"
        return (
            f"boundary at normalised intensity {self.intensity_at_boundary:.3f}\n"
            f"sharpness {self.sharpness_at_boundary:.4f} per voxel, transition width {width}\n"
            f"step across the boundary {self.contrast_across_boundary:+.0f}\n"
            f"placeable by intensity: {self.is_placeable_by_intensity}"
        )


def _renormalise(distances: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Rescale so the inside plateau is 0 and the outside plateau is 1.

    Anchored on the profile's own extremes rather than a fixed depth, because
    a structure may be too thin to have voxels at any particular depth, and on
    its own plateaus rather than an external level, so the result is
    polarity-agnostic.
    """
    valid = ~np.isnan(values)
    inside = valid & (distances < 0)
    outside = valid & (distances > 0)
    if inside.sum() < 2 or outside.sum() < 2:
        return np.full_like(values, np.nan)
    deep = values[inside][np.argsort(distances[inside])[:3]]
    far = values[outside][np.argsort(-distances[outside])[:3]]
    low, high = float(np.mean(deep)), float(np.mean(far))
    if abs(high - low) <= 1e-9:
        return np.full_like(values, np.nan)
    return (values - low) / (high - low)


def profile_boundary(
    image: Volume,
    label: Volume,
    *,
    extent: int = EXTENT,
    min_voxels: int = MIN_VOXELS_PER_BIN,
) -> BoundaryProfile:
    """Measure intensity as a function of signed distance to a label's surface.

    Args:
        image: Specimen intensities.
        label: Label whose surface defines the boundary. Either a propagated
            label or a manual annotation; comparing the two profiles shows how
            far the propagated boundary sits from the annotated one.
    """
    image.assert_compatible(label)
    mask = label.data > 0
    if not mask.any():
        raise ValueError("label is empty")

    signed = ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)
    within = np.abs(signed) <= extent
    bins = np.round(signed[within]).astype(int)
    values = image.data[within]

    centres = np.arange(-extent, extent + 1)
    means = np.full(centres.shape, np.nan)
    counts = np.zeros(centres.shape, dtype=int)
    for i, centre in enumerate(centres):
        selected = bins == centre
        counts[i] = int(selected.sum())
        if counts[i] >= min_voxels:
            means[i] = float(values[selected].mean())

    raw_contrast = _plateau_difference(centres, means)
    normalised = _renormalise(centres, means)

    valid = ~np.isnan(normalised)
    if valid.sum() < 3:
        at_boundary, sharpness, width = float("nan"), float("nan"), None
    else:
        c, y = centres[valid], normalised[valid]
        at_boundary = float(np.interp(0, c, y))
        sharpness = float(np.gradient(y, c)[int(np.argmin(np.abs(c)))])
        width = _crossing_width(c, y)

    return BoundaryProfile(
        distances=centres.tolist(),
        intensity=[None if np.isnan(v) else round(float(v), 5) for v in normalised],
        counts=counts.tolist(),
        intensity_at_boundary=round(at_boundary, 4) if np.isfinite(at_boundary) else float("nan"),
        sharpness_at_boundary=round(sharpness, 4) if np.isfinite(sharpness) else float("nan"),
        transition_width=round(width, 3) if width is not None else None,
        contrast_across_boundary=round(raw_contrast, 3),
    )


def _plateau_difference(distances: np.ndarray, values: np.ndarray) -> float:
    valid = ~np.isnan(values)
    inside = valid & (distances <= -PLATEAU_DISTANCE)
    outside = valid & (distances >= PLATEAU_DISTANCE)
    if not inside.any() or not outside.any():
        inside = valid & (distances < 0)
        outside = valid & (distances > 0)
    if not inside.any() or not outside.any():
        return float("nan")
    return float(values[outside].mean() - values[inside].mean())


def _crossing_width(distances: np.ndarray, values: np.ndarray) -> float | None:
    """Distance between the 0.2 and 0.8 crossings of a normalised profile."""
    def crossing(level: float) -> float | None:
        signs = np.sign(values - level)
        change = np.where(np.diff(signs) != 0)[0]
        if change.size == 0:
            return None
        i = int(change[0])
        span = values[i + 1] - values[i]
        if abs(span) < 1e-9:
            return float(distances[i])
        return float(distances[i] + (level - values[i]) * (distances[i + 1] - distances[i]) / span)

    low, high = crossing(0.2), crossing(0.8)
    if low is None or high is None:
        return None
    return abs(high - low)
