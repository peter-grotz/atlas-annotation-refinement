"""Annotation-free screening of a produced label.

A correction is fitted on the specimens that carry a manual annotation and then
applied to every other specimen in the cohort, where nothing checks it. That
majority is the part of the output no one has looked at, and the failures that
occur there are not subtle: a label that leaked into a neighbouring structure,
lost a lobe, or fragmented into hundreds of pieces is obviously wrong to anyone
who sees it, and completely invisible to a pipeline that reports no number for
it.

The measurements here need no reference annotation. They cannot say a label is
correct - only agreement with a manual annotation can do that, and where such an
annotation exists it should be used instead. What they can do is say a label is
implausible, and rank a cohort so the specimens most worth a closer look come
first.

Two kinds of evidence are combined. Shape properties are judged against
absolutes that hold for any anatomical structure: a structure is a small number
of connected pieces, not hundreds, and a bilateral structure is roughly
symmetric. Everything else is judged relative to the cohort, because the
absolute volume of a structure is meaningful only against the other specimens
processed the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from ..analysis.boundary import profile_boundary
from ..io.volumes import Volume

#: A label holding a smaller share of its voxels in one piece than this is
#: reported as fragmented. Anatomical structures are not confetti; a low value
#: usually means a threshold admitted scattered noise.
COHESION_FLOOR = 0.90

#: Pieces smaller than this share of the largest are counted as debris rather
#: than as genuine separate lobes.
DEBRIS_FRACTION = 0.01

#: Substantial pieces beyond this count are reported. A structure may genuinely
#: be a few lobes; it is not dozens. Counted on pieces above DEBRIS_FRACTION,
#: because the raw count of single-voxel specks is large for any real label and
#: says nothing - every brain in a cohort will have hundreds of them.
MAX_PIECES = 5

#: Left-right volume difference, as a fraction of the total, beyond which a
#: bilateral structure is reported as lopsided.
ASYMMETRY_CEILING = 0.15

#: An enclosed region larger than this share of the label is taken to be
#: anatomy the structure wraps around rather than a hole in it. A sheet-like
#: structure encloses whatever it surrounds - a cortical label encloses most of
#: the brain - and counting that as a defect flags every such label.
ENCLOSED_ANATOMY_FRACTION = 0.10

#: Combined share of the label made up of small enclosed regions, beyond which
#: a threshold is taken to have punched through the structure's interior.
PUNCTURE_CEILING = 0.02

#: Cohort z-score beyond which a measurement is called out.
COHORT_SIGMA = 2.0


@dataclass(frozen=True)
class LabelScreen:
    """Plausibility measurements for one produced label, needing no reference.

    Attributes:
        specimen: Identifier this label belongs to.
        voxels: Total labelled voxels.
        volume_mm3: Physical volume, using the volume's own spacing and unit.
        components: Connected pieces, counting only those above
            :data:`DEBRIS_FRACTION` of the largest.
        debris: Pieces below that fraction. Reported, not judged: hundreds of
            single-voxel specks are normal for a label of any size, and the
            share of volume they hold is what matters, which ``cohesion``
            already reports.
        cohesion: Share of labelled voxels in the single largest piece.
        surface_ratio: Boundary voxels divided by total voxels. Rises when a
            label is riddled with holes or has a ragged surface, and is
            comparable across specimens only for the same structure.
        enclosed: Voxels the label surrounds but does not include. Large for
            any sheet-like structure, which surrounds whatever it wraps, so this
            is reported rather than judged.
        punctures: Voxels in small enclosed pockets only, excluding any region
            big enough to be anatomy the structure wraps. This is the figure
            that indicates a threshold punched holes through the interior.
        lateral_asymmetry: Absolute left-right volume difference as a fraction
            of the total, or None when no lateral axis was supplied. Only
            meaningful for a bilateral structure.
        boundary_sharpness: Steepness of the intensity profile at the label
            surface. A label placed on a real edge scores high; one sitting in
            a flat region does not. None when the profile could not be formed.
        concerns: Human-readable statements of what looks wrong, empty when
            nothing does.
    """

    specimen: str
    voxels: int
    volume_mm3: float
    components: int
    debris: int
    cohesion: float
    surface_ratio: float
    enclosed: int
    punctures: int
    lateral_asymmetry: float | None
    boundary_sharpness: float | None
    concerns: list[str] = field(default_factory=list)

    @property
    def looks_implausible(self) -> bool:
        """Whether any absolute check failed. Cohort-relative checks are added
        by :func:`rank_cohort` and are not counted here, because a single label
        has no cohort to be unusual within."""
        return bool(self.concerns)

    def summary(self) -> str:
        parts = [
            f"{self.specimen}: {self.voxels:,} vox ({self.volume_mm3:.2f} mm3), "
            f"{self.components} piece(s), cohesion {self.cohesion:.3f}"
        ]
        if self.lateral_asymmetry is not None:
            parts.append(f"lateral asymmetry {self.lateral_asymmetry:.1%}")
        if self.boundary_sharpness is not None:
            parts.append(f"boundary sharpness {self.boundary_sharpness:.4f}")
        if self.concerns:
            parts.extend(f"  concern: {c}" for c in self.concerns)
        return "\n".join(parts)


def _physical_volume(label: Volume) -> float:
    """Labelled volume in cubic millimetres."""
    per_voxel = float(np.prod(np.asarray(label.spacing, dtype=float)))
    if label.unit == "um":
        per_voxel /= 1e9
    elif label.unit == "mm":
        pass
    else:  # pragma: no cover - Volume restricts the unit vocabulary
        raise ValueError(f"unsupported unit {label.unit!r}")
    return float(np.count_nonzero(label.data) * per_voxel)


def screen_label(
    label: Volume,
    image: Volume | None = None,
    *,
    specimen: str = "",
    lateral_axis: int | None = None,
) -> LabelScreen:
    """Measure whether a produced label is anatomically plausible.

    Args:
        label: The produced label to screen.
        image: Specimen intensities. Supplying them adds the boundary-sharpness
            measurement, which is the strongest single indicator that a label
            was placed on a real edge rather than in a flat region.
        specimen: Identifier, carried through to the report.
        lateral_axis: Array axis running left to right, enabling the symmetry
            check. Left unset the check is skipped rather than guessed -
            inferring anatomical axes from unreliable headers is how a label
            silently ends up mirrored.

    Returns:
        A :class:`LabelScreen`. It can indicate that a label is implausible; it
        cannot establish that one is correct.
    """
    mask = label.data > 0
    total = int(mask.sum())
    if total == 0:
        return LabelScreen(
            specimen=specimen, voxels=0, volume_mm3=0.0, components=0, debris=0,
            cohesion=0.0, surface_ratio=float("nan"), enclosed=0, punctures=0,
            lateral_asymmetry=None, boundary_sharpness=None,
            concerns=["the label is empty"],
        )

    pieces, _ = ndi.label(mask)
    sizes = np.bincount(pieces.ravel())[1:]
    largest = int(sizes.max())
    substantial = int((sizes >= largest * DEBRIS_FRACTION).sum())
    debris = int(len(sizes) - substantial)
    cohesion = largest / total

    eroded = ndi.binary_erosion(mask, iterations=1)
    surface_ratio = float((mask & ~eroded).sum() / total)
    # Separate the region a sheet wraps from pockets punched through it: the
    # first is anatomy, the second is a defect, and they are indistinguishable
    # by total enclosed volume alone.
    filled = ndi.binary_fill_holes(mask)
    interior = filled & ~mask
    enclosed = int(interior.sum())
    punctures = 0
    if enclosed:
        pockets, _ = ndi.label(interior)
        pocket_sizes = np.bincount(pockets.ravel())[1:]
        small = pocket_sizes[pocket_sizes < total * ENCLOSED_ANATOMY_FRACTION]
        punctures = int(small.sum())

    asymmetry: float | None = None
    if lateral_axis is not None:
        counts = mask.sum(axis=tuple(a for a in range(3) if a != lateral_axis))
        centre = len(counts) // 2
        left, right = float(counts[:centre].sum()), float(counts[centre:].sum())
        asymmetry = abs(left - right) / max(left + right, 1.0)

    sharpness: float | None = None
    if image is not None:
        image.assert_compatible(label)
        try:
            sharpness = float(profile_boundary(image, label).sharpness_at_boundary)
        except ValueError:
            sharpness = None
        if sharpness is not None and not np.isfinite(sharpness):
            sharpness = None

    concerns: list[str] = []
    if cohesion < COHESION_FLOOR:
        concerns.append(
            f"only {cohesion:.1%} of the label is in one piece, so it is "
            f"fragmented rather than a single structure"
        )
    if substantial > MAX_PIECES:
        concerns.append(
            f"{substantial} separate pieces above {DEBRIS_FRACTION:.0%} of the "
            f"largest, which is more than a structure is made of"
        )
    if asymmetry is not None and asymmetry > ASYMMETRY_CEILING:
        concerns.append(
            f"left and right differ by {asymmetry:.1%}, which is lopsided for a "
            f"bilateral structure and suggests a one-sided failure"
        )
    if punctures > total * PUNCTURE_CEILING:
        concerns.append(
            f"{punctures:,} voxels sit in small pockets enclosed by the label, "
            f"so a threshold has punched holes through its own interior"
        )
    if sharpness is not None and sharpness <= 0:
        concerns.append(
            "intensity does not rise across the label surface, so the boundary "
            "is not sitting on an edge and is probably misplaced"
        )

    return LabelScreen(
        specimen=specimen,
        voxels=total,
        volume_mm3=round(_physical_volume(label), 5),
        components=substantial,
        debris=debris,
        cohesion=round(cohesion, 5),
        surface_ratio=round(surface_ratio, 5),
        enclosed=enclosed,
        punctures=punctures,
        lateral_asymmetry=None if asymmetry is None else round(asymmetry, 5),
        boundary_sharpness=None if sharpness is None else round(sharpness, 5),
        concerns=concerns,
    )


@dataclass(frozen=True)
class CohortReview:
    """One specimen's place in a screened cohort, most suspicious first."""

    screen: LabelScreen
    #: Combined suspicion, higher meaning more worth looking at. Comparable
    #: only within the cohort it was computed for.
    suspicion: float
    #: Statements of what is unusual relative to the other specimens, on top of
    #: the absolute concerns already in ``screen``.
    relative_concerns: list[str] = field(default_factory=list)

    @property
    def specimen(self) -> str:
        return self.screen.specimen

    def describe(self) -> str:
        lines = [f"{self.specimen}  suspicion {self.suspicion:.2f}"]
        lines.extend(f"    {c}" for c in self.screen.concerns)
        lines.extend(f"    {c}" for c in self.relative_concerns)
        if len(lines) == 1:
            lines.append("    nothing unusual")
        return "\n".join(lines)


def _z(values: Mapping[str, float]) -> dict[str, float]:
    keys = [k for k, v in values.items() if v is not None and np.isfinite(v)]
    if len(keys) < 3:
        return {k: 0.0 for k in values}
    arr = np.array([values[k] for k in keys], dtype=float)
    spread = float(arr.std())
    if spread == 0:
        return {k: 0.0 for k in values}
    centre = float(arr.mean())
    out = {k: 0.0 for k in values}
    out.update({k: (float(values[k]) - centre) / spread for k in keys})
    return out


def rank_cohort(screens: Sequence[LabelScreen]) -> list[CohortReview]:
    """Order screened labels by how much they warrant a closer look.

    Absolute concerns dominate, because a fragmented or lopsided label is wrong
    regardless of what the rest of the cohort looks like. Cohort-relative
    deviation is added on top, and is suppressed below three specimens, where
    a standard deviation over the sample says nothing.

    The ordering is a triage aid. It does not score correctness, and a specimen
    at the bottom of the list has not been verified - only found unremarkable.
    """
    if not screens:
        return []

    volumes = {s.specimen: float(s.volume_mm3) for s in screens}
    sharp = {
        s.specimen: (float(s.boundary_sharpness) if s.boundary_sharpness is not None else float("nan"))
        for s in screens
    }
    z_volume, z_sharp = _z(volumes), _z(sharp)

    reviews = []
    for s in screens:
        relative = []
        score = 2.0 * len(s.concerns)

        zv = z_volume.get(s.specimen, 0.0)
        if abs(zv) >= COHORT_SIGMA:
            direction = "larger" if zv > 0 else "smaller"
            relative.append(
                f"volume is {abs(zv):.1f} sd {direction} than the cohort "
                f"({s.volume_mm3:.2f} mm3)"
            )
            score += abs(zv)

        zs = z_sharp.get(s.specimen, 0.0)
        if zs <= -COHORT_SIGMA:
            relative.append(
                f"boundary is {abs(zs):.1f} sd less sharp than the cohort, so the "
                f"correction landed less cleanly here than elsewhere"
            )
            score += abs(zs)

        reviews.append(CohortReview(screen=s, suspicion=round(score, 4),
                                    relative_concerns=relative))

    # Absolute concerns dominate by rank, not by weight: no amount of
    # cohort-relative deviation should push a label that is wrong on its own
    # terms below one that is merely unusual.
    return sorted(
        reviews,
        key=lambda r: (-len(r.screen.concerns), -r.suspicion, r.specimen),
    )
