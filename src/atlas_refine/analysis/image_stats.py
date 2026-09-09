"""Per-specimen image description, computed without annotations.

Every measurement here runs on an acquisition alone, so a whole cohort can be
described before any of it is annotated. Two uses follow from that.

Selecting what to annotate. Annotator time is the binding constraint, so the
choice of which specimen to hand-correct next matters more than any parameter.
A specimen far from those already annotated, on a property the correction
depends on, is where a fitted rule is least likely to hold and where an
annotation is therefore most informative.

Explaining why a rule transfers unevenly. A correction fitted on one specimen
and applied to a cohort will behave differently across it, and the cause is
usually a measurable difference in acquisition rather than in anatomy. These
fields are the search space for that explanation.

Measurements are ratios or voxel counts rather than raw intensities, so
specimens of differing brightness remain comparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

from ..io.volumes import Volume, background_level

#: Tissue is taken as intensities above this multiple of the modal background.
TISSUE_THRESHOLD = 1.15

#: Stride for the measurements whose cost scales with volume. Fine structure is
#: not the object of interest here; the fields describe global properties.
STRIDE = 2


@dataclass(frozen=True)
class ImageProfile:
    """Acquisition properties of one specimen.

    Attributes:
        background: Modal intensity of the non-zero histogram.
        tissue_level: Median intensity of confident tissue.
        contrast: ``tissue_level / background``. The dimension a threshold
            expressed as a fraction of the background-to-tissue span already
            normalises for.
        stripe_anisotropy: Ratio of gradient energy between the worst and best
            axis. Illumination striping raises it, and it degrades any method
            driven by intensity gradients, since the contour locks onto stripe
            edges rather than tissue boundaries.
        shading: Coefficient of variation of the low-frequency field within
            tissue. Uneven illumination or clearing raises it. It defeats a
            single global threshold while leaving local gradients intact.
        background_spread: Variation of block-wise background estimates about
            their mean. Raised by regional clearing failure, where one global
            background value does not describe the whole volume.
        separability: Fraction of in-tissue variance explained by the best
            two-class split. An upper bound on what any threshold can achieve:
            a low value means the tissue classes are not separable by intensity
            at all, whatever the threshold.
        noise: High-frequency variation within tissue, relative to tissue level.
        tissue_fraction: Proportion of the field of view occupied by tissue.
        extent: Slices containing tissue along each axis. Differences here
            reflect dissection and framing rather than anatomy, and drive
            coverage mismatch against an atlas.
    """

    background: float
    tissue_level: float
    contrast: float
    stripe_anisotropy: float
    shading: float
    background_spread: float
    separability: float
    noise: float
    tissue_fraction: float
    extent: tuple[int, int, int]

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"contrast {self.contrast:.2f}x  separability {self.separability:.2f}  "
            f"shading {self.shading:.3f}  striping {self.stripe_anisotropy:.2f}  "
            f"noise {self.noise:.3f}\n"
            f"tissue occupies {self.tissue_fraction:.0%} of the field, "
            f"extent {self.extent}"
        )


def otsu_separability(values: np.ndarray, bins: int = 256) -> float:
    """Fraction of variance explained by the best two-class intensity split.

    One means perfectly separable, zero means unimodal. Reported because it
    bounds every threshold-based correction: where it is low, no choice of
    threshold distinguishes the classes and a different family is required.
    """
    if values.size == 0:
        return float("nan")
    counts, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    if total == 0:
        return float("nan")
    weight0 = np.cumsum(counts)[:-1].astype(float)
    weight1 = total - weight0
    sum_all = float((counts * centres).sum())
    sum0 = np.cumsum(counts * centres)[:-1]
    mean0 = sum0 / np.maximum(weight0, 1)
    mean1 = (sum_all - sum0) / np.maximum(weight1, 1)
    between = (weight0 / total) * (weight1 / total) * (mean0 - mean1) ** 2
    variance = float(values.var())
    return float(between.max() / variance) if variance > 0 else float("nan")


#: Gradient energy below this fraction of the strongest axis is treated as
#: absent rather than as a ratio, which would otherwise diverge.
_ANISOTROPY_FLOOR = 1e-3


def _anisotropy(gradients: np.ndarray) -> float:
    """Ratio of gradient energy between the strongest and weakest axis.

    One means isotropic. A volume with no texture on any axis has no measurable
    anisotropy and is reported as isotropic rather than as a ratio of zeros,
    and an axis with negligible energy is floored relative to the strongest so
    the ratio stays finite.
    """
    strongest = float(gradients.max())
    if strongest <= 0:
        return 1.0
    weakest = max(float(gradients.min()), strongest * _ANISOTROPY_FLOOR)
    return float(strongest / weakest)


def profile_image(volume: Volume, *, stride: int = STRIDE) -> ImageProfile:
    """Describe one specimen. Requires no annotation."""
    data = volume.data
    background = background_level(volume)
    tissue = data > background * TISSUE_THRESHOLD
    if not tissue.any():
        raise ValueError("no tissue found above the background threshold")
    tissue_level = float(np.median(data[tissue]))

    coarse, coarse_tissue = data[::stride, ::stride, ::stride], tissue[::stride, ::stride, ::stride]

    gradients = []
    for axis in range(3):
        difference = np.abs(np.diff(coarse, axis=axis))
        mask = coarse_tissue.take(range(difference.shape[axis]), axis=axis)
        gradients.append(float(np.median(difference[mask])) if mask.any() else 0.0)
    stripe = _anisotropy(np.asarray(gradients))

    low_frequency = ndi.uniform_filter(coarse, size=max(3, 51 // stride))
    interior = low_frequency[coarse_tissue]
    shading = float(interior.std() / max(interior.mean(), 1e-6)) if interior.size else float("nan")

    # Scale the block to the volume: a fixed size larger than the array yields
    # a single block and no spread to measure.
    block = max(4, min(96 // stride, min(coarse.shape[:2]) // 3))
    modes = []
    for i in range(0, max(coarse.shape[0] - block, 1), block):
        for j in range(0, max(coarse.shape[1] - block, 1), block):
            values = coarse[i : i + block, j : j + block, :]
            values = values[values > 0]
            if values.size > 200:
                counts, edges = np.histogram(values, bins=128)
                k = int(counts.argmax())
                modes.append(0.5 * (edges[k] + edges[k + 1]))
    spread = (
        float(np.std(modes) / max(np.mean(modes), 1e-6)) if len(modes) > 3 else float("nan")
    )

    separability = otsu_separability(data[tissue][::7])

    high_frequency = coarse - ndi.uniform_filter(coarse, size=3)
    noise = (
        float(high_frequency[coarse_tissue].std() / max(tissue_level, 1e-6))
        if coarse_tissue.any()
        else float("nan")
    )

    extent = tuple(
        int(tissue.any(axis=tuple(a for a in range(3) if a != axis)).sum()) for axis in range(3)
    )

    return ImageProfile(
        background=round(background, 2),
        tissue_level=round(tissue_level, 2),
        contrast=round(tissue_level / background, 4),
        stripe_anisotropy=round(stripe, 4),
        shading=round(shading, 4),
        background_spread=round(spread, 4),
        separability=round(separability, 4),
        noise=round(noise, 4),
        tissue_fraction=round(float(tissue.mean()), 4),
        extent=extent,  # type: ignore[arg-type]
    )
