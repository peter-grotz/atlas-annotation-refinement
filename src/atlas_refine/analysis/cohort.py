"""Comparison across a cohort, and the choice of what to annotate next.

Two questions are answered here, and the second is the one that determines how
much the whole workflow costs.

Where does a specimen sit relative to the rest? A correction fitted on one
specimen is applied to the others, and it will hold least well on whichever is
most unlike them. Identifying those before applying a correction converts a
silent failure into a flagged one.

Which specimen should be annotated next? Annotator time is the binding
constraint. The most informative annotation is the one least predictable from
those already held, so the specimen furthest from the annotated set is the one
to take. Choosing at random, or by convenience, wastes the scarcest resource.

Also provided is a check on whether a fitted parameter tracks any measured
property. Correlations at this sample size are reported with the number of
observations attached, because at three or four specimens almost any feature
will order them plausibly by chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .image_stats import ImageProfile

#: Fields compared when placing specimens relative to one another. Scale-free,
#: so specimens of differing brightness remain comparable.
COMPARISON_FIELDS = (
    "contrast",
    "stripe_anisotropy",
    "shading",
    "background_spread",
    "separability",
    "noise",
    "tissue_fraction",
)

#: Standard deviations beyond which a specimen is reported as unlike the cohort.
OUTLIER_SIGMA = 2.0

#: Below this many observations a correlation is not reported as evidence.
MIN_OBSERVATIONS_FOR_CORRELATION = 5


@dataclass(frozen=True)
class CohortPosition:
    """Where one specimen sits relative to the rest of the cohort."""

    specimen: str
    #: Largest absolute z-score across the compared fields.
    deviation: float
    #: Field responsible for that deviation.
    driven_by: str
    #: Signed z-score of that field.
    z: float
    #: Distance to the nearest annotated specimen, in the same standardised
    #: space. Infinite when nothing is annotated yet.
    distance_to_annotated: float
    is_outlier: bool

    def describe(self) -> str:
        direction = "above" if self.z > 0 else "below"
        return (
            f"{self.specimen}: {self.driven_by} {abs(self.z):.1f} sd {direction} the cohort"
            + (f", {self.distance_to_annotated:.1f} from the nearest annotated specimen"
               if np.isfinite(self.distance_to_annotated) else "")
        )


def _matrix(profiles: Mapping[str, ImageProfile]) -> tuple[list[str], np.ndarray]:
    specimens = sorted(profiles)
    rows = []
    for specimen in specimens:
        values = profiles[specimen].to_dict()
        rows.append([float(values[field]) for field in COMPARISON_FIELDS])
    return specimens, np.asarray(rows, dtype=float)


def standardise(profiles: Mapping[str, ImageProfile]) -> tuple[list[str], np.ndarray]:
    """Z-score each field across the cohort.

    Fields with no variation contribute nothing and are zeroed rather than
    producing infinities.
    """
    specimens, values = _matrix(profiles)
    # A field may be undefined for some specimens - separability is undefined
    # for a uniform region, and a block-wise spread needs enough blocks. Such a
    # field carries no information about those specimens rather than
    # invalidating the comparison, so it is centred on the specimens that do
    # have it and contributes zero for the rest.
    centre = np.nanmean(values, axis=0)
    spread = np.nanstd(values, axis=0)
    spread[(spread == 0) | ~np.isfinite(spread)] = 1.0
    centre[~np.isfinite(centre)] = 0.0
    z = (values - centre) / spread
    return specimens, np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def position_in_cohort(
    profiles: Mapping[str, ImageProfile],
    annotated: Sequence[str] = (),
    *,
    sigma: float = OUTLIER_SIGMA,
) -> list[CohortPosition]:
    """Place every specimen relative to the cohort, most unusual first."""
    if len(profiles) < 2:
        raise ValueError("comparison needs at least two specimens")
    specimens, z = standardise(profiles)
    anchors = [specimens.index(s) for s in annotated if s in specimens]

    positions = []
    for i, specimen in enumerate(specimens):
        worst = int(np.argmax(np.abs(z[i])))
        distance = (
            min(float(np.linalg.norm(z[i] - z[a])) for a in anchors)
            if anchors
            else float("inf")
        )
        positions.append(
            CohortPosition(
                specimen=specimen,
                deviation=round(float(np.abs(z[i]).max()), 3),
                driven_by=COMPARISON_FIELDS[worst],
                z=round(float(z[i][worst]), 3),
                distance_to_annotated=round(distance, 3) if np.isfinite(distance) else distance,
                is_outlier=bool(np.abs(z[i]).max() >= sigma),
            )
        )
    return sorted(positions, key=lambda p: -p.deviation)


def next_to_annotate(
    profiles: Mapping[str, ImageProfile],
    annotated: Sequence[str],
) -> CohortPosition | None:
    """The un-annotated specimen least like those already annotated.

    With nothing annotated yet, returns the specimen furthest from the cohort
    centre: a first annotation on an atypical specimen constrains more than one
    on a representative specimen, because it reveals the range the correction
    must span.
    """
    positions = {p.specimen: p for p in position_in_cohort(profiles, annotated)}
    candidates = [p for s, p in positions.items() if s not in annotated]
    if not candidates:
        return None
    if annotated:
        return max(candidates, key=lambda p: p.distance_to_annotated)
    return max(candidates, key=lambda p: p.deviation)


@dataclass(frozen=True)
class FeatureAssociation:
    """Whether a fitted parameter tracks a measured image property."""

    field: str
    correlation: float
    n: int
    #: True only when there are enough observations for the correlation to mean
    #: anything. Below the threshold, ordering a handful of points is expected
    #: by chance and the value is reported for inspection only.
    interpretable: bool

    def describe(self) -> str:
        if not self.interpretable:
            return (
                f"{self.field}: r={self.correlation:+.2f} on n={self.n} — not "
                f"interpretable; at this sample size most fields will order the "
                f"points by chance. A candidate to test on the next annotation."
            )
        return f"{self.field}: r={self.correlation:+.2f} on n={self.n}"


def associate_with_parameter(
    profiles: Mapping[str, ImageProfile],
    parameter: Mapping[str, float],
    *,
    minimum_n: int = MIN_OBSERVATIONS_FOR_CORRELATION,
) -> list[FeatureAssociation]:
    """Correlate a per-specimen fitted parameter against each profile field.

    Args:
        profiles: Image profiles, keyed by specimen.
        parameter: The fitted value per specimen. Only specimens present in
            both are used.

    Returns:
        One association per field, strongest absolute correlation first, each
        carrying whether the sample size supports interpreting it.
    """
    shared = sorted(set(profiles) & set(parameter))
    if len(shared) < 2:
        raise ValueError("need at least two specimens with both a profile and a value")
    target = np.array([float(parameter[s]) for s in shared])

    associations = []
    for field in COMPARISON_FIELDS:
        values = np.array([float(profiles[s].to_dict()[field]) for s in shared])
        if values.std() == 0 or target.std() == 0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(values, target)[0, 1])
        associations.append(
            FeatureAssociation(
                field=field,
                correlation=round(correlation, 4),
                n=len(shared),
                interpretable=len(shared) >= minimum_n,
            )
        )
    return sorted(
        associations,
        key=lambda a: -abs(a.correlation) if np.isfinite(a.correlation) else 0.0,
    )
