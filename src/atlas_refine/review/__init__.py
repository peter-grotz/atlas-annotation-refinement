"""Review of produced labels where no manual annotation exists.

A correction is fitted on the few specimens that carry a manual annotation and
applied to the rest, which is the majority and is unverified. This package
covers that gap in the only two ways available without more annotation:
measuring whether a label is anatomically plausible, and rendering it so it can
be looked at.

Neither establishes correctness. Where a manual annotation exists, agreement
with it is exact and settles the question; these tools are for where it does
not. They are a triage aid, and their output ranks what deserves attention
rather than scoring what is right.

The division of labour is deliberate: this package measures and renders, and
stops there. Interpreting a render - deciding whether an outline follows the
structure or has strayed into a neighbour - is left to the reviewer, whether a
person or a model. Keeping that judgement outside the package is what allows
the package itself to remain deterministic, dependency-light, and reproducible.
"""

from .panel import OUTLINE_COLOURS, Panel, render_panel
from .screen import (
    ASYMMETRY_CEILING,
    COHESION_FLOOR,
    CohortReview,
    LabelScreen,
    rank_cohort,
    screen_label,
)

__all__ = [
    "ASYMMETRY_CEILING",
    "COHESION_FLOOR",
    "CohortReview",
    "LabelScreen",
    "OUTLINE_COLOURS",
    "Panel",
    "rank_cohort",
    "render_panel",
    "screen_label",
]
