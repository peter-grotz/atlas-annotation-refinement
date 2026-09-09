"""Measurement and interpretation supporting correction selection.

Four kinds of question, each answered by one module:

    image_stats  What are this acquisition's properties, without annotations?
    cohort       Where does a specimen sit relative to the rest, and what
                 should be annotated next?
    boundary     What does intensity do across a label's surface, and can a
                 boundary be placed by intensity at all?
    anatomy      What does the structure border, and what does that rule out?
"""

from .anatomy import (
    KNOWN,
    NeighbourRelation,
    OperationScreen,
    StructureContext,
    context_for,
    describe_structure,
    screen_operations,
    screening_brief,
)
from .boundary import BoundaryProfile, profile_boundary
from .cohort import (
    COMPARISON_FIELDS,
    CohortPosition,
    FeatureAssociation,
    associate_with_parameter,
    next_to_annotate,
    position_in_cohort,
    standardise,
)
from .image_stats import ImageProfile, otsu_separability, profile_image

__all__ = [
    "KNOWN",
    "NeighbourRelation",
    "OperationScreen",
    "StructureContext",
    "context_for",
    "describe_structure",
    "screen_operations",
    "screening_brief",
    "BoundaryProfile",
    "profile_boundary",
    "COMPARISON_FIELDS",
    "CohortPosition",
    "FeatureAssociation",
    "associate_with_parameter",
    "next_to_annotate",
    "position_in_cohort",
    "standardise",
    "ImageProfile",
    "otsu_separability",
    "profile_image",
]
