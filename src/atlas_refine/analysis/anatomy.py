"""What a structure borders, and what that rules in or out.

An error signature says how a label is wrong. It does not say which corrections
are possible, because that depends on what lies on the other side of each
boundary. Two structures can present near-identical signatures and admit
opposite corrections: growing a label recovers material where it borders
something distinguishable, and admits a neighbour where it does not.

That is anatomical context, not image statistics, and it is the difference
between a correction that works and one that fails for a reason no amount of
parameter search will reveal.

Knowledge here is a cache, not a closed catalogue. A structure absent from it is
expected, not exceptional: describe it from atlas parcellation neighbours and
literature, record the description, and it becomes available thereafter. Every
entry carries its provenance, because a wrong relation produces a confident
wrong screening result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Sequence

#: How a neighbour's intensity compares with the structure's own.
Relation = Literal["brighter", "darker", "similar", "unknown"]

#: Where a boundary lies relative to the structure's surface.
Aspect = Literal["outer", "inner", "lateral", "any"]


@dataclass(frozen=True)
class NeighbourRelation:
    """One boundary of a structure, and what lies across it."""

    neighbour: str
    relation: Relation
    aspect: Aspect = "any"
    #: Where this relation comes from: an atlas parcellation adjacency, a
    #: measurement on this cohort, or the literature. A relation asserted
    #: without provenance cannot be checked when it turns out to mislead.
    source: str = "unspecified"
    note: str = ""

    @property
    def separable_by_intensity(self) -> bool:
        return self.relation in ("brighter", "darker")


@dataclass(frozen=True)
class StructureContext:
    """A structure and the boundaries that constrain how it can be corrected."""

    name: str
    boundaries: tuple[NeighbourRelation, ...]
    #: Typical thickness in voxels at the working resolution, where known.
    #: A structure thinner than the depth used to sample its interior needs the
    #: fallback path, and a correction that erodes will consume it.
    typical_thickness: int | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def by_aspect(self, aspect: Aspect) -> tuple[NeighbourRelation, ...]:
        return tuple(b for b in self.boundaries if b.aspect in (aspect, "any"))

    @property
    def separable_boundaries(self) -> tuple[NeighbourRelation, ...]:
        return tuple(b for b in self.boundaries if b.separable_by_intensity)

    @property
    def indistinct_boundaries(self) -> tuple[NeighbourRelation, ...]:
        return tuple(b for b in self.boundaries if b.relation == "similar")

    @property
    def fully_separable(self) -> bool:
        """Whether every boundary can in principle be placed by intensity."""
        return bool(self.boundaries) and all(
            b.separable_by_intensity for b in self.boundaries
        )


#: Relations established by measurement on light-sheet volumes of cleared
#: tissue. Deliberately small: an entry that is merely plausible is worse than
#: an absent one, because absence prompts a description while a wrong entry is
#: acted on. Extend through `describe_structure`.
KNOWN: dict[str, StructureContext] = {
    "ventricles": StructureContext(
        name="ventricles",
        boundaries=(
            NeighbourRelation(
                neighbour="surrounding parenchyma",
                relation="brighter",
                aspect="any",
                source="measured: lumen sits at the image background level while "
                "surrounding tissue is well above it",
                note="A fluid-filled lumen, so the interior may sit at or below "
                "the modal background. Normalising to the interior level is "
                "unstable for that reason; normalise to the surrounding tissue.",
            ),
        ),
        note="Thin and branching. Many specimens have no voxel deep enough to "
        "sample an interior core, so any method depending on one needs a "
        "fallback.",
    ),
    "isocortex": StructureContext(
        name="isocortex",
        boundaries=(
            NeighbourRelation(
                neighbour="background and meninges",
                relation="darker",
                aspect="outer",
                source="measured: propagated over-coverage sits near background "
                "while cortex sits well above it",
            ),
            NeighbourRelation(
                neighbour="white matter",
                relation="brighter",
                aspect="inner",
                source="measured: myelinated tracts are the brightest tissue in "
                "these acquisitions",
                note="Because the neighbour is brighter rather than dimmer, "
                "growing the label admits white matter. Correction at this "
                "boundary must come from registration or from a bounded band, "
                "not from expansion.",
            ),
        ),
        note="A sheet of roughly constant thickness with two boundaries of "
        "opposite polarity. A single one-sided threshold can only address one "
        "of them.",
    ),
}


def describe_structure(
    name: str,
    boundaries: Sequence[NeighbourRelation],
    *,
    typical_thickness: int | None = None,
    note: str = "",
    register: bool = True,
) -> StructureContext:
    """Record a structure not already known.

    Intended for the common case of meeting an unfamiliar structure. Derive the
    neighbours from the atlas parcellation's adjacency, establish each relation
    by measuring intensity on either side of the propagated boundary or from the
    literature, and state the source on each relation.
    """
    context = StructureContext(
        name=name,
        boundaries=tuple(boundaries),
        typical_thickness=typical_thickness,
        note=note,
    )
    if register:
        KNOWN[name] = context
    return context


@dataclass(frozen=True)
class OperationScreen:
    """Which corrections the anatomy permits, before any is tried."""

    structure: str
    ruled_out: dict[str, str] = field(default_factory=dict)
    supported: dict[str, str] = field(default_factory=dict)
    cautions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.structure}:"]
        for name, reason in sorted(self.supported.items()):
            lines.append(f"  supported   {name} — {reason}")
        for name, reason in sorted(self.ruled_out.items()):
            lines.append(f"  ruled out   {name} — {reason}")
        for caution in self.cautions:
            lines.append(f"  caution     {caution}")
        return "\n".join(lines)


def screen_operations(context: StructureContext) -> OperationScreen:
    """Rule correction families in or out from the structure's boundaries alone.

    This runs before any parameter is fitted. A family ruled out here fails for
    a reason a parameter search cannot surface: the search will simply return
    the least bad setting of an inapplicable rule.
    """
    ruled_out: dict[str, str] = {}
    supported: dict[str, str] = {}
    cautions: list[str] = []

    separable = context.separable_boundaries
    indistinct = context.indistinct_boundaries

    if separable:
        sides = ", ".join(f"{b.neighbour} ({b.relation})" for b in separable)
        supported["threshold"] = (
            f"at least one boundary is separable by intensity: {sides}"
        )
    else:
        ruled_out["threshold"] = (
            "no boundary differs in intensity from its neighbour, so no "
            "threshold can place one; the correction must come from registration"
        )

    brighter = [b for b in context.boundaries if b.relation == "brighter"]
    darker = [b for b in context.boundaries if b.relation == "darker"]

    if brighter and darker:
        supported["band_threshold"] = (
            "boundaries of opposite polarity, so the structure is bounded from "
            "both sides and a two-sided band is meaningful"
        )
    elif len(context.boundaries) == 1:
        cautions.append(
            "a single boundary means a one-sided threshold is sufficient; a "
            "band spends a second parameter without a second constraint to use it"
        )

    if brighter:
        names = ", ".join(b.neighbour for b in brighter)
        ruled_out["dilate_then_threshold"] = (
            f"the label borders brighter tissue ({names}), so expanding it "
            f"admits that neighbour and no upper-open threshold rejects it"
        )
    else:
        supported["dilate_then_threshold"] = (
            "every neighbour is dimmer than the structure, so expansion can be "
            "recovered by thresholding"
        )

    if indistinct:
        names = ", ".join(b.neighbour for b in indistinct)
        cautions.append(
            f"the boundary with {names} cannot be placed by intensity; error "
            f"there is registration-limited and will not respond to correction"
        )

    if context.typical_thickness is not None and context.typical_thickness <= 4:
        cautions.append(
            f"typically {context.typical_thickness} voxels thick: erosion will "
            f"consume it, and any interior sampled at depth needs a fallback"
        )

    return OperationScreen(
        structure=context.name,
        ruled_out=ruled_out,
        supported=supported,
        cautions=cautions,
    )


def context_for(name: str) -> StructureContext | None:
    """Look up a structure, returning None when it has not been described."""
    return KNOWN.get(name)


def screening_brief(name: str) -> str:
    """A short statement of what is known and what it implies, for a structure.

    Returns guidance on how to proceed when the structure is not yet described,
    rather than failing: an unfamiliar structure is the expected case.
    """
    context = context_for(name)
    if context is None:
        return (
            f"'{name}' has not been described. Before fitting, establish what it "
            f"borders and whether each neighbour is brighter, darker, or "
            f"indistinguishable — from the atlas parcellation's adjacency, from "
            f"intensity measured either side of the propagated boundary, or from "
            f"the literature. Record it with describe_structure() so the next "
            f"cohort inherits it. Without this, a family ruled out by the anatomy "
            f"will still be fitted, and will return the least bad setting of a "
            f"rule that cannot work."
        )
    return f"{context.note}\n\n{screen_operations(context).summary()}" if context.note \
        else screen_operations(context).summary()
