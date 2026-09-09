---
name: neuroanatomy
description: Establish what an anatomical structure borders and whether each neighbour is brighter, darker, or indistinguishable, then use that to rule correction families in or out before any parameter is fitted. Use when starting on a structure for the first time, when a correction fails for reasons the error signature does not explain, or when a structure is not yet described.
---

# Structural context

An error signature says how a label is wrong. It does not say which corrections
are possible, because that depends on what lies across each boundary. Two
structures can present near-identical signatures and admit opposite corrections.

```python
from atlas_refine.analysis import screening_brief, describe_structure, NeighbourRelation

print(screening_brief("isocortex"))
```

## Why this decides things a parameter search cannot

Growing a label recovers material where it borders something distinguishable,
and admits a neighbour where it does not. That is anatomy, not image statistics,
and no sweep will reveal it — the search simply returns the least bad setting of
a rule that cannot work.

Worked example. Isocortex borders background on its outer surface (darker) and
white matter on its inner surface (brighter). Because the inner neighbour is
*brighter*, expanding the label admits white matter and no upper-open threshold
rejects it. Expansion is therefore ruled out for cortex — while for a structure
whose every neighbour is dimmer, the same operation works.

## Describing a structure that is not yet known

This is the expected case, not an exception. The catalogue is deliberately
small: an entry that is merely plausible is worse than an absent one, because
absence prompts a description while a wrong entry is acted on.

1. Take the neighbours from the atlas parcellation's adjacency.
2. Establish each relation by measuring intensity either side of the propagated
   boundary — the `boundary` module's profile gives this directly — or from the
   literature where measurement is ambiguous.
3. State the source on every relation.

```python
describe_structure(
    "dentate_nucleus",
    boundaries=[
        NeighbourRelation(
            neighbour="cerebellar white matter", relation="brighter", aspect="any",
            source="measured: intensity either side of the propagated boundary",
        ),
    ],
    typical_thickness=6,
)
```

Once recorded it is available to every later cohort.

## Reading the screen

- **supported** — the anatomy permits it; whether it works is still an
  empirical question.
- **ruled out** — it cannot work here. Do not fit it; the result would be
  meaningless rather than merely poor.
- **caution** — it may work but a stated condition applies, such as a structure
  too thin to erode or a boundary that no intensity rule can place.

A boundary whose neighbour is `similar` is registration-limited. Error there
will not respond to any correction, and effort belongs in the registration
instead.

## Guidelines

- Screen before fitting, not after a family disappoints.
- Record provenance on every relation. A wrong relation produces a confident
  wrong screening result, which is harder to detect than an absent one.
- Prefer measurement over recollection: the relation that matters is the one in
  *these* acquisitions, which can differ from the canonical description.
