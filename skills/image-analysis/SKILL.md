---
name: image-analysis
description: Describe volumetric acquisitions without annotations - contrast, shading, striping, separability, noise, extent - and use those properties to choose which specimen to annotate next and to predict where a fitted correction will not hold. Use when starting on a new cohort, when a correction transfers unevenly across specimens, or when deciding where scarce annotator time should go.
---

# Image analysis

Every measurement here runs on an acquisition alone. A whole cohort can be
described before any of it is annotated, which is what makes it useful for
deciding where to spend annotation effort.

```python
from atlas_refine.analysis import profile_image, position_in_cohort, next_to_annotate

profiles = {sid: profile_image(load(sid)) for sid in cohort}
print(next_to_annotate(profiles, annotated=already_done).describe())
```

## Reading the fields

| field | what a high value means | what it breaks |
|---|---|---|
| `contrast` | tissue well separated from background | — |
| `separability` | tissue classes split cleanly by intensity | **low: no threshold can work, whatever the value** |
| `shading` | strong low-frequency illumination variation | a single global threshold; local gradients survive |
| `stripe_anisotropy` | illumination striping | gradient-driven methods, which lock onto stripe edges |
| `background_spread` | regional clearing failure | one global background level describing the volume |
| `noise` | high-frequency variation | small structures and thin boundaries |
| `extent` | tissue extent per axis | reflects dissection, drives atlas coverage mismatch |

`separability` is the one to read first. It bounds every threshold-based
correction, so a low value means reaching for a different family rather than a
different parameter.

## Choosing what to annotate next

Annotator time is the binding constraint, so this decision matters more than
any parameter. Take the specimen furthest from those already annotated: a rule
fitted on the annotated set holds least well there, so an annotation there is
the most informative.

With nothing annotated yet, take the specimen furthest from the cohort centre.
A first annotation on an atypical specimen reveals the range a correction must
span; one on a representative specimen does not.

## Predicting where a correction will not hold

`position_in_cohort` flags specimens more than two standard deviations from the
cohort on any field. Apply a fitted correction to those, but treat the output as
provisional and report the flag alongside it. A correction failing silently on
one specimen is worse than one failing loudly.

The same measurement should also inform *which family* to try. A specimen
unlike those a rule was fitted on is a reason to reconsider the rule, not only
to annotate it.

## Guidelines

- Run this before any fitting, on the whole cohort. It is cheap and needs no
  annotations.
- Report `n` whenever relating these fields to a fitted parameter. At three or
  four specimens most fields will order them plausibly by chance; see the
  `evidence-analysis` skill.
- Do not treat a flagged specimen as bad data. It is usually where the method
  needs to become more general.
