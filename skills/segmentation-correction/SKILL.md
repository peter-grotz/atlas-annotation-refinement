---
name: segmentation-correction
description: Read an error signature describing how a propagated label differs from a manual annotation, decide which correction family suits it, and fit and validate that correction. Use when a registration-propagated label needs correcting against ground truth, when choosing between thresholding, morphological and contour-based approaches, or when no existing family fits and a new one must be authored.
---

# Correcting a propagated label

The correction a structure needs follows from how its label fails, not from
what the structure is. Measure first, then choose.

```python
from atlas_refine.characterize import characterize
from atlas_refine.analysis import screening_brief, profile_boundary
from atlas_refine.experiments import TrialLog

signature = characterize(image, propagated_label, annotation)
print(signature.summary())
print(screening_brief(structure))                     # what the anatomy forbids
print(TrialLog(path).families_tried(signature.to_dict()))   # what already failed here
```

Do all three before proposing anything. The signature says what kind of failure
it is, the anatomy rules families out, and the log says which were already tried
against a comparable failure and why they failed.

## Reading the signature

| `dominant_error` | meaning | direction |
|---|---|---|
| `over-coverage` | the label is too large | remove material |
| `under-coverage` | the label is too small | add material |
| `displacement` | right volume, wrong place | **move it — neither adding nor removing helps** |
| `close-agreement` | small errors only | probably leave it |

`displacement` is the one most easily misread. A label holding the right amount
of material can disagree over most of its extent, and the sign of a negligible
count difference is arbitrary, so a rule keyed on voxel counts alone will
propose resizing when the label needs moving.

Then `intensity_separation`, which decides whether intensity can work at all:

- **positive** — the excess and the structure barely overlap in intensity. A
  threshold can separate them.
- **near zero or negative** — they are drawn from the same distribution. No
  threshold can separate them, at any parameter. The correction must come from
  geometry, from registration, or not at all.

Then the geometry. Excess that is surface-adjacent and largely one connected
component is a shell, and shells respond to thresholding and erosion. Excess
scattered through many components is not a shell and will not.

Finally `subtractive_recall_ceiling`: the best recall any material-removing
correction can reach. Search stops there. Exceeding it requires growing or
moving the label, which is a different family and often a different problem.

## Choosing a family

| signature | family |
|---|---|
| over-coverage, one shell, separation positive | `contrast_threshold` |
| over-coverage, bounded by both brighter and dimmer neighbours | `band_threshold` |
| under-coverage, structure partly below a single cut but contiguous | `hysteresis_threshold` |
| over-coverage, shell of constant thickness, separation near zero | `morphological_cleanup` |
| displacement, or separation negative at every boundary | none — return to registration |

Prefer the fewest parameters that fits. A family with more free parameters than
there are independent annotations cannot be validated, and fitting is refused.
A second parameter that improves the fit by less than the spread across the grid
is not buying anything.

## Fitting and validating

```python
from atlas_refine.evaluate import fit, measure_transfer, leave_one_out

result   = fit(algorithm, store.fitting_set(structure), loader)
transfer = measure_transfer(algorithm, result.params, fitted_on, held_out, loader)
```

Only independent annotations may be fitted against. An annotation edited from an
algorithm's output has its optimum at the parameter that produced it, so scoring
against it reports the fit back to itself.

Held-out performance matching or exceeding the fitted score means the parameter
is not tuned to its examples. A large gap means it is.

## Authoring a new family

Only after the log shows the existing families were tried against a comparable
signature and fell short, or the signature is unlike anything recorded. A family
added for a small margin on one signature widens the search space for every
later structure and rarely survives a second annotation.

See the repository's `docs/authoring.md`.

## Guidelines

- Record every attempt, including failures, with a reason that says *why* the
  family was unsuitable rather than restating its score. "Inside a thick bright
  structure the local mean is the structure itself" rules a family out for every
  future structure with that signature; "scored 0.63" rules out nothing.
- Express parameters relative to per-specimen intensity statistics, never as
  absolute intensities. Absolute values do not transfer between acquisitions.
- Stop at the ceiling. Continuing past it searches for a correction that cannot
  exist.
