---
name: registration
description: Run atlas-to-specimen registration and label propagation correctly - affine initialization, reproducibility settings, background handling, and reintroducing validated ground truth as a registration constraint. Use when registering a new specimen or template, when a registration will not reproduce, when propagated labels look globally misplaced rather than boundary-imprecise, or when deciding whether enough ground truth exists to constrain a re-registration round.
---

# Registration and label propagation

`register()` and `propagate_labels()` are the first stage of the loop this
package supports: they produce the zero-cost baseline annotation that
`characterize`, `algorithms`, and `evaluate` then correct. Errors introduced
here look like segmentation errors downstream and are frequently misdiagnosed
as such — a boundary-imprecision problem is fixed by fitting a better
threshold; a registration problem is not, and time spent trying is wasted.
Distinguishing the two before fitting anything is `neuroanatomy` and
`segmentation-correction`'s job; this skill is about not introducing a
registration problem in the first place, and about recognizing the signature
of one when it shows up regardless.

```python
from atlas_refine.registration import MetricTerm, RegistrationConfig, register, propagate_labels

registration = register(moving=template, fixed=specimen_image)
propagated = propagate_labels(template_labels, specimen_image, registration)
```

## Two stages, and why both are explicit

`register()` always solves an affine first, on image intensity alone and
without spatial masking, then hands it to ANTs as `initial_transform` for the
deformable stage. Neither substitution is safe: the deformable stage does not
itself estimate an affine component, so skipping the first leaves large-scale
misalignment for the deformable stage to absorb, which it does badly. Masking
the affine stage to a region of interest removes most of the voxels that
constrain a global transform, so the affine degrades rather than improves.

If a propagated label is offset by something larger than the deformable
stage's typical residual — not "boundary is a few voxels wide," but "the
label is at the wrong end of the specimen" or mirrored — suspect the affine
first, not the correction algorithm. Fitting a threshold cannot repair a
transform that never found the right global alignment.

## Reproducibility is not the default

```python
config = RegistrationConfig(random_seed=1, threads=1)
register(moving=template, fixed=specimen_image, config=config)
```

Two settings are required together, and `RegistrationConfig`'s defaults
already set both — the point of naming this here is what breaks if you
override either one without the other. The similarity metric samples voxels
stochastically, so a run needs a seed; ANTs reads that seed from the
`ANTS_RANDOM_SEED` environment variable, not from a value passed to the
registration call, and this package sets it into the environment for the
duration of the call and restores whatever was there before. Multithreaded
reduction is its own separate source of nondeterminism, independent of the
seed, so single-threaded execution is required as well.

`random_seed=0` is rejected outright — the underlying implementation treats
zero as "seed from the clock," which silently produces a different
transform on every run while looking like a fixed, reasonable setting. If a
registration needs to be exactly reproducible (for validating the pipeline
itself, or for a leave-one-out comparison where the only thing that should
change between runs is the input), verify it the way `test_registration.py`
does: register the same pair twice and check the warped outputs are
bit-identical, not just visually similar.

## Background subtraction

`subtract_background=True` (the default) offsets both volumes by their modal
background before registering, so the intensity metric is not dominated by a
background plateau that carries no informative gradient. This changes what
the metric optimizes but not the geometry it recovers — confirmed by
`test_background_subtraction_does_not_change_the_geometry`, which registers
the same pair with and without subtraction and checks the recovered
transform agrees to within a small tolerance. Disable it only if the
specimen's background is not a clean unimodal peak — a volume with strong
shading or regional clearing failure can have a background whose "mode" is
not representative, in which case subtracting it can remove signal instead
of removing noise. Check with `image-analysis`'s `background_spread` field
first.

## Propagating labels

```python
propagate_labels(template_labels, specimen_image, registration, values=[1, 2, 6])
```

`propagate_labels` applies the full transform chain in one resampling step
and uses nearest-equivalent interpolation throughout — never linear or
cubic. A label volume interpolated linearly produces fractional values
between label indices that correspond to no label at all; composing the
transform in stages compounds interpolation error at each step. `values`
restricts propagation to specific label indices, useful when only one
structure's ground truth is being reintroduced (see below) rather than the
whole parcellation.

`labels.frame` must equal `registration.source_frame`, or propagation raises
rather than silently resampling from the wrong space. This is a real
failure mode, not a defensive check for its own sake: a transform's source
and target frames are named strings, and mismatching them — applying a
specimen→template transform to a label that is actually in template→specimen
space, or reusing a transform fitted against one template version for a
different one — produces a label that resamples without error and looks
plausible at a glance. The only defense against that is checking frame
names match what you intend, since the geometry alone cannot tell you which
direction a chain runs. A transform intended for one target should not be
reused against a different one on the assumption that "close enough"
templates produce compatible chains — verify by re-registering, or by
checking a known landmark's position after applying the chain, not by
comparing template dimensions or voxel sizes alone.

## Declared units are not reliable

Neither `register` nor `propagate_labels` reads spacing from a file header —
both operate on `Volume` objects, which carry spacing as an explicit,
separately-validated field (see `io.volumes`). This is deliberate: NIfTI and
NRRD headers frequently declare a unit that does not match the stored voxel
size, and toolkits differ in whether they honor the declared code at all.
Registering two volumes whose actual and declared units disagree by three
orders of magnitude — the specific, common failure is a file whose header
says millimeters while the pixdim values are in micrometers — produces a
transform that is geometrically nonsensical but will still converge to
*something*, since the optimizer has no way to know the scale is wrong.
Construct every `Volume` with spacing supplied explicitly, verified against
the physical extent you expect the specimen to have, rather than trusting
whatever a loader reports.

## Reintroducing ground truth as a registration constraint

```python
term = MetricTerm(fixed=specimen_annotation, moving=template_annotation, weight=1.0)
registration = register(moving=template, fixed=specimen_image, extra_metrics=[term])
```

This is the mechanism behind the "reintroduce" stage of the propagate →
correct → validate → reintroduce loop (see the package README's *Purpose*
section): once ground truth exists for a structure, it constrains the
deformable stage directly, using anatomy that image intensity alone may not
localize well — a structure with low contrast against its neighbor is
exactly the case an intensity-only metric places poorly and a matching
label pair places precisely. `extra_metrics` are added alongside the image
metric, not in place of it, weighted by `MetricTerm.weight` against the
image term's implicit weight of 1.0.

Two things this does not do automatically. First, it does not decide
*when* enough ground truth exists to be worth reintroducing — a single
annotation used this way constrains one specimen's registration using
information from that same specimen, which is not the same claim as
constraining registration for the whole cohort from evidence gathered
across it. Second, re-running propagation with a constrained registration
changes the baseline that `characterize` and `evaluate` are measuring
against for every specimen, not only the one whose annotation supplied the
constraint — treat a new round of propagation as a new baseline requiring
its own characterization, not as a strictly better version of the old one
to be assumed superior without checking. A registration constrained this
way should be evaluated through the full correction chain — propagate, fit,
validate — not on raw label agreement immediately after registering, since
a registration whose residual error is now of a form the correction stage
removes can score lower immediately after registering and higher once the
whole chain runs.

## Guidelines

- Suspect the affine before the correction algorithm when a propagated
  label is grossly misplaced rather than boundary-imprecise.
- Never compare or reuse transforms across template versions on the
  assumption that similar dimensions mean compatible geometry; re-register
  or check a landmark.
- Verify reproducibility by comparing warped output arrays directly when it
  matters, not by inspecting the transform files or trusting the config.
- A `MetricTerm` improves one registration's placement using that
  specimen's own ground truth; it is not evidence the change generalizes to
  the rest of the cohort until the whole chain is re-evaluated.
