---
name: visual-review
description: Review produced labels on specimens that carry no manual annotation - rendering them over the image, reading the plausibility screen, and deciding which need a human. Use after applying a correction across a cohort, when iterating over candidate algorithms, or when a numeric result looks fine but the output has never been seen.
---

# Reviewing labels nobody has looked at

A correction is fitted on the few specimens with a manual annotation and applied
to the rest. That remainder is the majority of the output and nothing checks it.
This is the gap to close, and closing it does not require more annotation - only
looking, and knowing what to look for.

## Where this applies, and where it does not

Use agreement with a manual annotation wherever one exists. It is exact, cheap,
and cannot be argued with; no visual judgement improves on it, and substituting
one is a downgrade.

Everything here is for specimens where no such annotation exists. The tools
report that a label is **implausible**, never that it is correct. A specimen
that raises no concern has not been verified — only found unremarkable.

## The two-step pass

Screen first, then render what the screen questions. Screening is cheap and runs
over a whole cohort; rendering and looking is the expensive part, so spend it
where the numbers already point.

```
screens = [screen_label(label, image, specimen=sid, lateral_axis=1) for sid ...]
for review in rank_cohort(screens):
    print(review.describe())
```

`rank_cohort` puts absolute concerns first — a fragmented or lopsided label is
wrong whatever the rest of the cohort looks like — then adds cohort-relative
deviation. Below three specimens the relative part is suppressed, because a
standard deviation over two points says nothing.

Set `lateral_axis` when the structure is bilateral. Leave it unset and the
symmetry check is skipped rather than guessed; a mirrored label is one of the
failures this catches, and guessing the axis would hide it.

## What the screen is telling you

**Fragmentation** (`cohesion` below 0.90, or many `debris` pieces). A threshold
admitted scattered noise. Usually the threshold is too permissive, or the label
is being applied to a specimen whose background sits higher than the one it was
fitted on.

**Lopsidedness** (`lateral_asymmetry` above 0.15). A one-sided failure: a
registration that collapsed on one hemisphere, or a structure clipped by the
field of view. Check the volume's extent before blaming the correction.

**Cavities.** A threshold has punched holes through the structure's own
interior. The threshold is too aggressive for this specimen, or the interior is
genuinely dimmer here than where the parameter was fitted.

**Boundary sharpness at or below zero.** The intensity does not rise across the
label surface at all, so the boundary is not on an edge. This is the strongest
single indicator that a label is misplaced rather than merely imprecise, and it
points at registration rather than at the threshold.

**Volume far from the cohort.** Only interpretable relative to the others, and
only once there are enough specimens to have a cohort. A large deviation with no
absolute concern often means the structure is genuinely different in that
specimen, not that the label is wrong.

## Reading a render

`render_panel` draws label outlines over greyscale intensities, sliced through
the label's own extent. Pass the propagated label and the correction together —
seeing what the correction *did* is more informative than seeing either alone.

Outlines, not fills: the intensities underneath are what tell you whether the
boundary follows a real edge.

Look for these, in this order:

1. **Missing regions.** A lobe or chunk absent on one side or at one end. This
   is registration failure and no threshold will recover it.
2. **Leakage into a neighbour.** The outline crossing into an adjacent
   structure. Check what the structure borders first — where the neighbour is
   brighter, an outward error cannot be removed by an upper-open threshold.
3. **Boundary offset.** The outline running parallel to a visible edge but
   consistently inside or outside it. This is the failure a fitted parameter
   *can* fix.
4. **Speckle.** Scattered voxels away from the structure, confirming what
   `cohesion` reported.

Distinguishing 1 and 2 from 3 is the point of looking. The first two are
registration problems and should stop you from spending more effort on
thresholds; the third is a parameter problem and is worth fitting.

## Report what you saw, not a score

Do not invent a number for a render. A visual review produces a judgement and a
reason — "the outline crosses into brighter tissue along axis 0, slices 240-300"
— which is actionable and checkable. A fabricated accuracy figure is neither,
and it competes with the exact measurements that do exist.

State plainly which specimens you looked at and which you did not. A cohort
where three of ten were reviewed has seven unreviewed specimens, and saying so
is part of the result.

## Limits worth stating when you report

A render is a handful of 2-D slices through a 3-D volume. A thin structure can
be wrong between the slices drawn, and a small error anywhere can be missed
entirely. Increase `per_axis` when a structure is thin or branching, and treat
absence of a visible problem as weak evidence.

Array axes are drawn as stored, with no anatomical reorientation. Do not
describe a finding as dorsal, lateral or rostral unless the mapping from array
axis to anatomy has been established for that cohort — say "axis 0, slice 320"
instead, which is unambiguous and checkable.
