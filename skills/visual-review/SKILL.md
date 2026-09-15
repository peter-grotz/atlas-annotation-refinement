---
name: visual-review
description: Review produced labels by screening every specimen and rendering every new one - reading the plausibility measurements, looking at the label over the image, and deciding which need a human. Use whenever a correction is applied to a specimen, especially a new cohort where no manual annotation exists and this is the only verification available.
---

# Reviewing labels nobody has looked at

A correction is fitted on the few specimens with a manual annotation and applied
to the rest. That remainder is the majority of the output and nothing checks it.
This is the gap to close, and closing it does not require more annotation - only
looking, and knowing what to look for.

## Two parallel obligations, not a hierarchy

Agreement with a manual annotation and visual review answer different questions,
and neither substitutes for the other.

**Dice asks how accurate a correction is**, on the specimens that can be scored.
Where an annotation exists it is exact and cannot be argued with, so no visual
judgement improves on it *for that specimen*.

**Screening and rendering ask whether a label is plausible at all**, on every
specimen — including the ones nothing else can check. On a typical cohort that
is most of them: two of ten brains carried an independent cortex annotation, so
Dice covered two while the screen covered ten and was the only check on eight.

On a new cohort the asymmetry is total. Applying a fitted correction to ten
fresh brains yields no Dice anywhere, because re-annotating is the cost the
method exists to avoid. Visual review is then the entire verification story.

So: **screen every specimen, render every new one, and compute Dice wherever an
annotation happens to exist.** Do not treat rendering as a fallback for when
Dice is unavailable and something already looks wrong — rendering a new sample
once is cheap, and it is the only look anyone will get.

The tools report that a label is **implausible**, never that it is correct. A
specimen that raises no concern has not been verified — only found
unremarkable. That limitation applies equally whether or not Dice is available.

This is not hypothetical. A 24% hemispheric asymmetry, against a cohort range of
0.009 to 0.121, was found on a brain with no annotation; follow-up measurement
showed the correction had inherited it from registration rather than caused it.
No amount of Dice on the two annotated brains would have surfaced it, because
the failure lived entirely in the unscoreable majority.

## The pass

Screen every specimen; the measurements are cheap and run over a whole cohort.
Then render: every specimen on a cohort seen for the first time, and thereafter
whatever the screen questions. Rendering and looking is the expensive part, so
on a cohort already reviewed once, spend it where the numbers point.

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

**Fragmentation** (`cohesion` below 0.90, or `components` above five). A
threshold admitted scattered noise, or the structure has broken into pieces.
Usually the threshold is too permissive, or the label is being applied to a
specimen whose background sits higher than the one it was fitted on.

Read `cohesion`, not `debris`. Every real label carries hundreds of
single-voxel specks — measured on a ten-brain cohort, all ten had between 263
and 696 of them while holding over 99.9% of their volume in one piece. The
count is reported for information; the share of volume is what matters.

**Lopsidedness** (`lateral_asymmetry` above 0.15). A one-sided failure: a
registration that collapsed on one hemisphere, or a structure clipped by the
field of view. Check the volume's extent before blaming the correction.

**Punctures.** A threshold has punched holes through the structure's own
interior — too aggressive for this specimen, or the interior is genuinely
dimmer here than where the parameter was fitted.

Read `punctures`, not `enclosed`. A sheet-like structure surrounds whatever it
wraps, so a cortical label legitimately encloses most of the brain: measured on
a hollow shell, `enclosed` reaches 89% of the label while nothing is wrong with
it. `punctures` counts only pockets too small to be wrapped anatomy.

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
