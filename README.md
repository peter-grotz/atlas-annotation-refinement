# atlas-refine

Tools for correcting atlas-propagated brain annotations against a small
number of manual ones, and for knowing when the correction can be trusted.
Built to be operated by a human and an agent working together — neither
alone is enough, for reasons this document tries to make concrete rather
than assert.

## The problem

Registering a specimen to a reference atlas and reverse-transforming the
atlas parcellation gives a complete annotation of every specimen in a
cohort, at no manual cost. The boundaries it produces follow atlas anatomy,
not the anatomy actually visible in each acquisition, so they are wrong —
but *systematically* wrong: the same deformation resolves the same
structure the same way across specimens, which is what makes the error
correctable from a small amount of evidence rather than needing to be
re-annotated from scratch every time.

The correction that fixes one structure's error is often the wrong
correction for another, and not by a small margin — the two commonest
failure modes call for operations that are close to opposite. A label that
over-covers a lower-intensity margin (cortex spilling into background) is
fixed by an intensity threshold: keep what's brighter than the margin,
discard the rest. A label that under-covers an interior with real contrast
on both sides (a structure the deformation shrank away from) needs the
opposite move — expansion, then a contour or deformable method to stop the
expansion at the right edge. Applying either operation to the other failure
mode doesn't just underperform, it actively damages the label: expanding a
label that already over-covers makes the over-coverage worse. Whether a
given structure is failing the first way or the second is not knowable from
its name or from what it typically looks like — it's a property of how the
registration resolved *that* structure in *that* cohort, visible only after
propagation, and it can flip between two structures in the same cohort or
even between two neighbours of the same structure. That's the shell/interior
finding this repo produced directly: cortex's outer boundary and inner
boundary needed opposite treatment, in the same specimens, on the same run.

This also isn't a regime where you can learn your way out of the
problem. The number of hand-corrected examples per structure is one to
five, because each one costs an expert hours. Supervised segmentation
models need orders of magnitude more than that, so the method has to work
well below the sample size where learning becomes reliable, or it isn't
usable at all here. What's left is closer to program synthesis from a
handful of examples than to fitting a model: a compositional space of
thresholds, morphological operations, connectivity constraints, and
deformable methods, searched using a measured description of *how* the
current label is wrong to propose only the operations that could plausibly
fix that specific failure, then kept only if they hold up on annotations
excluded from the search.

Two things make that search viable instead of a way to fool yourself with
enough attempts. First, the reasoning about which operation suits which
failure — what a structure borders, whether that neighbour is brighter or
darker, whether an interior has any contrast to work with — is exactly the
kind of judgment call that doesn't reduce to a formula, but does reduce to
evidence a reader (human or agent) can weigh directly. Second, the check on
any proposal is exact: agreement with a manual annotation on data the
proposal never saw. That combination — reasoning that has to stay open-
ended, checked against a criterion that can't be argued with — is what this
repository is built around, and it's why the tool is a set of measurements
and gates for something that reasons, not a fixed pipeline and not a model
you train once.

## What this is, and how it's meant to run

A library and a small set of gates, not a pipeline. It does not decide
which correction to try, whether a structure needs a new algorithm, or when
a result is good enough to apply to the cohort — none of that is
encodable in advance, for the reasons above. What the code guarantees
instead is that whoever *is* making those calls can't be fooled by the
common ways this kind of fitting produces a plausible wrong answer: too few
annotations for the parameters being fit, an annotation that was edited
from an algorithm's own output being used to score that same algorithm,
mismatched units or coordinate frames, a contributed algorithm whose
justification is the placeholder text it was scaffolded with.

That's a division of labor between two roles, and the tool needs both
present — it's not built to be safely run by either alone.

**A human anchors it to ground truth and holds approval.** Only a person can
produce the manual annotation the entire method is calibrated against, and
only a person should decide what gets applied back to the cohort. Nothing
here should be trusted to self-certify: the point of the exact-agreement
check is that it's checked by someone who isn't the one who proposed the
fix.

**An agent does the per-structure reasoning that can't be pre-written.**
Deciding whether a label's error is threshold-shaped or registration-shaped,
which of six correction families the anatomy even permits before spending
compute on the rest, whether a two-parameter fit that ties with the
one-parameter version means the second parameter is real or inert, whether
a rendered overlay shows leakage into a neighbour or an acceptable trim —
these are judgments over evidence, made fresh each time, and hard-coding
them is exactly what "different structures fail in opposite ways" rules
out. A human doing this by hand, structure by structure, cohort by cohort,
is the expert-time bottleneck the method exists to relieve; an agent reading
the measurements this package produces is what makes revisiting that
reasoning cheap enough to do for every structure rather than a few.

There is no command that runs this loop automatically — `atlas-refine`
exposes three commands, listed below, and everything else is driven through
the Python API in an interactive session where a human and an agent are
both present. That's a deliberate absence, not a missing feature: the
reasoning step is the part that has to stay under active judgment, and
automating past it would be automating past the thing this whole design
exists to get right.

### What that looked like in practice

On isocortex, this took Dice from 0.905 to 0.976 using a single free
parameter fitted on two independent annotations, and the fit transferred to
a held-out specimen it never saw (0.968). The same pass also showed a
two-parameter version of the same correction — one threshold for the part
of the label facing background, another for the part facing white matter —
tied with the one-parameter version exactly: the second parameter searched
its entire range for a 0.0005 change in score. And screening every
specimen a correction was applied to, including the eight with no manual
annotation to score against, caught a registration failure (24%
left-right volume asymmetry against a 0.01–0.12 range elsewhere in the
cohort) that no threshold could have produced or fixed.

None of those three findings came from a function call. They came from a
person and an agent looking at what the functions returned and asking
whether it meant what it appeared to mean — the tied two-parameter fit
could have been reported as "confirms cortex needs two thresholds" instead
of "the second parameter is dead weight," and the asymmetry could have
gone unlooked-at entirely on a specimen with no ground truth to flag it.
That's the gap this repository's design is aimed at: not the fitting, which
is a few dozen lines per algorithm, but making sure the fitting is never
the last word.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10+. Registration needs ANTsPy; characterisation, analysis, and
review need only NumPy, SciPy, and Matplotlib.

## Usage

### What a human does

Annotate a specimen, save it as `<specimen>__<structure>__<basis>__<annotator>__<YYYY-MM-DD>.seg.nrrd`
in `inbox/`, and run:

```bash
atlas-refine ingest --store ground_truth --inbox inbox \
    --reference "data/<sid>/image.nii.gz" --frame "spec-{sid}" \
    --spacing 80 80 80 --unit um
```

`basis` is the one field intake cannot check: it's whether you started from
the raw propagated label (`pushed`), traced from nothing (`scratch`), or
edited an algorithm's output (`refined`). The first two count as evidence for
fitting; the third does not, because its optimum is pinned to whatever
produced the starting file. Get this field right — it's the fact that
decides what the annotation is allowed to prove.

```bash
atlas-refine status       # holdings by structure, split by admissibility
atlas-refine algorithms   # registered correction families and their parameters
```

Then review what an agent proposes before it's applied to the cohort — see
*Review* below. Approving that step is a human action; nothing in this
package applies a correction to the store on its own.

### What an agent does

Everything between ingest and approval: propagate labels, characterise how
each one fails, screen out corrections the anatomy rules out, fit and score
what's left, and produce evidence for a human to review.

```python
from atlas_refine.algorithms import REGISTRY
from atlas_refine.analysis import next_to_annotate, profile_image, screening_brief
from atlas_refine.characterize import characterize
from atlas_refine.evaluate import check_admissible, fit, measure_transfer
from atlas_refine.io import GroundTruthStore
from atlas_refine.review import rank_cohort, render_panel, screen_label

store = GroundTruthStore("ground_truth")
independent = store.fitting_set("isocortex")

# what's known before any annotation exists
print(screening_brief("isocortex"))
profiles = {sid: profile_image(image) for sid, image in cohort_images.items()}
print(next_to_annotate(profiles, independent).describe())

# how a propagated label is wrong, and what that supports fitting
signature = characterize(image, propagated, reference)
print(signature.summary())

algorithm = REGISTRY.create("contrast_threshold")
check_admissible(algorithm, independent)              # raises if under-evidenced
result = fit(algorithm, independent, loader)
transfer = measure_transfer(algorithm, result.params, independent[:1], independent[1:], loader)

# every specimen the fit above was never scored against
screens = [screen_label(labels[sid], images[sid], specimen=sid) for sid in cohort]
for review in rank_cohort(screens):
    print(review.describe())
render_panel(images[sid], {"propagated": pushed, "refined": corrected}, f"review/{sid}.png")
```

`loader` maps a specimen id to `(image, propagated_label, reference)`. Full
walkthrough in [docs/workflow.md](docs/workflow.md).

## Review

Agreement with a manual annotation is exact wherever one exists, but a
correction fitted on a few specimens gets applied to the rest of the cohort,
where nothing else checks it. `atlas_refine.review` covers that gap two
ways, and both should run on every specimen a correction is applied to —
not only the ones that look suspicious:

- `screen_label` measures whether a label is anatomically plausible with no
  reference needed: fragmentation, hemispheric asymmetry, holes punched
  through a structure's own interior versus anatomy it legitimately
  encloses, and whether the label sits on a real intensity edge at all.
  `rank_cohort` orders a whole cohort by what deserves a closer look.
- `render_panel` draws label outlines over the specimen image, sliced
  through the label's own extent, for whoever looks next — a person or a
  vision-capable agent.

Neither establishes that a label is correct — only that it's implausible, or
that nothing implausible was found. On a real cohort this caught the
registration failure described above on a specimen with no manual
annotation at all; Dice couldn't have found it because nothing was there to
compute Dice against. See `skills/visual-review/SKILL.md`.

## What's in the package

```
src/atlas_refine/
    io/              volume container with unit/frame checks; the annotation store
    registration/    two-stage atlas-to-specimen registration and label propagation
    characterize/    measures how a propagated label disagrees with a reference
    algorithms/      correction families, their registry, and the framework
                     for authoring and admitting new ones
    analysis/        acquisition profiling, cohort comparison, boundary
                     profiling, and anatomical context — all annotation-free
    evaluate/        fitting, transfer scoring, admissibility, leave-one-out
    experiments/     append-only log of every correction attempted
    review/          plausibility screening and diagnostic rendering for
                     specimens with no manual annotation
    cli.py           ingest / status / algorithms
```

Six correction families ship built in — `contrast_threshold`,
`band_threshold`, `hysteresis_threshold`, `morphological_cleanup`, and two
that fit a different parameter per region of a structure,
`regional_threshold` (shell vs. interior, by neighbour brightness) and
`caliber_threshold` (thin vs. thick, by local width). A new family is added
by subclassing `RefinementAlgorithm`, and `algorithms/contribute.py` gates
admission on it behaving correctly and carrying a real justification, not on
how well it scores — see [docs/authoring.md](docs/authoring.md). Authoring a
new family is agent work in the same sense as choosing among existing ones:
the package checks that it's correct and honestly justified, not that the
idea behind it is good.

## Skills

Five `SKILL.md` packages under `skills/`, in the Agent Skills open standard,
carry the judgment that turns these measurements into decisions. They are
the concrete form the "agent" side of this design takes — written for
whichever agent is doing the reasoning to read, not executed by the
package itself:

| skill | answers |
|---|---|
| `image-analysis` | What does this acquisition look like, and which specimen is most worth annotating next? |
| `neuroanatomy` | What does this structure border, and what does that rule out before fitting anything? |
| `segmentation-correction` | Which family suits this error signature, and does the fit transfer to held-out specimens? |
| `visual-review` | What does a rendered label show, and which unscored specimens need a look? |
| `evidence-analysis` | What does this result actually support, at this sample size? |

The modules supply numbers; the skills supply how to read them, because a
number nobody knows how to interpret isn't a capability. Every failure in
building this came from that gap: a proxy metric that pointed the wrong way,
a correlation reported on three points, a family the anatomy had already
ruled out but got fitted anyway.

## Guarantees enforced in code

- **Provenance.** An annotation edited from an algorithm's output can't be
  used to fit or validate that algorithm; `GroundTruthStore` tracks what
  each annotation started from and excludes it automatically.
- **Parameter budget.** Fitting is refused when a family has more free
  parameters than there are independent annotations to constrain them.
- **Geometry.** Volumes carry an explicit unit and coordinate frame;
  operations between them assert compatibility rather than assume it.
- **Contribution rationale.** A newly authored algorithm is rejected if its
  justification is the scaffold's placeholder text, checked by content, not
  just length.

None of this replaces the judgment of the human and agent operating the
tool — it constrains what a wrong or lazy judgment from either of them is
able to get away with. A human can still approve a bad correction and an
agent can still propose one; the guarantees only rule out the specific
failure modes that arise from evidence being misused, not from the wrong
conclusion being drawn from evidence that was used correctly.

## Testing

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

224 tests. Registration is checked against synthetic phantoms with a known
displacement, including reproducibility across repeated runs — which
requires a non-zero random seed and single-threaded execution; the seed and
thread count ANTs treats as "use the clock" otherwise are rejected.

## License

MIT
