# atlas-refine

Tools for correcting atlas-propagated brain annotations against a small
number of manual ones, and for knowing when the correction can be trusted.

## The problem

Registering a specimen to a reference atlas and reverse-transforming the
atlas parcellation gives a complete annotation of every specimen for free.
The boundaries it produces follow atlas anatomy, not the anatomy visible in
each acquisition, so they are systematically — not randomly — wrong.

A small number of hand-corrected specimens measure that disagreement well
enough to fit a minimal correction and apply it to the rest of the cohort.
On one structure this took Dice from 0.905 to 0.976 using a single free
parameter fitted on two annotations, and the fit transferred to a specimen
it never saw. But the same evidence also showed that roughly half the
remaining error is registration failure that no threshold can touch, and
that a two-parameter version of the same correction bought nothing over the
one-parameter version — the extra parameter searched a whole range for a
0.0005 change in score. Knowing which of those is true for a given
structure is the actual difficulty; fitting a threshold is not.

## What this is

A library and a small set of gates, not a pipeline. It does not decide which
correction to try, whether a structure needs a new algorithm, or when a
result is good enough to apply. Those are left to whoever is operating it.
What the code guarantees is that the decision can't be fooled by the most
common ways this kind of fitting goes wrong: too few annotations for the
parameters being fit, an annotation that was edited from an algorithm's own
output being used to score that algorithm, mismatched units or grids, a
contributed algorithm whose justification is the placeholder text it was
scaffolded with.

It is meant to be operated by a person and an agent together. The person
annotates specimens and approves what gets applied to the cohort. The agent
reads the measurements this package produces, decides what to try, and
decides when the evidence is sufficient — using judgment the code
deliberately does not encode. There is no command that runs this loop
automatically; `atlas-refine` exposes three commands, described below, and
everything else is driven through the Python API in an interactive session.

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
*Review* below.

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
that nothing implausible was found. On a real cohort this caught a
registration failure (24% left-right volume asymmetry, against 0.01–0.12
across the rest of the cohort) on a specimen with no manual annotation at
all; Dice couldn't have found it because nothing was there to compute Dice
against. See `skills/visual-review/SKILL.md`.

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
how well it scores — see [docs/authoring.md](docs/authoring.md).

## Skills

Five `SKILL.md` packages under `skills/`, in the Agent Skills open standard,
carry the judgment that turns these measurements into decisions:

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

None of this replaces judgment about which correction to try — it constrains
what a wrong or lazy judgment is able to get away with.

## Testing

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

216 tests. Registration is checked against synthetic phantoms with a known
displacement, including reproducibility across repeated runs — which
requires a non-zero random seed and single-threaded execution; the seed and
thread count ANTs treats as "use the clock" otherwise are rejected.

## License

MIT
