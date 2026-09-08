# atlas-refine

An agentic architecture for propagating atlas annotations onto a cohort of
volumetric microscopy acquisitions and refining them against a small number of
manual annotations.

## Overview

Anatomical annotation of a cohort of whole-brain volumes is normally limited by
expert time rather than by compute. Registering each specimen to a reference
atlas and reverse-transforming the atlas parcellation yields a complete
annotation for every specimen at no manual cost, but registration alone places
label boundaries according to atlas anatomy rather than the anatomy visible in
each individual acquisition. The residual disagreement is systematic, not
random: it reflects a consistent bias in how the deformation resolves a
particular structure.

This package implements a workflow that exploits that regularity. A single
specimen is annotated by hand; the disagreement between that annotation and the
propagated label is measured; a correction is fitted to it and applied across
the cohort. Each additional manual annotation both validates the correction on
held-out data and increases the complexity of correction that can be supported.
Once a structure is annotated across the cohort, those annotations are
reintroduced as similarity terms in the registration itself, improving label
placement at the source for subsequent cohorts.

The correction appropriate to a structure follows from how its label fails, and
different structures fail in opposite ways. A label that over-covers a
lower-intensity margin is corrected by an intensity criterion; one that
under-covers an interior with contrast on both boundaries requires a
deformable or contour-based method. The package therefore separates
*characterisation* of the disagreement from *selection* of the correction, and
provides a registry of correction families rather than a fixed pipeline.

## An agentic architecture

This package is designed to be driven by an agent rather than executed as a
fixed sequence. The correction applied to a structure is not specified in
advance; it is derived at runtime by an agent that measures how the propagated
label fails, proposes candidate corrections consistent with that measurement,
tests each against the available annotations, and discards those that do not
improve on the alternatives. The deterministic components — geometry
validation, characterisation, fitting, scoring — are deliberately not agentic.
The agent supplies hypotheses; the code supplies verdicts.

### Why a fixed pipeline is insufficient

A pipeline must commit to a correction family at authoring time, and no family
is correct for more than a subset of structures. The two commonest failure modes
call for operations that are not merely different but opposed: over-coverage of
a low-intensity margin is removed by an intensity threshold, while
under-coverage of an interior is corrected by expansion followed by
contour-driven contraction. A pipeline encoding either will systematically
damage structures of the other kind, and the distinction is not predictable from
the identity of a structure. It is a property of how the deformation resolved
that structure in that cohort, observable only after propagation.

Compounding this, corrections that appear plausible fail for reasons specific to
the local image formation. Normalising a threshold to the local intensity mean
succeeds for thin low-intensity features and fails for thick high-intensity
ones, because within a large bright structure the local mean is the structure
itself. Expanding a label before thresholding recovers material only where the
label borders background; where it borders tissue of comparable intensity, the
expansion admits neighbouring structures and precision collapses. These are
determinable in advance only by reasoning about the intensity relationship
between a structure and its specific neighbours — which is what the agent does,
and what a fixed pipeline cannot.

### Why supervised learning is insufficient

The regime is one to five labelled examples per structure. Learned segmentation
models require orders of magnitude more, and the constraint is not incidental:
each annotation costs an expert hours to days, so the method must be effective
at very small sample size or it is not usable.

The problem is better posed as program synthesis from few examples. The search
space is compositional — thresholds, morphological operations, connectivity
constraints, contour evolution, and their combinations — rather than a fixed
parameter vector, and the object recovered must be an interpretable program with
very few free parameters, because that is what a handful of examples can
support. An agent conditioned on a measured error signature is a practical
proposal distribution over that space, and the exact verifier below makes wide
search affordable.

### What makes the loop safe to automate

Agentic search is normally limited by weak evaluation. Here it is not. The
objective is agreement with a manual annotation, which is deterministic, exact,
inexpensive to evaluate, and not susceptible to being satisfied by a plausible
argument. Candidate corrections can therefore be generated liberally and
rejected on evidence.

Three constraints are enforced in code rather than left to the agent's
discretion, because each is a way for a search with an exact verifier to
nonetheless produce a result that does not hold:

- **Annotation admissibility.** An annotation edited from an algorithm's output
  is recorded as such at intake and excluded from fitting. Without this the
  system optimises against its own prior output.
- **Parameter budget.** Fitting is refused when a family has more free
  parameters than there are independent annotations, since such a fit cannot be
  validated on held-out data.
- **Reachable ceiling.** Characterisation reports the highest recall a
  subtractive correction could attain, so search terminates against limits
  imposed by the registration rather than continuing indefinitely.

### The role of accumulated negative results

Most candidate corrections fail, and the reasons generalise across structures
more reliably than the successes do. A record of which families failed against
which error signatures, and why, is therefore a primary artefact of the
workflow rather than a by-product: it narrows the proposal space for each
subsequent structure. The registry records the conditions under which each
family applies for the same reason.

## Method

### Propagation

Registration proceeds in two stages. An affine transform is solved on image
intensity alone and without spatial masking, then supplied as the
initialisation for a deformable stage. Deformable optimisation does not itself
estimate an affine component, and restricting the affine stage to masked
regions discards most of the samples that constrain it.

Label volumes are resampled through the composed transform chain in a single
step using nearest-equivalent interpolation. Applying transforms sequentially
compounds interpolation error, and interpolating label values linearly produces
intensities corresponding to no label.

### Characterisation

For a propagated label and a manual reference on the same grid, the package
reports:

- intensity distributions of the matched, excess, and missed voxel populations,
  each normalised to the specimen's modal background;
- the separation between the excess and matched distributions, which determines
  whether an intensity criterion can distinguish them at all;
- the depth of excess voxels beneath the label surface, and the fraction lying
  within three voxels of it, distinguishing a surface shell from a
  displaced region;
- the connected-component decomposition of the excess, distinguishing one
  coherent region from scattered disagreement;
- the recall ceiling — the highest recall any subtractive correction can reach,
  since missed voxels lie outside the propagated label and cannot be recovered
  by removing material.

Reporting the ceiling explicitly prevents search continuing against a limit
imposed by the registration rather than by the correction.

### Correction

Corrections are parameterised in units derived from each specimen's own
intensity statistics rather than as absolute values. A threshold expressed as an
absolute intensity does not transfer between acquisitions; the same threshold
expressed as a position between that specimen's modal background and its
structure interior level accounts for variation in both offset and contrast.

Interior level is sampled at depth beneath the label surface so that boundary
transition voxels do not bias it, with a fallback for structures too thin to
have a core at that depth.

### Validation

Two constraints make a fitted correction trustworthy rather than merely
high-scoring.

**Annotation provenance.** An annotation produced by editing an algorithm's
output has its optimum at the parameter that produced it. Scoring against it
reports the fit back to itself. Such annotations still measure something
useful — the manual effort remaining after automatic correction — but must be
excluded when fitting or validating. The store records what each annotator
started from and derives admissibility from it, so this cannot be established
by inspecting filenames after the fact.

**Parameter budget.** A correction family with more free parameters than there
are independent annotations will fit them and generalise arbitrarily, with no
held-out score available to detect it. Fitting is refused in that case.

Where three or more independent annotations exist, leave-one-out refitting
reports the generalisation gap and the stability of each fitted parameter across
folds.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10 or later. Registration depends on ANTsPy; the
characterisation and correction modules require only NumPy and SciPy.

## Usage

### Admitting a manual annotation

Annotations enter the store through a validated intake. Name the file:

```
<specimen>__<structure>__<basis>__<annotator>__<YYYY-MM-DD>.seg.nrrd
```

`basis` records what the annotator started from and determines whether the
annotation may be used to fit:

| basis | meaning | admissible for fitting |
|---|---|---|
| `pushed` | the propagated label, unmodified | yes |
| `scratch` | traced from the image | yes |
| `refined` | an algorithm's output | no |

```bash
atlas-refine ingest \
    --reference data/specimens/100001/image.nii.gz \
    --frame specimen-100001 \
    --inbox inbox/
```

Intake verifies that the filename parses, that the mask lies on the reference
grid, that labels are binary and non-empty, and that the segmentation was
exported at full extent rather than on a cropped bounding box. Failures are
reported with a reason and nothing is written.

```bash
atlas-refine status       # holdings by structure, split by admissibility
atlas-refine algorithms   # available correction families and their parameters
```

### Propagating labels

```python
from atlas_refine import load_nifti
from atlas_refine.registration import MetricTerm, propagate_labels, register

atlas = load_nifti("atlas/template.nii.gz", frame="atlas")
specimen = load_nifti("data/specimens/100001/image.nii.gz", frame="specimen-100001")

registration = register(moving=atlas, fixed=specimen)
labels = propagate_labels(
    load_nifti("atlas/parcellation.nii.gz", frame="atlas"),
    reference=specimen,
    registration=registration,
)
```

Where manual annotations exist for a structure, supplying them as an additional
metric term constrains the deformation using anatomy that intensity alone
localises poorly:

```python
registration = register(
    moving=atlas,
    fixed=specimen,
    extra_metrics=[MetricTerm(fixed=specimen_annotation, moving=atlas_annotation, weight=1.0)],
)
```

A registration constrained this way should be evaluated through the whole
correction chain rather than on raw label agreement. A registration whose
residual error is of a form the correction removes may score lower before
correction and higher after.

### Fitting and validating a correction

```python
from atlas_refine.algorithms import REGISTRY
from atlas_refine.characterize import characterize
from atlas_refine.evaluate import fit, leave_one_out, measure_transfer
from atlas_refine.io import GroundTruthStore

store = GroundTruthStore("ground_truth")
admissible = store.fitting_set("isocortex")

print(characterize(specimen, labels, reference).summary())

algorithm = REGISTRY.create("contrast_threshold")
result = fit(algorithm, admissible[:1], loader)
transfer = measure_transfer(algorithm, result.params, admissible[:1], admissible[1:], loader)

print(transfer.generalises, transfer.held_out_mean_dice)
if len(admissible) >= 3:
    print(leave_one_out(algorithm, admissible, loader))
```

`loader` is a callable mapping a specimen id to `(image, propagated_label,
reference)`.

## Geometry and units

Voxel spacing is stored with an explicit unit, and every volume declares the
coordinate frame its axes are expressed in. Operations combining volumes assert
compatibility of shape, spacing, affine, and frame.

Header metadata in NIfTI and NRRD is frequently inconsistent with the stored
voxel sizes, and toolkits differ in whether they honour the declared unit code.
A volume whose spacing is implausible for its declared unit is rejected rather
than loaded, and `load_nifti` accepts an explicit spacing override for files
whose headers are known to be unreliable. Displacement fields and gradient
scales are expressed in the spacing supplied to the routine that produced them;
mixing conventions produces geometrically wrong output without raising.

## Layout

```
src/atlas_refine/
    io/              volume container with unit and frame assertions;
                     ground-truth store with declared provenance
    registration/    two-stage registration; single-resampling label propagation
    characterize/    error signature battery
    algorithms/      correction families and their registry
    evaluate/        fitting, transfer measurement, admissibility, leave-one-out
    cli.py
tests/
config/
docs/
```

## Extending

Correction families are the unit the agent selects among, so adding one widens
the search space rather than changing the workflow. A new family subclasses
`RefinementAlgorithm`, declares its free parameters and their search spaces, and
registers itself:

```python
from atlas_refine.algorithms.base import REGISTRY, Parameter, RefinementAlgorithm

@REGISTRY.register
class MyCorrection(RefinementAlgorithm):
    name = "my_correction"
    description = "One line describing when this family applies."

    @property
    def parameters(self):
        return (Parameter("radius", [1, 2, 3], "Structuring element radius."),)

    def apply(self, image, label, **params):
        ...
        return label.with_data(result)
```

It is then available to `fit`, `measure_transfer`, and `leave_one_out` without
further changes, and its parameter count is checked against the available
annotations automatically. The `description` is what an agent reads when
deciding whether the family suits a measured error signature, so it should state
the conditions under which the family applies rather than what it computes.

## License

MIT
