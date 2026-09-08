# Workflow

## Stages

1. **Propagate.** Register each specimen to the atlas and reverse-transform the
   parcellation. Every specimen has a complete annotation at no manual cost.
2. **Annotate one.** An expert corrects the propagated label for a single
   specimen. Admit it with `basis=pushed`.
3. **Characterise.** Measure the disagreement between the propagated label and
   the annotation. The signature determines which correction family applies.
4. **Fit.** Search the family's parameter grid against the annotation. The
   parameter budget is checked first.
5. **Apply.** Run the fitted correction across the cohort. Compare the
   signature of the removed or added material per specimen; a correction
   behaving consistently removes material with the same signature everywhere.
6. **Validate.** Annotate a second specimen and score the existing parameters
   against it without refitting. Held-out performance matching or exceeding
   in-sample performance indicates the parameter is not tuned to the first
   example.
7. **Reintroduce.** Once a structure is annotated across the cohort, supply
   those annotations as registration metric terms and repropagate. This improves
   placement for every structure simultaneously rather than correcting each
   afterwards.

## What each annotation count permits

| independent annotations | permitted | not supported |
|---|---|---|
| 0 | propagation; unsupervised cohort description; selecting which specimen to annotate first | any accuracy claim |
| 1 | fitting a single-parameter family; cohort-wide application with consistency flags | reporting the fitted score as accuracy, since it is in-sample |
| 2 | measuring transfer on held-out data; fitting one normalisation | pooling a derived annotation; concluding that a cohort feature explains a parameter |
| 3–5 | leave-one-out; per-specimen conditioning where a measured feature predicts the parameter | more free parameters than annotations |
| cohort-wide | registration metric terms; repropagation | promoting a registration on raw label agreement rather than end-to-end |

## Choosing which specimen to annotate next

Annotator time is the binding constraint, so the choice matters more than any
parameter. Prefer the specimen least similar to those already annotated,
measured on unsupervised image properties — contrast, low-frequency intensity
variation, tissue-class separability, noise, acquisition extent. A specimen that
is out of distribution on a property the correction depends on is where the
fitted parameter is least likely to hold, and therefore where an annotation is
most informative.

## Interpreting the signature

| signature | indicated family |
|---|---|
| over-coverage; excess forms one surface shell; excess and structure intensities barely overlap | intensity threshold |
| over-coverage; excess bounded by higher- and lower-intensity neighbours | band threshold |
| under-coverage; structure interior separable but partly below a single threshold | hysteresis |
| over-coverage; excess a shell of roughly constant thickness with no intensity separation | morphological |
| right volume, both error populations large (`displacement`) | registration-limited; the label needs moving, not resizing |
| disagreement with no intensity separation at either boundary | registration-limited; correction cannot help |

The last row is the case to detect early. Where a boundary borders material of
similar intensity, no intensity criterion can place it and the residual error is
a property of the deformation rather than of the correction.
