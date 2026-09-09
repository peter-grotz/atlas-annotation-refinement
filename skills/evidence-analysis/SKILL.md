---
name: evidence-analysis
description: Judge what a fitted result actually supports - distinguishing in-sample from held-out performance, detecting when ground truth was derived from the output being tested, and reporting correlations honestly at small sample sizes. Use when reporting a fit, comparing corrections, relating a parameter to image properties, or deciding whether a result is strong enough to act on.
---

# What a result supports

Fitting is straightforward. Knowing what the number means is where results go
wrong, and every failure mode below produces a plausible number rather than an
error.

## In-sample is not accuracy

A parameter fitted on an annotation and scored against that same annotation
reports the fit back to itself. It is a lower bound on how well the family
*can* do, not an estimate of how well it *will* do.

Report it as in-sample and say what it was fitted on. The number that estimates
performance is the one measured on annotations excluded from the fit.

Held-out performance **matching or exceeding** the fitted score is the signature
of a parameter that is not overfit. A held-out score well below the fitted one
means the fit captured the examples rather than the phenomenon.

## Ground truth derived from the output under test

An annotation produced by editing an algorithm's output has its optimum at the
parameter that produced it. Scoring it against that algorithm manufactures
evidence.

Such an annotation still measures something useful — the manual effort remaining
after automatic correction — but it cannot fit or validate the parameter that
produced its starting point. The store records this as `basis`, and
`fitting_set()` returns only independent annotations. Query it rather than
assuming.

This is train-test contamination arriving through a person helpfully starting
from the best available draft, which makes it easy to miss.

## Parameter budget

A family with more free parameters than there are independent annotations will
fit them and generalise arbitrarily, with no held-out score available to detect
it. Fitting is refused rather than reported with a caveat.

Where three or more independent annotations exist, `leave_one_out` reports the
gap between in-sample and held-out performance, and the stability of each fitted
parameter across folds. An unstable parameter is a parameter the data does not
determine.

## Correlations at small n

```python
from atlas_refine.analysis import associate_with_parameter
for association in associate_with_parameter(profiles, per_specimen_parameter):
    print(association.describe())
```

At two or three specimens almost any feature will order them plausibly by
chance — with two points, half of all monotonic features will. A correlation
below the interpretability threshold is a candidate to test against the next
annotation, never a finding.

Report `n` alongside every correlation. A coefficient without a sample size
attached is not a result.

## Proxies

A metric that is not the objective can point the wrong way. Inter-specimen
volume consistency, for instance, appears to measure whether a correction
generalises — but propagated labels are artificially consistent, being one atlas
label warped several similar ways, so a correction that improves accuracy can
*reduce* consistency.

Prefer the objective. Where a proxy is used, state what it is a proxy for and
how it can mislead.

## Judging one correction against another

Compare families fitted on the same annotations, so the comparison reflects the
family rather than differing tuning effort.

Evaluate end to end. A component can score worse in isolation and better in the
whole chain, if its error is of a form a later stage removes — so a component
promoted on its own score can be the wrong choice.

## Guidelines

- State the evidence class on every reported figure: in-sample, held-out, or
  derived.
- Prefer the simplest family within noise of the best. Flat objectives transfer;
  sharp optima are usually fitted to their examples.
- A negative result with a mechanism is more valuable than a positive result
  without one, and generalises further.
