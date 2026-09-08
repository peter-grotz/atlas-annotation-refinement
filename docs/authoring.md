# Authoring correction families

Where a new family goes, when to write one, and how it becomes visible to
whoever works the loop next.

## Where

```
src/atlas_refine/algorithms/
    base.py                 interface and registry
    intensity.py            built-in families
    contribute.py           scaffolding, validation, admission
    contributed/
        __init__.py         imports every module here at load time
        <family>.py         one family per module, RATIONALE required

experiments/
    trials.jsonl            append-only record of every attempt

docs/algorithms/
    <family>.md             rationale and evidence at admission
    <family>.json           the same, machine-readable
    INDEX.md                generated: families, and the dead ends

tests/contributed/
    test_<family>.py        generated regression test
```

One family per module. Modules in `contributed/` are imported automatically, so
a family becomes available to `fit`, `measure_transfer`, and `leave_one_out`
with no registration step beyond being installed there.

Draft work belongs anywhere outside `contributed/` — the working directory is
fine. A draft is not importable and not discoverable until it is admitted, which
keeps unfinished work out of the search space without discarding it.

## When

Try the registry first. It is cheaper than authoring, and the trial log will
often say directly whether an existing family suits the measured signature.

```python
from atlas_refine.characterize import characterize
from atlas_refine.experiments import TrialLog

signature = characterize(image, label, reference)
log = TrialLog("experiments/trials.jsonl")

for family, record in log.families_tried(signature.to_dict()).items():
    print(family, record["best_dice"], record["reasons"])
```

Author a new family when the log shows the existing ones were tried against a
comparable signature and fell short, or when the signature is unlike anything
recorded. Both are visible from the query above; neither requires guessing.

Do not author one to obtain a small margin over an incumbent on the same
signature. That widens the search space for every later structure and the gain
will not usually survive a second annotation.

## How

```python
from atlas_refine.algorithms import contribute, scaffold

scaffold("my_family")                    # writes a template with a RATIONALE stub
# author the implementation and the rationale in that file

record = contribute(
    "my_family.py",
    specimens=store.fitting_set("my_structure"),
    loader=loader,
    structure="my_structure",
    log=log,
    notes="Why this mechanism was expected to suit the measured signature.",
)
```

Admission is permissive. What is enforced is correctness, not quality: the
family must return a binary mask on the input grid and frame, leave its
arguments unmodified, produce the same result twice, return empty for an empty
label, and tolerate a structure too thin to have an interior core. Those
failures produce plausible output rather than an error, which is why they are
checked rather than trusted.

Everything else is recorded rather than refused.

### Tracks

The recorded track states what the output may be used for, rather than gating on
a single universal bar.

| track | admitted when | output may be used for |
|---|---|---|
| `starting_point` | it improves on the annotations available, including a single one | seeding manual refinement, where an annotator reviews every voxel |
| `validated` | it holds on annotations excluded from the fit | labels consumed as data without review |

A family fitted on one annotation enters as `starting_point`. That is the
intended path: the alternative is annotating the rest of the cohort from
scratch, and a draft that is imperfect still leaves less to fix. Pass `held_out`
once further annotations exist and the family is promoted if it holds.

## How it becomes visible to whoever comes next

Every attempt is appended to `experiments/trials.jsonl` — admitted, rejected,
superseded, or merely explored. Failures are the more valuable half. A family
that fails on one error signature will usually fail on any comparable one, and
that is knowledge no amount of re-derivation improves on.

Trials are indexed by **measured error signature**, not by structure name. Two
anatomically unrelated structures with similar signatures call for similar
corrections, and the same structure in a different cohort may present a
signature never seen before. Retrieval is therefore by similarity:

```python
log.similar(signature.to_dict(), limit=10)          # nearest prior attempts
log.families_tried(signature.to_dict())             # per family: outcomes, reasons
```

The `reason` recorded on a rejection is what makes this useful, so it should say
why the family was unsuitable for that kind of failure rather than restating the
score. Compare:

> Scored 0.63, below the incumbent.

against

> Inside a thick high-intensity structure the local mean is the structure
> itself, so the ratio flattens and the criterion loses discrimination.

The first prevents nothing. The second rules out an entire family for every
future structure presenting that signature.

`docs/algorithms/INDEX.md` renders the same record for a human reader, including
a *Recorded dead ends* section listing rejections with their reasons.

## Superseding

Families are not edited in place once admitted, since the generated test and the
rationale record both refer to the version that was measured. Author a new
family, and record the relationship:

```python
log.record(..., outcome="superseded", supersedes="old_family_name", ...)
```

The log is append-only, so what was believed and when remains readable. That
matters when a later result contradicts an earlier one and the question is which
evidence each rested on.
