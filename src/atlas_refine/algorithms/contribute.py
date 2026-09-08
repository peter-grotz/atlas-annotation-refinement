"""Framework for admitting newly authored correction families.

The built-in families cover the failure modes that recur, but a structure may
disagree with its propagated label in a way none of them addresses. An agent
working the loop should be able to author a new family and install it durably
rather than applying a one-off script whose reasoning is lost.

Admission is deliberately permissive. A family is admitted on the evidence it
actually has, and the recorded track states what its output may be used for. A
family fitted on a single annotation is admitted as a starting point, because
the alternative is annotating the remaining cohort from scratch and an annotator
reviews every voxel of the result anyway. It is promoted when it survives
annotations excluded from the fit.

What is enforced is correctness rather than quality: a family must conform to
the interface and behave sanely under adversarial inputs, since those failures
produce plausible output rather than an error. Everything else is recorded, not
refused.

Every attempt is written to the trial log, whether admitted or rejected. The
failures are the more valuable half, because the reason a family fails on one
error signature usually holds for every structure presenting a similar one.

Workflow::

    scaffold("my_family")                    # writes a template
    # author the implementation and RATIONALE in that file
    contribute("path/to/my_family.py", specimens, loader, structure="...", log=log)

On admission the module is installed under ``algorithms/contributed/``, where it
is discovered automatically; a rationale record is written to
``docs/algorithms/``; a regression test is generated; and a trial is appended.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..evaluate.search import Loader, check_admissible, fit, measure_transfer, scores
from ..experiments.trials import TrialLog
from ..io.volumes import Volume
from .base import REGISTRY, RefinementAlgorithm

PACKAGE_ROOT = Path(__file__).resolve().parent
CONTRIBUTED = PACKAGE_ROOT / "contributed"
REPO_ROOT = PACKAGE_ROOT.parents[2]

#: Admission tracks. A family is admitted on the evidence it actually has, and
#: the track states what its output may be used for rather than gating on a
#: single universal bar.
#:
#:   starting_point  Improves on the annotations available, including a single
#:                   one. Output seeds manual refinement, where an annotator
#:                   reviews every voxel, so the risk of an imperfect fit is
#:                   bounded and the alternative is annotating from scratch.
#:   validated       Holds on annotations excluded from the fit. Output may be
#:                   consumed as data without review.
#:
#: A family enters as `starting_point` and is promoted when the evidence
#: arrives; promotion records a new trial rather than rewriting the old one.
TRACKS = ("starting_point", "validated")

#: Improvement over the best incumbent required for the `starting_point` track.
#: Set at zero deliberately: a family that merely matches an incumbent on the
#: cohort mean may still be the only one that works on an atypical specimen, and
#: discarding it forecloses that. Ranking happens at selection time, not here.
MIN_IMPROVEMENT = 0.0


class ContributionRejected(ValueError):
    """Raised when a candidate family does not qualify for admission."""


#: Prompt text written into a scaffolded module. Used both to build the template
#: and to detect a rationale that was never written, so the two cannot drift.
PLACEHOLDERS = {
    "target_signature": (
        "Describe the error signature this addresses, using the quantities "
        "reported by characterize(): over- or under-coverage, intensity "
        "separation, error depth and topology."
    ),
    "why_incumbents_insufficient": (
        "State which existing families were considered and the specific reason "
        "each fails on this signature."
    ),
    "limitations": (
        "State when this family should not be selected, including any "
        "structure geometry or contrast regime where it degrades the result."
    ),
}

#: Word overlap with the placeholder above which a field counts as unedited.
_PLACEHOLDER_OVERLAP = 0.8


def _looks_like_placeholder(text: str, placeholder: str) -> bool:
    """Whether a field is the scaffold prompt rather than a written rationale.

    Compares word overlap rather than requiring an exact match, so that
    reordering or lightly editing the prompt does not defeat the check.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    prompt = set(re.findall(r"[a-z]+", placeholder.lower()))
    if not words:
        return True
    return len(words & prompt) / len(words) >= _PLACEHOLDER_OVERLAP


@dataclass(frozen=True)
class Rationale:
    """Why a new family is needed, and when it should not be used.

    Every field is required. A family whose applicability is not stated cannot
    be matched to an error signature later, which defeats the purpose of
    installing it.
    """

    #: The error signature this family targets, in the terms reported by
    #: :func:`atlas_refine.characterize.characterize`.
    target_signature: str
    #: Why each relevant incumbent family is inadequate for that signature.
    why_incumbents_insufficient: str
    #: Conditions under which this family should not be selected.
    limitations: str
    #: Optional literature or method references.
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, placeholder in PLACEHOLDERS.items():
            value = getattr(self, name)
            if not isinstance(value, str) or len(value.strip()) < 40:
                raise ContributionRejected(
                    f"Rationale.{name} must be a substantive statement "
                    f"(at least 40 characters); got {value!r}"
                )
            if _looks_like_placeholder(value, placeholder):
                raise ContributionRejected(
                    f"Rationale.{name} is still the scaffold prompt. A length "
                    f"check cannot tell a considered rationale from boilerplate, "
                    f"so the prompt itself is refused: state what this family "
                    f"targets, why the incumbents fail on it, and when not to "
                    f"use it."
                )


@dataclass(frozen=True)
class ValidationReport:
    checks: dict[str, bool]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class BenchmarkReport:
    candidate: str
    candidate_params: dict[str, float]
    candidate_mean_dice: float
    incumbent_mean_dice: dict[str, float]
    best_incumbent: str | None
    improvement: float
    specimens: list[str]

    @property
    def qualifies(self) -> bool:
        return self.improvement >= MIN_IMPROVEMENT


@dataclass(frozen=True)
class Contribution:
    name: str
    track: str
    installed_at: str
    contributed_utc: str
    n_parameters: int
    rationale: dict
    validation: dict
    benchmark: dict
    trial_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# scaffolding


TEMPLATE = '''"""{name}: one-line statement of what this family corrects.

Authored for the loop in `atlas_refine`. Replace this docstring with a
description of the mechanism and the reasoning behind it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from atlas_refine.algorithms.base import REGISTRY, Parameter, RefinementAlgorithm
from atlas_refine.algorithms.contribute import Rationale
from atlas_refine.io.volumes import Volume, background_level

#: Required for admission. Replace every field: the prompts below are refused,
#: because a length check cannot distinguish boilerplate from a real rationale.
RATIONALE = Rationale(
    target_signature="{ph_target}",
    why_incumbents_insufficient="{ph_why}",
    limitations="{ph_limits}",
)


@REGISTRY.register
class {cls}(RefinementAlgorithm):
    """One-line summary used when matching a family to a signature."""

    name = "{name}"
    description = "Conditions under which this family applies."

    @property
    def parameters(self) -> Sequence[Parameter]:
        # Keep the count at or below the number of independent annotations
        # available. Express values relative to per-specimen intensity
        # statistics rather than as absolute intensities.
        return (
            Parameter("example", [0.1, 0.2, 0.3], "What this controls."),
        )

    def apply(self, image: Volume, label: Volume, **params: float) -> Volume:
        image.assert_compatible(label)
        mask = label.data > 0
        # Implement the correction. Return a binary mask on the input grid.
        result = mask
        return label.with_data(result.astype("uint8"))
'''


def scaffold(name: str, directory: str | Path | None = None) -> Path:
    """Write a template module for a new family.

    Args:
        name: Registry name, lowercase with underscores.
        directory: Where to write. Defaults to the working directory.
    """
    if not name.replace("_", "").isalnum() or name != name.lower():
        raise ContributionRejected(
            f"name must be lowercase alphanumeric with underscores; got {name!r}"
        )
    if name in REGISTRY.names():
        raise ContributionRejected(f"'{name}' is already registered")
    cls = "".join(part.capitalize() for part in name.split("_"))
    path = Path(directory or ".") / f"{name}.py"
    if path.exists():
        raise ContributionRejected(f"{path} already exists")
    path.write_text(
        TEMPLATE.format(
            name=name,
            cls=cls,
            ph_target=PLACEHOLDERS["target_signature"],
            ph_why=PLACEHOLDERS["why_incumbents_insufficient"],
            ph_limits=PLACEHOLDERS["limitations"],
        )
    )
    return path


# --------------------------------------------------------------------------
# validation


def validate(
    algorithm: RefinementAlgorithm,
    image: Volume,
    label: Volume,
    *,
    source: str | None = None,
) -> ValidationReport:
    """Check interface conformance and behaviour under adversarial inputs.

    These are the properties whose violation produces plausible but wrong
    output rather than an error.

    Args:
        algorithm: Candidate to check.
        image: Representative specimen intensities.
        label: Representative propagated label on the same grid.
        source: Module text, for the checks that inspect the implementation.
            Supply this for a candidate loaded from a file path: such a class
            has no resolvable source file and :func:`inspect.getsource` raises.
    """
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def record(key: str, ok: bool, message: str) -> None:
        checks[key] = ok
        if not ok:
            failures.append(f"{key}: {message}")

    record("has_name", bool(algorithm.name) and algorithm.name != "unnamed",
           "algorithm must set a registry name")
    record("has_description", len(algorithm.description or "") >= 20,
           "description must state when the family applies")
    record("declares_parameters", algorithm.n_parameters >= 0,
           "parameters property must be implemented")

    params = algorithm.defaults()
    original = label.data.copy()

    try:
        result = algorithm.apply(image, label, **params)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        record("runs", False, f"raised {type(exc).__name__}: {exc}")
        return ValidationReport(checks, failures)
    record("runs", True, "")

    record("returns_volume", isinstance(result, Volume), "apply must return a Volume")
    if isinstance(result, Volume):
        record("preserves_grid", result.shape == label.shape,
               f"output shape {result.shape} != input {label.shape}")
        record("preserves_frame", result.frame == label.frame,
               "output must remain in the input's coordinate frame")
        record("binary_output", set(np.unique(result.data).tolist()) <= {0, 1},
               "output must be binary")

    record("does_not_mutate_input", np.array_equal(label.data, original),
           "apply must not modify its inputs")

    try:
        again = algorithm.apply(image, label, **params)
        record("deterministic", np.array_equal(result.data, again.data),
               "repeated application with identical inputs must agree")
    except Exception as exc:  # noqa: BLE001
        record("deterministic", False, f"second call raised {exc}")

    empty = label.with_data(np.zeros_like(label.data))
    try:
        out = algorithm.apply(image, empty, **params)
        record("handles_empty_label", int((out.data > 0).sum()) == 0,
               "an empty label must yield an empty result")
    except Exception as exc:  # noqa: BLE001
        record("handles_empty_label", False, f"raised on an empty label: {exc}")

    thin = np.zeros_like(label.data)
    thin[thin.shape[0] // 2, :, :] = 1
    try:
        algorithm.apply(image, label.with_data(thin), **params)
        record("handles_thin_label", True, "")
    except Exception as exc:  # noqa: BLE001
        record("handles_thin_label", False,
               f"raised on a label with no interior core: {exc}")

    grid = list(algorithm.grid())
    record("grid_non_empty", bool(grid), "parameter grid must contain at least one point")
    if grid:
        try:
            for point in (grid[0], grid[-1]):
                algorithm.apply(image, label, **point)
            record("grid_endpoints_run", True, "")
        except Exception as exc:  # noqa: BLE001
            record("grid_endpoints_run", False, f"raised at a grid endpoint: {exc}")

    if source is None:
        try:
            source = inspect.getsource(type(algorithm))
        except (TypeError, OSError):
            source = None
    if source is None:
        checks["no_absolute_intensity_literals"] = True  # not inspectable, not a failure
    else:
        record(
            "no_absolute_intensity_literals",
            not re.search(r">\s*\d{4,}", source),
            "thresholds must be relative to per-specimen intensity statistics, "
            "not absolute values, or they will not transfer between acquisitions",
        )

    return ValidationReport(checks, failures)


# --------------------------------------------------------------------------
# benchmarking


def benchmark(
    candidate: RefinementAlgorithm,
    specimens: Sequence[str],
    loader: Loader,
    *,
    incumbents: Sequence[str] | None = None,
) -> BenchmarkReport:
    """Fit the candidate and every incumbent, then compare best against best.

    Both sides are fitted on the same annotations, so the comparison reflects
    the family rather than a difference in tuning effort.
    """
    check_admissible(candidate, specimens)
    candidate_fit = fit(candidate, specimens, loader)

    names = list(incumbents) if incumbents is not None else REGISTRY.names()
    incumbent_scores: dict[str, float] = {}
    for name in names:
        if name == candidate.name:
            continue
        algorithm = REGISTRY.create(name)
        if algorithm.n_parameters > len(specimens):
            continue  # not admissible at this annotation count
        incumbent_scores[name] = fit(algorithm, specimens, loader).mean_dice

    best = max(incumbent_scores, key=incumbent_scores.get) if incumbent_scores else None
    baseline = incumbent_scores[best] if best else 0.0
    return BenchmarkReport(
        candidate=candidate.name,
        candidate_params=candidate_fit.params,
        candidate_mean_dice=candidate_fit.mean_dice,
        incumbent_mean_dice=incumbent_scores,
        best_incumbent=best,
        improvement=round(candidate_fit.mean_dice - baseline, 6),
        specimens=list(specimens),
    )


# --------------------------------------------------------------------------
# admission


def load_candidate(module_path: str | Path) -> tuple[RefinementAlgorithm, Rationale]:
    """Import a candidate module and return its algorithm and rationale."""
    module_path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ContributionRejected(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rationale = getattr(module, "RATIONALE", None)
    if not isinstance(rationale, Rationale):
        raise ContributionRejected(
            f"{module_path.name} must define RATIONALE as a Rationale instance"
        )

    classes = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, RefinementAlgorithm)
        and obj is not RefinementAlgorithm
        and obj.__module__ == module.__name__
    ]
    if len(classes) != 1:
        raise ContributionRejected(
            f"{module_path.name} must define exactly one RefinementAlgorithm subclass, "
            f"found {len(classes)}"
        )
    return classes[0](), rationale


def contribute(
    module_path: str | Path,
    specimens: Sequence[str],
    loader: Loader,
    *,
    structure: str = "unspecified",
    held_out: Sequence[str] = (),
    incumbents: Sequence[str] | None = None,
    log: TrialLog | None = None,
    install: bool = True,
    notes: str = "",
) -> Contribution:
    """Validate, benchmark, and admit a candidate family.

    Args:
        module_path: File authored from :func:`scaffold`.
        specimens: Independent annotations to fit on.
        loader: Supplies ``(image, label, reference)`` per specimen.
        structure: Structure the family was authored against, recorded on the
            trial so later retrieval can be filtered.
        held_out: Annotations excluded from the fit. Supplying any promotes the
            family to the ``validated`` track if it holds on them.
        incumbents: Families to compare against. All registered by default.
        log: Trial log to append to. Both admission and rejection are recorded.
        install: Copy the module into the package and write the records.
        notes: The authoring agent's reasoning, recorded on the trial.

    Raises:
        ContributionRejected: If the module cannot be loaded or fails the
            correctness checks. Nothing is installed, but the attempt is still
            logged when a log is supplied.
    """
    algorithm, rationale = load_candidate(module_path)

    image, label, _ = loader(specimens[0])
    report = validate(algorithm, image, label, source=Path(module_path).read_text())
    if not report.passed:
        if log is not None:
            log.record(
                structure=structure, specimens=list(specimens), family=algorithm.name,
                params={}, signature={}, scores={}, outcome="rejected",
                reason="Failed correctness validation: " + "; ".join(report.failures),
                notes=notes,
            )
        raise ContributionRejected("validation failed:\n  " + "\n  ".join(report.failures))

    marks = benchmark(algorithm, specimens, loader, incumbents=incumbents)

    track, evidence = "starting_point", "in-sample"
    if held_out:
        transfer = measure_transfer(algorithm, marks.candidate_params, specimens, held_out, loader)
        evidence = "held-out"
        if transfer.generalises:
            track = "validated"

    trial_id = ""
    if log is not None:
        trial = log.record(
            structure=structure,
            specimens=list(specimens),
            family=algorithm.name,
            params=marks.candidate_params,
            signature={},
            scores={"dice": marks.candidate_mean_dice, "improvement": marks.improvement},
            outcome="admitted",
            reason=rationale.target_signature,
            evidence=evidence,
            notes=notes,
        )
        trial_id = trial.trial_id

    record = Contribution(
        name=algorithm.name,
        track=track,
        installed_at=str(CONTRIBUTED / f"{algorithm.name}.py") if install else "",
        contributed_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_parameters=algorithm.n_parameters,
        rationale=asdict(rationale),
        validation={"checks": report.checks, "failures": report.failures},
        benchmark=asdict(marks),
        trial_id=trial_id,
    )
    if install:
        _install(Path(module_path), algorithm, record)
    return record


def _install(module_path: Path, algorithm: RefinementAlgorithm, record: Contribution) -> None:
    CONTRIBUTED.mkdir(parents=True, exist_ok=True)
    destination = CONTRIBUTED / f"{algorithm.name}.py"
    if destination.exists():
        raise ContributionRejected(f"{destination} already exists")
    shutil.copy2(module_path, destination)

    docs = REPO_ROOT / "docs" / "algorithms"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / f"{algorithm.name}.md").write_text(_render_record(algorithm, record))
    (docs / f"{algorithm.name}.json").write_text(json.dumps(record.to_dict(), indent=2))

    tests = REPO_ROOT / "tests" / "contributed"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / f"test_{algorithm.name}.py").write_text(_render_test(algorithm, record))


def _render_record(algorithm: RefinementAlgorithm, record: Contribution) -> str:
    marks = record.benchmark
    rationale = record.rationale
    incumbents = "\n".join(
        f"| `{name}` | {score:.4f} |" for name, score in sorted(marks["incumbent_mean_dice"].items())
    ) or "| _none admissible at this annotation count_ | |"
    references = "\n".join(f"- {r}" for r in rationale["references"]) or "_none_"
    return f"""# `{algorithm.name}`

{algorithm.description}

Contributed {record.contributed_utc} · track **{record.track}** · {record.n_parameters} free parameter(s)

## Target signature

{rationale['target_signature']}

## Why existing families were insufficient

{rationale['why_incumbents_insufficient']}

## Limitations

{rationale['limitations']}

## Evidence at admission

Fitted on {len(marks['specimens'])} independent annotation(s).

| family | mean Dice |
|---|---|
| **`{marks['candidate']}`** | **{marks['candidate_mean_dice']:.4f}** |
{incumbents}

Improvement over the best incumbent (`{marks['best_incumbent']}`):
**{marks['improvement']:+.4f}**

Fitted parameters: `{marks['candidate_params']}`

These figures reflect the annotations available at admission. They are an
in-sample fit unless the specimen count exceeded the parameter count, and are
superseded by any later transfer or leave-one-out measurement.

## References

{references}
"""


def _render_test(algorithm: RefinementAlgorithm, record: Contribution) -> str:
    return f'''"""Regression test for the contributed family `{algorithm.name}`.

Generated at admission. Pins the interface and the reported parameter count so
that later edits cannot silently change either.
"""

import numpy as np
import pytest

from atlas_refine.algorithms import REGISTRY
from atlas_refine.io.volumes import Volume

NAME = "{algorithm.name}"
N_PARAMETERS = {record.n_parameters}
ADMITTED_MEAN_DICE = {record.benchmark["candidate_mean_dice"]!r}


def make_volume(shape=(12, 12, 12), value=1000.0, frame="test"):
    spacing = (20.0, 20.0, 20.0)
    return Volume(
        data=np.full(shape, value, dtype="float32"),
        spacing=spacing,
        unit="um",
        affine=np.diag([-spacing[0], -spacing[1], spacing[2], 1.0]),
        frame=frame,
    )


def test_registered():
    assert NAME in REGISTRY.names()


def test_parameter_count_unchanged():
    assert REGISTRY.create(NAME).n_parameters == N_PARAMETERS


def test_returns_binary_mask_on_input_grid():
    algorithm = REGISTRY.create(NAME)
    image = make_volume()
    label = image.with_data(np.ones(image.shape, dtype="uint8"))
    result = algorithm.apply(image, label, **algorithm.defaults())
    assert result.shape == label.shape
    assert result.frame == label.frame
    assert set(np.unique(result.data).tolist()) <= {{0, 1}}


def test_empty_label_yields_empty_result():
    algorithm = REGISTRY.create(NAME)
    image = make_volume()
    empty = image.with_data(np.zeros(image.shape, dtype="uint8"))
    assert int((algorithm.apply(image, empty, **algorithm.defaults()).data > 0).sum()) == 0
'''
