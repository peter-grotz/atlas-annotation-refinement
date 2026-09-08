"""Parameter fitting, transfer measurement, and admissibility checks.

Fitting an algorithm's parameters against manual annotations is straightforward
supervised optimisation. Two constraints make the result trustworthy rather than
merely high.

Only independent annotations may enter a fit. An annotation produced by editing
an algorithm's output has its optimum at the parameter that produced it; scoring
against it reports the fit back to itself.

The number of free parameters must not exceed the number of independent
annotations. A family with more degrees of freedom than examples will fit them
and generalise arbitrarily, and no held-out score is available to detect it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..algorithms.base import RefinementAlgorithm
from ..io.volumes import Volume


class AdmissibilityError(ValueError):
    """Raised when a fit is not supportable by the available annotations."""


def dice(predicted: np.ndarray, truth: np.ndarray) -> float:
    p, t = predicted > 0, truth > 0
    total = int(p.sum()) + int(t.sum())
    return 2 * int((p & t).sum()) / total if total else 0.0


def scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p, t = predicted > 0, truth > 0
    intersection = int((p & t).sum())
    return {
        "dice": round(dice(p, t), 6),
        "recall": round(intersection / int(t.sum()), 6) if t.any() else 0.0,
        "precision": round(intersection / int(p.sum()), 6) if p.any() else 0.0,
    }


@dataclass(frozen=True)
class Trial:
    params: dict[str, float]
    per_specimen: dict[str, dict[str, float]]
    mean_dice: float
    min_dice: float


@dataclass(frozen=True)
class FitResult:
    algorithm: str
    params: dict[str, float]
    fitted_on: list[str]
    mean_dice: float
    min_dice: float
    n_parameters: int
    trials: list[Trial] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["trials"] = [asdict(t) for t in self.trials]
        return out

    def sensitivity(self) -> float:
        """Spread of mean Dice across the searched grid.

        A small value indicates the objective is flat and the fitted value is
        not finely tuned, which is favourable for transfer. A large value
        indicates the result depends sharply on the parameter.
        """
        if len(self.trials) < 2:
            return float("nan")
        values = [t.mean_dice for t in self.trials]
        return float(max(values) - min(values))


@dataclass(frozen=True)
class TransferResult:
    """Performance on annotations excluded from the fit."""

    params: dict[str, float]
    fitted_on: list[str]
    held_out: list[str]
    fitted_mean_dice: float
    held_out_mean_dice: float
    per_specimen: dict[str, dict[str, float]]

    @property
    def generalises(self) -> bool:
        """Whether held-out performance is at least as good as in-sample.

        Held-out performance matching or exceeding the fitted score indicates
        the parameter is not tuned to the fitting examples.
        """
        return self.held_out_mean_dice >= self.fitted_mean_dice

    def to_dict(self) -> dict:
        out = asdict(self)
        out["generalises"] = self.generalises
        return out


Loader = Callable[[str], tuple[Volume, Volume, Volume]]
"""Returns ``(image, propagated_label, reference)`` for a specimen id."""


def check_admissible(algorithm: RefinementAlgorithm, specimens: Sequence[str]) -> None:
    """Raise if the algorithm cannot be fitted on this many annotations."""
    if not specimens:
        raise AdmissibilityError("no independent annotations available to fit against")
    if algorithm.n_parameters > len(specimens):
        raise AdmissibilityError(
            f"'{algorithm.name}' has {algorithm.n_parameters} free parameters but only "
            f"{len(specimens)} independent annotation(s) are available. Choose a family "
            f"with at most {len(specimens)} parameter(s), or add annotations."
        )


def fit(
    algorithm: RefinementAlgorithm,
    specimens: Sequence[str],
    loader: Loader,
    *,
    objective: str = "mean",
) -> FitResult:
    """Search the algorithm's parameter grid against the given annotations.

    Args:
        algorithm: Algorithm to fit.
        specimens: Specimen ids, all of which must be independent annotations.
        loader: Supplies volumes for a specimen id.
        objective: ``"mean"`` maximises mean Dice; ``"min"`` maximises the worst
            specimen, which is preferable when uniform behaviour matters more
            than average performance.
    """
    check_admissible(algorithm, specimens)
    cached = {s: loader(s) for s in specimens}

    trials: list[Trial] = []
    for params in algorithm.grid():
        per_specimen = {}
        for specimen, (image, label, reference) in cached.items():
            refined = algorithm.apply(image, label, **params)
            per_specimen[specimen] = scores(refined.data, reference.data)
        values = [v["dice"] for v in per_specimen.values()]
        trials.append(Trial(params, per_specimen, round(float(np.mean(values)), 6),
                            round(float(np.min(values)), 6)))

    key = (lambda t: t.mean_dice) if objective == "mean" else (lambda t: t.min_dice)
    best = max(trials, key=key)
    return FitResult(
        algorithm=algorithm.name,
        params=best.params,
        fitted_on=list(specimens),
        mean_dice=best.mean_dice,
        min_dice=best.min_dice,
        n_parameters=algorithm.n_parameters,
        trials=trials,
    )


def measure_transfer(
    algorithm: RefinementAlgorithm,
    params: Mapping[str, float],
    fitted_on: Sequence[str],
    held_out: Sequence[str],
    loader: Loader,
) -> TransferResult:
    """Score fixed parameters on annotations that were not fitted against."""
    if set(fitted_on) & set(held_out):
        raise AdmissibilityError(
            f"specimens appear in both the fitting and held-out sets: "
            f"{sorted(set(fitted_on) & set(held_out))}"
        )
    per_specimen = {}
    for specimen in list(fitted_on) + list(held_out):
        image, label, reference = loader(specimen)
        refined = algorithm.apply(image, label, **dict(params))
        per_specimen[specimen] = scores(refined.data, reference.data)

    def mean(group: Sequence[str]) -> float:
        return round(float(np.mean([per_specimen[s]["dice"] for s in group])), 6) if group else float("nan")

    return TransferResult(
        params=dict(params),
        fitted_on=list(fitted_on),
        held_out=list(held_out),
        fitted_mean_dice=mean(fitted_on),
        held_out_mean_dice=mean(held_out),
        per_specimen=per_specimen,
    )


def leave_one_out(
    algorithm: RefinementAlgorithm,
    specimens: Sequence[str],
    loader: Loader,
) -> dict:
    """Refit excluding each specimen in turn and score on the exclusion.

    The gap between in-sample and held-out performance estimates how much of the
    fitted score is attributable to the fit itself.
    """
    if len(specimens) < 3:
        raise AdmissibilityError(
            f"leave-one-out needs at least 3 annotations, got {len(specimens)}"
        )
    folds = []
    for held in specimens:
        remaining = [s for s in specimens if s != held]
        result = fit(algorithm, remaining, loader)
        transfer = measure_transfer(algorithm, result.params, remaining, [held], loader)
        folds.append(
            {
                "held_out": held,
                "params": result.params,
                "fitted_mean_dice": result.mean_dice,
                "held_out_dice": transfer.per_specimen[held]["dice"],
            }
        )
    fitted = float(np.mean([f["fitted_mean_dice"] for f in folds]))
    out = float(np.mean([f["held_out_dice"] for f in folds]))
    return {
        "algorithm": algorithm.name,
        "n_parameters": algorithm.n_parameters,
        "folds": folds,
        "mean_fitted_dice": round(fitted, 6),
        "mean_held_out_dice": round(out, 6),
        "generalisation_gap": round(fitted - out, 6),
        "parameter_stability": {
            name: round(float(np.std([f["params"][name] for f in folds])), 6)
            for name in folds[0]["params"]
        },
    }
