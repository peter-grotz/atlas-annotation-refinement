"""Append-only record of every correction attempt.

Most attempted corrections fail, and the reasons generalise across structures
more reliably than the successes do. A family that suits a thin low-intensity
feature will fail on a thick high-intensity one for a reason that is a property
of the image, not of the structure — so a record of what was tried, against what
error signature, and how it fared is the artefact that lets each structure start
from where the last one finished rather than from scratch.

Trials are indexed by the measured error signature rather than by structure
name. Two structures with similar signatures call for similar corrections even
when they are anatomically unrelated, and the same structure in a different
cohort may present a signature that has never been seen. Retrieval is therefore
by measured similarity, not by label.

The log is append-only. Superseding a family records a new entry referencing the
old one; nothing is rewritten, so the history of what was believed and when
remains readable.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np

Outcome = Literal["admitted", "rejected", "superseded", "explored"]

#: Signature fields compared when retrieving similar prior attempts. Each is
#: scale-free, so specimens of differing brightness remain comparable.
COMPARISON_FIELDS = (
    "volume_ratio",
    "intensity_separation",
    "excess_median_depth",
    "excess_surface_fraction",
    "excess_largest_component",
    "subtractive_recall_ceiling",
)

#: Typical spread of each field, used to weight the distance so that no single
#: field dominates purely because of its natural range.
_FIELD_SCALE = {
    "volume_ratio": 0.30,
    "intensity_separation": 0.40,
    "excess_median_depth": 3.00,
    "excess_surface_fraction": 0.30,
    "excess_largest_component": 0.30,
    "subtractive_recall_ceiling": 0.15,
}


@dataclass(frozen=True)
class Trial:
    """One attempt to correct a propagated label."""

    trial_id: str
    recorded_utc: str
    structure: str
    specimens: list[str]
    family: str
    params: dict[str, float]
    signature: dict[str, float]
    scores: dict[str, float]
    outcome: Outcome
    #: Why the attempt ended as it did, in terms a later reader can act on.
    reason: str
    #: Whether the score is an in-sample fit or measured on excluded annotations.
    evidence: Literal["in-sample", "held-out"] = "in-sample"
    #: Family this attempt replaces, if any.
    supersedes: str | None = None
    #: Free-form notes; the authoring agent's reasoning belongs here.
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def signature_vector(signature: dict[str, float]) -> np.ndarray:
    """Project a signature onto the comparison fields, scaled."""
    return np.array(
        [
            float(signature.get(field, np.nan)) / _FIELD_SCALE[field]
            for field in COMPARISON_FIELDS
        ],
        dtype=float,
    )


def signature_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Scaled Euclidean distance, ignoring fields absent from either signature."""
    va, vb = signature_vector(a), signature_vector(b)
    usable = np.isfinite(va) & np.isfinite(vb)
    if not usable.any():
        return float("inf")
    return float(np.linalg.norm(va[usable] - vb[usable]) / np.sqrt(usable.sum()))


class TrialLog:
    """Append-only JSON Lines record of correction attempts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------

    def record(
        self,
        *,
        structure: str,
        specimens: Sequence[str],
        family: str,
        params: dict[str, float],
        signature: dict[str, float],
        scores: dict[str, float],
        outcome: Outcome,
        reason: str,
        evidence: Literal["in-sample", "held-out"] = "in-sample",
        supersedes: str | None = None,
        notes: str = "",
    ) -> Trial:
        """Append a trial. Records failures as well as successes."""
        trial = Trial(
            trial_id=uuid.uuid4().hex[:12],
            recorded_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            structure=structure,
            specimens=list(specimens),
            family=family,
            params=dict(params),
            signature={k: float(v) for k, v in signature.items() if isinstance(v, (int, float))},
            scores=dict(scores),
            outcome=outcome,
            reason=reason,
            evidence=evidence,
            supersedes=supersedes,
            notes=notes,
        )
        with self.path.open("a") as handle:
            handle.write(json.dumps(trial.to_dict()) + "\n")
        return trial

    # -- reading ---------------------------------------------------------

    def __iter__(self) -> Iterator[Trial]:
        if not self.path.exists():
            return iter(())
        for line in self.path.read_text().splitlines():
            if line.strip():
                yield Trial(**json.loads(line))

    def all(self) -> list[Trial]:
        return list(self)

    def similar(
        self,
        signature: dict[str, float],
        *,
        limit: int = 10,
        max_distance: float = float("inf"),
    ) -> list[tuple[float, Trial]]:
        """Prior attempts against the most similar signatures, nearest first.

        This is the retrieval an agent performs before proposing anything: what
        has already been tried against a comparable failure, and how did it go.
        """
        scored = [
            (signature_distance(signature, trial.signature), trial)
            for trial in self
            if trial.signature
        ]
        scored = [(d, t) for d, t in scored if np.isfinite(d) and d <= max_distance]
        scored.sort(key=lambda pair: pair[0])
        return scored[:limit]

    def families_tried(self, signature: dict[str, float], *, max_distance: float = 1.0) -> dict:
        """Summarise outcomes per family against comparable signatures.

        Returns families ordered by best observed score, each carrying the
        reasons recorded for its failures. The reasons are the point: they say
        why a family is unsuitable for this kind of failure, which is what
        prevents the same dead end being explored again.
        """
        out: dict[str, dict] = {}
        for distance, trial in self.similar(signature, limit=1000, max_distance=max_distance):
            entry = out.setdefault(
                trial.family,
                {"attempts": 0, "best_dice": None, "outcomes": [], "reasons": []},
            )
            entry["attempts"] += 1
            dice = trial.scores.get("dice")
            if dice is not None and (entry["best_dice"] is None or dice > entry["best_dice"]):
                entry["best_dice"] = dice
            entry["outcomes"].append(trial.outcome)
            if trial.reason and trial.reason not in entry["reasons"]:
                entry["reasons"].append(trial.reason)
        return dict(
            sorted(out.items(), key=lambda kv: (kv[1]["best_dice"] is None, -(kv[1]["best_dice"] or 0)))
        )

    def summary(self) -> dict:
        trials = self.all()
        by_outcome: dict[str, int] = {}
        by_structure: dict[str, int] = {}
        for trial in trials:
            by_outcome[trial.outcome] = by_outcome.get(trial.outcome, 0) + 1
            by_structure[trial.structure] = by_structure.get(trial.structure, 0) + 1
        return {
            "total": len(trials),
            "by_outcome": by_outcome,
            "by_structure": by_structure,
            "families": sorted({t.family for t in trials}),
        }


def render_index(log: TrialLog) -> str:
    """Render a human-readable index of families and the signatures they suit."""
    trials = log.all()
    if not trials:
        return "# Correction families\n\nNo trials recorded yet.\n"

    lines = [
        "# Correction families",
        "",
        "Generated from the trial log. Each family is listed with the error",
        "signatures it has been attempted against and how it fared.",
        "",
        "| family | attempts | admitted | best dice | structures |",
        "|---|---|---|---|---|",
    ]
    families: dict[str, dict] = {}
    for trial in trials:
        entry = families.setdefault(
            trial.family, {"attempts": 0, "admitted": 0, "best": None, "structures": set()}
        )
        entry["attempts"] += 1
        entry["admitted"] += trial.outcome == "admitted"
        entry["structures"].add(trial.structure)
        dice = trial.scores.get("dice")
        if dice is not None and (entry["best"] is None or dice > entry["best"]):
            entry["best"] = dice
    for name, entry in sorted(families.items(), key=lambda kv: -(kv[1]["best"] or 0)):
        best = f"{entry['best']:.4f}" if entry["best"] is not None else "—"
        lines.append(
            f"| `{name}` | {entry['attempts']} | {entry['admitted']} | {best} | "
            f"{', '.join(sorted(entry['structures']))} |"
        )

    lines += ["", "## Recorded dead ends", "",
              "Attempts that were rejected, with the reason. Consult before proposing.", ""]
    seen: set[tuple[str, str]] = set()
    for trial in trials:
        if trial.outcome != "rejected" or not trial.reason:
            continue
        key = (trial.family, trial.reason)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **`{trial.family}`** on *{trial.structure}* — {trial.reason}")
    if not seen:
        lines.append("_none recorded_")
    return "\n".join(lines) + "\n"
