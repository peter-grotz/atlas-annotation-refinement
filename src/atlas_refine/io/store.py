"""Ground-truth store with declared provenance.

Manual annotations are immutable and are admitted only through :func:`ingest`,
which validates geometry against the reference volume and records how the
annotation was produced.

The recorded ``basis`` — what the annotator started from — determines whether an
annotation may be used to fit or validate an algorithm. An annotation edited
from an algorithm's own output has its optimum pinned to the parameter that
produced it, so it measures remaining manual effort rather than accuracy.
Treating the two kinds interchangeably inflates reported performance.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from .volumes import Volume, load_nrrd_mask, save_mask

Basis = Literal["pushed", "refined", "scratch"]
Kind = Literal["independent", "derived"]

#: Which bases yield annotations that may be used to fit or validate.
BASIS_KIND: dict[str, Kind] = {
    "pushed": "independent",
    "scratch": "independent",
    "refined": "derived",
}

FILENAME = re.compile(
    r"^(?P<specimen>[A-Za-z0-9\-]+)"
    r"__(?P<structure>[a-z0-9_]+)"
    r"__(?P<basis>pushed|refined|scratch)"
    r"__(?P<annotator>[A-Za-z]{2,8})"
    r"__(?P<annotated>\d{4}-\d{2}-\d{2})\.seg\.nrrd$"
)

FILENAME_SPEC = (
    "<specimen>__<structure>__<basis>__<annotator>__<YYYY-MM-DD>.seg.nrrd\n"
    "  basis is one of: pushed | refined | scratch"
)


class IntakeError(ValueError):
    """Raised when a candidate annotation cannot be admitted."""


@dataclass(frozen=True)
class Provenance:
    specimen: str
    structure: str
    basis: Basis
    kind: Kind
    annotator: str
    annotated: str
    ingested_utc: str
    original_filename: str
    sha256: str
    shape: list[int]
    voxels: int
    usable_for_fitting: bool
    #: Why this annotation may or may not be used for fitting, stated on the
    #: record so the constraint travels with the data rather than living in
    #: documentation.
    note: str = ""

    @classmethod
    def from_record(cls, record: dict) -> "Provenance":
        """Build from a stored record, reporting unknown fields rather than
        failing on them. Records written by a later version may carry fields
        this one does not know; dropping them silently would hide a schema
        change, so they are named."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(record) - known
        if unknown:
            raise IntakeError(
                f"provenance record carries unrecognised field(s) "
                f"{sorted(unknown)}; it was written by a different version of "
                f"this package"
            )
        return cls(**record)


def parse_filename(name: str) -> dict[str, str]:
    match = FILENAME.match(name)
    if not match:
        raise IntakeError(f"filename does not match the spec:\n{FILENAME_SPEC}\ngot: {name}")
    return match.groupdict()


class GroundTruthStore:
    """Directory-backed collection of manual annotations.

    Layout::

        <root>/<structure>/<specimen>/annotation.seg.nrrd
                                     /annotation.nii.gz
                                     /provenance.json
        <root>/_index.json
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- reading ---------------------------------------------------------

    def path(self, structure: str, specimen: str) -> Path:
        return self.root / structure / specimen

    def provenance(self, structure: str, specimen: str) -> Provenance:
        record = json.loads((self.path(structure, specimen) / "provenance.json").read_text())
        return Provenance.from_record(record)

    def specimens(self, structure: str, *, kind: Kind | None = None) -> list[str]:
        """Specimen ids holding an annotation, optionally filtered by kind."""
        directory = self.root / structure
        if not directory.is_dir():
            return []
        out = []
        for child in sorted(directory.iterdir()):
            if not (child / "provenance.json").exists():
                continue
            if kind is None or self.provenance(structure, child.name).kind == kind:
                out.append(child.name)
        return out

    def fitting_set(self, structure: str) -> list[str]:
        """Specimens whose annotations may be used to fit or validate."""
        return self.specimens(structure, kind="independent")

    def load(self, structure: str, specimen: str, reference: Volume) -> Volume:
        return load_nrrd_mask(self.path(structure, specimen) / "annotation.seg.nrrd", reference)

    # -- writing ---------------------------------------------------------

    def ingest(
        self,
        source: str | Path,
        reference: Volume,
        *,
        overwrite: bool = False,
    ) -> Provenance:
        """Validate a candidate annotation and admit it to the store.

        Args:
            source: Path to a ``.seg.nrrd`` named per :data:`FILENAME_SPEC`.
            reference: Volume the annotation must be co-registered with.
            overwrite: Replace an existing annotation. Off by default; manual
                annotations are treated as immutable.

        Raises:
            IntakeError: If the name, geometry, or content fails validation.
                Nothing is written when this is raised.
        """
        source = Path(source)
        fields = parse_filename(source.name)

        mask = load_nrrd_mask(source, reference)  # geometry validated here
        count = int((mask.data > 0).sum())
        if count == 0:
            raise IntakeError("annotation is empty")

        destination = self.path(fields["structure"], fields["specimen"])
        if (destination / "annotation.seg.nrrd").exists() and not overwrite:
            raise IntakeError(
                f"an annotation already exists at {destination}; "
                f"pass overwrite=True to replace it deliberately"
            )

        kind = BASIS_KIND[fields["basis"]]
        record = Provenance(
            specimen=fields["specimen"],
            structure=fields["structure"],
            basis=fields["basis"],  # type: ignore[arg-type]
            kind=kind,
            annotator=fields["annotator"],
            annotated=fields["annotated"],
            ingested_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            original_filename=source.name,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            shape=list(mask.shape),
            voxels=count,
            usable_for_fitting=kind == "independent",
            note=(
                "Edited from an algorithm output. Measures the manual effort "
                "remaining after automatic correction, not accuracy. Must be "
                "excluded when fitting or validating the parameters that "
                "produced its starting point."
                if kind == "derived"
                else "Edited from the propagated label or traced from scratch. "
                "Independent of any algorithm parameter; usable for fitting "
                "and validation."
            ),
        )

        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / "annotation.seg.nrrd")
        save_mask(mask, destination / "annotation.nii.gz", reference)
        (destination / "provenance.json").write_text(json.dumps(asdict(record), indent=2))
        self.reindex()
        return record

    def reindex(self) -> dict:
        """Regenerate ``_index.json`` from the provenance records on disk."""
        annotations: dict[str, dict] = {}
        for structure_dir in sorted(self.root.iterdir()):
            if not structure_dir.is_dir() or structure_dir.name.startswith(("_", ".")):
                continue
            for specimen_dir in sorted(structure_dir.iterdir()):
                record = specimen_dir / "provenance.json"
                if record.exists():
                    annotations.setdefault(structure_dir.name, {})[specimen_dir.name] = json.loads(
                        record.read_text()
                    )
        summary = {
            structure: {
                "total": len(entries),
                "independent": sum(1 for e in entries.values() if e["kind"] == "independent"),
                "derived": sum(1 for e in entries.values() if e["kind"] == "derived"),
            }
            for structure, entries in annotations.items()
        }
        index = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": summary,
            "annotations": annotations,
        }
        (self.root / "_index.json").write_text(json.dumps(index, indent=2))
        return index
