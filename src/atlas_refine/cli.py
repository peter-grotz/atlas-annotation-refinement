"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .algorithms import REGISTRY
from .io.store import FILENAME_SPEC, GroundTruthStore, IntakeError, parse_filename
from .io.volumes import GeometryError, load_nifti


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", default="ground_truth", help="Ground-truth store root.")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Voxel spacing override, applied when headers are unreliable.",
    )
    parser.add_argument("--unit", choices=["um", "mm"], help="Unit of --spacing.")


def cmd_ingest(args: argparse.Namespace) -> int:
    store = GroundTruthStore(args.store)
    candidates = (
        sorted(Path(args.inbox).glob("*.seg.nrrd")) if args.inbox else [Path(p) for p in args.files]
    )
    if not candidates:
        print("nothing to ingest")
        print(f"\nexpected filenames:\n  {FILENAME_SPEC}")
        return 0

    # An inbox generally holds annotations of several specimens, each of which
    # must be checked against its own volume. A reference path containing
    # "{sid}" is resolved per file from the specimen in its name; a plain path
    # is used for every file, which is only correct for a single specimen.
    per_specimen = "{sid}" in args.reference
    references: dict[str, object] = {}

    def reference_for(specimen: str):
        key = specimen if per_specimen else ""
        if key not in references:
            path = args.reference.format(sid=specimen) if per_specimen else args.reference
            frame = args.frame.format(sid=specimen)
            references[key] = load_nifti(
                path, frame=frame, spacing=args.spacing, unit=args.unit
            )
        return references[key]

    accepted = rejected = 0
    for path in candidates:
        try:
            record = store.ingest(
                path,
                reference_for(parse_filename(path.name)["specimen"]),
                overwrite=args.overwrite,
            )
        except (IntakeError, GeometryError) as exc:
            rejected += 1
            print(f"REJECT  {path.name}\n        {exc}")
            continue
        accepted += 1
        print(
            f"OK      {path.name}\n"
            f"        {record.structure}/{record.specimen}  {record.voxels:,} voxels  "
            f"kind={record.kind}  fitting={'yes' if record.usable_for_fitting else 'no'}"
        )
    print(f"\n{accepted} ingested, {rejected} rejected")
    return 1 if rejected else 0


def cmd_status(args: argparse.Namespace) -> int:
    index = GroundTruthStore(args.store).reindex()
    summary = index["summary"]
    if not summary:
        print("store is empty")
        return 0
    print(f"{'structure':<24}{'total':>7}{'independent':>13}{'derived':>9}")
    print("-" * 53)
    for structure, counts in sorted(summary.items()):
        print(
            f"{structure:<24}{counts['total']:>7}{counts['independent']:>13}{counts['derived']:>9}"
        )
    print(
        "\nOnly independent annotations may be used to fit or validate. "
        "Derived annotations measure remaining manual effort."
    )
    return 0


def cmd_algorithms(args: argparse.Namespace) -> int:
    for name, description in REGISTRY.describe().items():
        algorithm = REGISTRY.create(name)
        params = ", ".join(p.name for p in algorithm.parameters)
        print(f"{name}\n  {description}\n  parameters ({algorithm.n_parameters}): {params}\n")
    print(
        "An algorithm may be fitted only when its parameter count does not exceed "
        "the number of independent annotations available."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas-refine",
        description="Atlas annotation propagation and ground-truth-anchored refinement.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Validate and admit manual annotations.")
    _add_common(ingest)
    ingest.add_argument(
        "--reference",
        required=True,
        help="Volume the annotations align to. May contain '{sid}', resolved per "
        "file from the specimen in its name, so one inbox can hold several specimens.",
    )
    ingest.add_argument(
        "--frame", required=True, help="Coordinate frame name. May contain '{sid}'."
    )
    ingest.add_argument("--inbox", help="Directory of candidate .seg.nrrd files.")
    ingest.add_argument("files", nargs="*", help="Individual files to ingest.")
    ingest.add_argument("--overwrite", action="store_true", help="Replace existing annotations.")
    ingest.set_defaults(func=cmd_ingest)

    status = sub.add_parser("status", help="Summarise the ground-truth store.")
    status.add_argument("--store", default="ground_truth")
    status.set_defaults(func=cmd_status)

    algorithms = sub.add_parser("algorithms", help="List available refinement algorithms.")
    algorithms.set_defaults(func=cmd_algorithms)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
