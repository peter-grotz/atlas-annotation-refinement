"""Tests for the command-line interface.

The CLI is how annotations enter the system, so its failure behaviour matters as
much as its success behaviour: a rejected file must leave nothing behind and
report why.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import nrrd
import pytest

from atlas_refine.cli import main

SHAPE = (16, 16, 16)
SPACING = (20.0, 20.0, 20.0)
AFFINE = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    """A specimen volume with a plausible background and a bright interior."""
    data = np.full(SHAPE, 1000.0, dtype="float32")
    data[4:12, 4:12, 4:12] = 2000.0
    header = nib.Nifti1Header()
    header.set_xyzt_units(xyz="micron")
    path = tmp_path / "specimen.nii.gz"
    nib.save(nib.Nifti1Image(data, AFFINE, header), str(path))
    return path


def write_mask(path: Path, lo: int = 4, hi: int = 12, empty: bool = False) -> Path:
    """A segmentation on the same grid, in the LPS convention NRRD uses."""
    mask = np.zeros(SHAPE, dtype="uint8")
    if not empty:
        mask[lo:hi, lo:hi, lo:hi] = 1
    header = {
        "space": "left-posterior-superior",
        "space directions": np.diag(SPACING),
        "space origin": np.zeros(3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(path), mask, header)
    return path


@pytest.fixture
def inbox(tmp_path: Path) -> Path:
    directory = tmp_path / "inbox"
    directory.mkdir()
    return directory


def run(*args: str) -> int:
    return main(list(args))


class TestStatus:
    def test_empty_store(self, tmp_path: Path, capsys):
        assert run("status", "--store", str(tmp_path / "gt")) == 0
        assert "empty" in capsys.readouterr().out

    def test_reports_holdings_split_by_admissibility(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        write_mask(inbox / "100001__ventricles__pushed__AB__2026-01-15.seg.nrrd")
        write_mask(inbox / "100002__ventricles__refined__AB__2026-01-16.seg.nrrd")
        run("ingest", "--store", str(store), "--reference", str(reference),
            "--frame", "spec", "--inbox", str(inbox))
        capsys.readouterr()
        assert run("status", "--store", str(store)) == 0
        out = capsys.readouterr().out
        assert "ventricles" in out
        assert "independent" in out


class TestIngest:
    def test_accepts_a_well_formed_annotation(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        write_mask(inbox / "100001__ventricles__pushed__AB__2026-01-15.seg.nrrd")
        assert run("ingest", "--store", str(store), "--reference", str(reference),
                   "--frame", "spec", "--inbox", str(inbox)) == 0
        out = capsys.readouterr().out
        assert "OK" in out and "1 ingested, 0 rejected" in out
        assert (store / "ventricles" / "100001" / "annotation.seg.nrrd").exists()
        assert (store / "ventricles" / "100001" / "annotation.nii.gz").exists()

    def test_records_admissibility_from_the_basis(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        write_mask(inbox / "100001__ventricles__refined__AB__2026-01-15.seg.nrrd")
        run("ingest", "--store", str(store), "--reference", str(reference),
            "--frame", "spec", "--inbox", str(inbox))
        record = json.loads((store / "ventricles" / "100001" / "provenance.json").read_text())
        assert record["kind"] == "derived"
        assert record["usable_for_fitting"] is False
        assert "fitting=no" in capsys.readouterr().out

    def test_rejects_a_malformed_filename_and_writes_nothing(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        write_mask(inbox / "not_the_expected_pattern.seg.nrrd")
        assert run("ingest", "--store", str(store), "--reference", str(reference),
                   "--frame", "spec", "--inbox", str(inbox)) == 1
        out = capsys.readouterr().out
        assert "REJECT" in out and "0 ingested, 1 rejected" in out
        assert not list(store.rglob("annotation.seg.nrrd"))

    def test_rejects_an_empty_annotation(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        write_mask(inbox / "100001__ventricles__pushed__AB__2026-01-15.seg.nrrd", empty=True)
        assert run("ingest", "--store", str(store), "--reference", str(reference),
                   "--frame", "spec", "--inbox", str(inbox)) == 1
        assert "empty" in capsys.readouterr().out

    def test_rejects_a_mask_on_a_different_grid(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        path = inbox / "100001__ventricles__pushed__AB__2026-01-15.seg.nrrd"
        wrong = np.zeros((16, 16, 17), dtype="uint8")
        wrong[4:12, 4:12, 4:12] = 1
        nrrd.write(str(path), wrong, {
            "space": "left-posterior-superior",
            "space directions": np.diag(SPACING),
            "space origin": np.zeros(3),
        })
        assert run("ingest", "--store", str(store), "--reference", str(reference),
                   "--frame", "spec", "--inbox", str(inbox)) == 1
        assert "shape mismatch" in capsys.readouterr().out

    def test_does_not_overwrite_existing_ground_truth(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        name = "100001__ventricles__pushed__AB__2026-01-15.seg.nrrd"
        write_mask(inbox / name)
        run("ingest", "--store", str(store), "--reference", str(reference),
            "--frame", "spec", "--inbox", str(inbox))
        write_mask(inbox / name, lo=5, hi=11)
        capsys.readouterr()
        assert run("ingest", "--store", str(store), "--reference", str(reference),
                   "--frame", "spec", "--inbox", str(inbox)) == 1
        assert "already exists" in capsys.readouterr().out

    def test_overwrite_flag_replaces_deliberately(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        name = "100001__ventricles__pushed__AB__2026-01-15.seg.nrrd"
        write_mask(inbox / name)
        run("ingest", "--store", str(store), "--reference", str(reference),
            "--frame", "spec", "--inbox", str(inbox))
        write_mask(inbox / name, lo=5, hi=11)
        capsys.readouterr()
        assert run("ingest", "--store", str(store), "--reference", str(reference),
                   "--frame", "spec", "--inbox", str(inbox), "--overwrite") == 0

    def test_a_rejection_does_not_block_the_rest_of_the_batch(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        write_mask(inbox / "bad_name.seg.nrrd")
        write_mask(inbox / "100002__ventricles__pushed__AB__2026-01-15.seg.nrrd")
        assert run("ingest", "--store", str(store), "--reference", str(reference),
                   "--frame", "spec", "--inbox", str(inbox)) == 1
        assert "1 ingested, 1 rejected" in capsys.readouterr().out

    def test_reports_the_spec_when_there_is_nothing_to_ingest(self, tmp_path, reference, inbox, capsys):
        assert run("ingest", "--store", str(tmp_path / "gt"), "--reference", str(reference),
                   "--frame", "spec", "--inbox", str(inbox)) == 0
        out = capsys.readouterr().out
        assert "nothing to ingest" in out and "basis" in out

    def test_named_files_can_be_passed_directly(self, tmp_path, reference, inbox, capsys):
        store = tmp_path / "gt"
        path = write_mask(inbox / "100001__ventricles__pushed__AB__2026-01-15.seg.nrrd")
        assert run("ingest", "--store", str(store), "--reference", str(reference),
                   "--frame", "spec", str(path)) == 0
        assert "1 ingested" in capsys.readouterr().out


class TestAlgorithms:
    def test_lists_families_with_parameter_counts(self, capsys):
        assert run("algorithms") == 0
        out = capsys.readouterr().out
        assert "contrast_threshold" in out
        assert "parameters" in out
        assert "independent annotations" in out


class TestArgumentHandling:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            main([])

    def test_rejects_an_unknown_subcommand(self):
        with pytest.raises(SystemExit):
            main(["nonexistent"])


class TestPerSpecimenReference:
    """An inbox generally holds several specimens, each on its own grid."""

    @staticmethod
    def _specimen(root: Path, sid: str, size: int) -> None:
        """A volume and a matching annotation, on a grid unique to this size."""
        shape = (size, size, size)
        data = np.full(shape, 1000.0, dtype="float32")
        data[2 : size - 2, 2 : size - 2, 2 : size - 2] = 2000.0
        header = nib.Nifti1Header()
        header.set_xyzt_units(xyz="micron")
        (root / sid).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, AFFINE, header), str(root / sid / "image.nii.gz"))

        mask = np.zeros(shape, dtype="uint8")
        mask[3 : size - 3, 3 : size - 3, 3 : size - 3] = 1
        nrrd.write(
            str(root / "inbox" / f"{sid}__isocortex__pushed__PG__2026-01-01.seg.nrrd"),
            mask,
            {
                "space": "left-posterior-superior",
                "space directions": np.diag(SPACING),
                "space origin": np.zeros(3),
            },
        )

    def test_resolves_a_reference_per_specimen(self, tmp_path, capsys):
        (tmp_path / "inbox").mkdir()
        self._specimen(tmp_path, "100001", 16)
        self._specimen(tmp_path, "100002", 20)

        assert (
            run(
                "ingest",
                "--store", str(tmp_path / "store"),
                "--reference", str(tmp_path / "{sid}" / "image.nii.gz"),
                "--frame", "spec-{sid}",
                "--inbox", str(tmp_path / "inbox"),
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "2 ingested, 0 rejected" in out

    def test_a_single_reference_rejects_the_other_specimen(self, tmp_path, capsys):
        """Why the substitution is needed: one fixed reference cannot check both,
        and the geometry check refuses the mismatch rather than mis-filing it."""
        (tmp_path / "inbox").mkdir()
        self._specimen(tmp_path, "100001", 16)
        self._specimen(tmp_path, "100002", 20)

        assert (
            run(
                "ingest",
                "--store", str(tmp_path / "store"),
                "--reference", str(tmp_path / "100001" / "image.nii.gz"),
                "--frame", "spec",
                "--inbox", str(tmp_path / "inbox"),
            )
            == 1
        )
        assert "1 ingested, 1 rejected" in capsys.readouterr().out
