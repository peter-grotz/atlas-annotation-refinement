"""Tests for the error signature battery.

The three fixtures below reproduce the failure modes the battery must tell
apart. Each is a cube of structure inside a dimmer margin, differing only in how
the propagated label sits relative to the reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas_refine.characterize import characterize
from atlas_refine.characterize.signature import classify_error
from atlas_refine.io.volumes import Volume

SHAPE = (24, 24, 24)
BACKGROUND, MARGIN, STRUCTURE = 1000.0, 1200.0, 2000.0


def image() -> Volume:
    """Bright structure inside a dimmer margin, above a background plateau."""
    data = np.full(SHAPE, BACKGROUND, dtype="float32")
    data[4:20, 4:20, 4:20] = MARGIN
    data[7:17, 7:17, 7:17] = STRUCTURE
    spacing = (20.0, 20.0, 20.0)
    return Volume(
        data=data,
        spacing=spacing,
        unit="um",
        affine=np.diag([-spacing[0], -spacing[1], spacing[2], 1.0]),
        frame="test",
    )


def box(lo: int, hi: int, offset: int = 0) -> Volume:
    mask = np.zeros(SHAPE, dtype="uint8")
    mask[lo + offset : hi + offset, lo:hi, lo:hi] = 1
    return image().with_data(mask)


REFERENCE = (7, 17)


class TestErrorClassification:
    """The taxonomy drives family selection, so each case must be named right."""

    def test_over_coverage(self):
        signature = characterize(image(), box(4, 20), box(*REFERENCE))
        assert signature.dominant_error == "over-coverage"
        assert signature.volume_ratio > 1.0

    def test_under_coverage(self):
        signature = characterize(image(), box(9, 15), box(*REFERENCE))
        assert signature.dominant_error == "under-coverage"
        assert signature.volume_ratio < 1.0

    def test_displacement_is_not_reported_as_under_coverage(self):
        """A label of the right size in the wrong place needs moving, not resizing.

        Comparing raw counts would name a direction from a negligible size
        difference, and the sign of that difference is arbitrary.
        """
        signature = characterize(image(), box(*REFERENCE, offset=5), box(*REFERENCE))
        assert signature.dominant_error == "displacement"
        assert signature.volume_ratio == pytest.approx(1.0, abs=1e-6)
        assert signature.dice < 0.7

    def test_close_agreement(self):
        signature = characterize(image(), box(*REFERENCE, offset=1), box(*REFERENCE))
        assert signature.dominant_error == "close-agreement"


class TestClassifyErrorDirectly:
    @pytest.mark.parametrize(
        "ratio,excess,missed,expected",
        [
            (1.00, 500, 500, "displacement"),      # right volume, both errors large
            (1.00, 10, 10, "close-agreement"),     # right volume, errors negligible
            (1.40, 500, 10, "over-coverage"),
            (0.60, 10, 500, "under-coverage"),
            (1.05, 500, 500, "displacement"),      # inside the deadband
            (1.20, 500, 500, "over-coverage"),     # outside it, so direction is real
        ],
    )
    def test_taxonomy(self, ratio, excess, missed, expected):
        assert classify_error(ratio, excess, missed, reference=1000) == expected

    def test_empty_reference_is_undefined(self):
        assert classify_error(float("nan"), 0, 0, reference=0) == "undefined"


class TestIntensitySeparation:
    def test_positive_when_populations_are_disjoint(self):
        """Excess in the dim margin separates cleanly from bright structure,
        which is what makes an intensity criterion applicable."""
        signature = characterize(image(), box(4, 20), box(*REFERENCE))
        assert signature.intensity_separation > 0
        assert signature.excess.median < signature.matched.median

    def test_negative_when_populations_overlap(self):
        """Displacement within one heterogeneous tissue draws excess and matches
        from the same distribution, so no threshold can separate them.

        The shared fixture is piecewise-uniform, so a displaced label there moves
        into the dimmer margin and does separate. Overlap requires the label to
        stay inside a single textured region, which is the situation in real
        acquisitions where a boundary is diffuse.
        """
        rng = np.random.default_rng(0)
        data = np.full(SHAPE, BACKGROUND, dtype="float32")
        data[2:22, 2:22, 2:22] = STRUCTURE + rng.normal(0, 250, (20, 20, 20))
        spacing = (20.0, 20.0, 20.0)
        textured = Volume(
            data=data,
            spacing=spacing,
            unit="um",
            affine=np.diag([-spacing[0], -spacing[1], spacing[2], 1.0]),
            frame="test",
        )

        def sub(offset: int) -> Volume:
            mask = np.zeros(SHAPE, dtype="uint8")
            mask[8 + offset : 16 + offset, 8:16, 8:16] = 1
            return textured.with_data(mask)

        signature = characterize(textured, sub(4), sub(0))
        assert signature.dominant_error == "displacement"
        assert signature.intensity_separation < 0
        assert -1.0 < signature.intensity_separation < 0

    def test_separation_is_bounded(self):
        """Uniform populations must not produce an unbounded or undefined value."""
        for label in (box(4, 20), box(9, 15), box(*REFERENCE, offset=5)):
            separation = characterize(image(), label, box(*REFERENCE)).intensity_separation
            assert np.isnan(separation) or -1.0 <= separation <= 1.0


class TestGeometryFields:
    def test_shell_excess_is_surface_adjacent_and_coherent(self):
        signature = characterize(image(), box(4, 20), box(*REFERENCE))
        assert signature.excess_largest_component == pytest.approx(1.0)
        assert signature.excess_surface_fraction > 0.5

    def test_no_excess_when_the_label_is_contained(self):
        signature = characterize(image(), box(9, 15), box(*REFERENCE))
        assert signature.excess.count == 0
        assert signature.excess_largest_component == 0.0


class TestSubtractiveCeiling:
    def test_equals_recall(self):
        """Missed voxels lie outside the label, so removal cannot recover them."""
        signature = characterize(image(), box(4, 20), box(*REFERENCE))
        assert signature.subtractive_recall_ceiling == signature.recall

    def test_a_contained_label_caps_recall_below_one(self):
        signature = characterize(image(), box(9, 15), box(*REFERENCE))
        assert signature.subtractive_recall_ceiling < 1.0

    def test_summary_names_it_as_subtractive(self):
        text = characterize(image(), box(4, 20), box(*REFERENCE)).summary()
        assert "subtractive recall ceiling" in text


class TestCompatibility:
    def test_rejects_a_label_on_a_different_grid(self):
        from atlas_refine.io.volumes import GeometryError

        img = image()
        wrong = Volume(
            data=np.ones((24, 24, 25), dtype="uint8"),
            spacing=img.spacing,
            unit=img.unit,
            affine=img.affine,
            frame=img.frame,
        )
        with pytest.raises(GeometryError, match="shape mismatch"):
            characterize(img, wrong, box(*REFERENCE))
