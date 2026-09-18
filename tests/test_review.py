"""Tests for plausibility screening and diagnostic rendering."""

from __future__ import annotations

import numpy as np
import pytest

from atlas_refine.io.volumes import Volume
from atlas_refine.review import (
    CohortReview,
    LabelScreen,
    rank_cohort,
    render_panel,
    screen_label,
)

SHAPE = (48, 48, 48)
SPACING = (20.0, 20.0, 20.0)
AFFINE = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])


def vol(data: np.ndarray, unit: str = "um") -> Volume:
    return Volume(data.astype("float32"), SPACING, unit, AFFINE, "test")


def ball(centre=(24, 24, 24), radius=14.0, shape=SHAPE) -> np.ndarray:
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    r = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
    return (r <= radius ** 2).astype("uint8")


def image_for(mask: np.ndarray, inside=3000.0, outside=1000.0) -> Volume:
    data = np.full(mask.shape, outside)
    data[mask > 0] = inside
    return vol(data)


class TestScreenShape:
    def test_a_solid_structure_raises_no_concern(self):
        mask = ball()
        screen = screen_label(vol(mask), image_for(mask), specimen="clean")
        assert screen.concerns == []
        assert not screen.looks_implausible
        assert screen.components == 1
        assert screen.cohesion == pytest.approx(1.0)

    def test_an_empty_label_is_reported_not_crashed(self):
        screen = screen_label(vol(np.zeros(SHAPE, dtype="uint8")), specimen="empty")
        assert screen.voxels == 0
        assert screen.looks_implausible
        assert any("empty" in c for c in screen.concerns)

    def test_scattered_noise_is_caught_as_fragmentation(self):
        rng = np.random.default_rng(0)
        mask = ball(radius=9.0)
        speckle = rng.random(SHAPE) < 0.02
        mask = (mask.astype(bool) | speckle).astype("uint8")
        screen = screen_label(vol(mask), specimen="speckled")
        assert screen.cohesion < 0.90
        assert any("one piece" in c for c in screen.concerns)

    def test_debris_count_excludes_the_main_structure(self):
        mask = ball(radius=12.0)
        mask[2, 2, 2] = 1
        screen = screen_label(vol(mask), specimen="mote")
        assert screen.debris >= 1
        assert screen.components == 1

    def test_speckle_alone_raises_no_concern_when_the_body_is_intact(self):
        """Every real label carries hundreds of single-voxel specks. A check
        that fires on all of them discriminates nothing - measured on a real
        cohort, all ten brains tripped a raw debris count."""
        mask = ball(radius=14.0)
        rng = np.random.default_rng(7)
        specks = (rng.random(SHAPE) < 0.004) & (mask == 0)
        mask = (mask.astype(bool) | specks).astype("uint8")
        screen = screen_label(vol(mask), specimen="speckly")
        assert screen.debris > 50               # the specks are there
        assert screen.cohesion > 0.95           # but hold almost no volume
        assert screen.concerns == []            # so nothing is reported

    def test_many_substantial_pieces_are_reported(self):
        """Dozens of real pieces is a different thing from dust."""
        mask = np.zeros(SHAPE, dtype="uint8")
        for i in range(8):
            z = 3 + i * 5
            mask[z:z + 4, 20:24, 20:24] = 1
        screen = screen_label(vol(mask), specimen="scattered")
        assert screen.components > 5
        assert any("separate pieces" in c for c in screen.concerns)

    def test_a_sheet_wrapping_anatomy_is_not_called_holed(self):
        """Isocortex is a sheet enclosing most of the brain. Counting what it
        wraps as a defect would flag every cortical label in a cohort."""
        zz, yy, xx = np.mgrid[0:48, 0:48, 0:48]
        r = np.sqrt((zz - 24) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2)
        shell = ((r <= 18) & (r >= 14)).astype("uint8")
        screen = screen_label(vol(shell), specimen="sheet")
        assert screen.enclosed > screen.voxels * 0.5      # it does wrap a lot
        assert screen.punctures == 0                       # none of it is a hole
        assert not any("holes" in c for c in screen.concerns)

    def test_holes_punched_in_a_sheet_are_still_caught(self):
        """The distinction must not become a blanket exemption for sheets."""
        zz, yy, xx = np.mgrid[0:48, 0:48, 0:48]
        r = np.sqrt((zz - 24) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2)
        shell = ((r <= 18) & (r >= 14)).astype("uint8")
        rng = np.random.default_rng(3)
        shell[(rng.random(SHAPE) < 0.30) & (shell > 0)] = 0
        screen = screen_label(vol(shell), specimen="moth-eaten")
        assert screen.punctures > 0

    def test_interior_holes_are_reported(self):
        mask = ball(radius=14.0)
        mask[20:28, 20:28, 20:28] = 0
        screen = screen_label(vol(mask), specimen="hollow")
        assert screen.punctures > 0
        assert any("holes" in c for c in screen.concerns)


class TestSymmetry:
    def test_a_lopsided_label_is_caught_when_the_axis_is_given(self):
        mask = ball(centre=(24, 24, 24), radius=12.0)
        mask[:, :, :24] = 0                      # remove one side along axis 2
        screen = screen_label(vol(mask), specimen="half", lateral_axis=2)
        assert screen.lateral_asymmetry is not None
        assert screen.lateral_asymmetry > 0.15
        assert any("lopsided" in c for c in screen.concerns)

    def test_a_symmetric_label_passes(self):
        screen = screen_label(vol(ball()), specimen="whole", lateral_axis=2)
        assert screen.lateral_asymmetry < 0.15

    def test_the_check_is_skipped_rather_than_guessed(self):
        """Guessing the lateral axis would hide exactly the failure it detects."""
        screen = screen_label(vol(ball()), specimen="unknown-axis")
        assert screen.lateral_asymmetry is None
        assert not any("lopsided" in c for c in screen.concerns)


class TestBoundary:
    def test_a_label_on_a_real_edge_scores_positive_sharpness(self):
        mask = ball()
        screen = screen_label(vol(mask), image_for(mask), specimen="edged")
        assert screen.boundary_sharpness is not None
        assert screen.boundary_sharpness > 0

    def test_a_label_in_a_flat_region_is_flagged_as_misplaced(self):
        mask = ball()
        flat = vol(np.full(SHAPE, 2000.0))
        screen = screen_label(vol(mask), flat, specimen="flat")
        assert screen.boundary_sharpness is None or screen.boundary_sharpness <= 0

    def test_sharpness_is_omitted_when_no_image_is_supplied(self):
        assert screen_label(vol(ball()), specimen="no-image").boundary_sharpness is None


class TestPhysicalVolume:
    def test_micron_spacing_converts_to_cubic_millimetres(self):
        mask = ball()
        screen = screen_label(vol(mask), specimen="um")
        expected = int(mask.sum()) * (20.0 ** 3) / 1e9
        assert screen.volume_mm3 == pytest.approx(expected, rel=1e-4)

    def test_millimetre_spacing_is_not_rescaled(self):
        mask = ball()
        v = Volume(mask.astype("float32"), (0.02, 0.02, 0.02), "mm", AFFINE, "test")
        screen = screen_label(v, specimen="mm")
        expected = int(mask.sum()) * (0.02 ** 3)
        assert screen.volume_mm3 == pytest.approx(expected, rel=1e-4)


def make_screen(specimen: str, volume_mm3: float, sharpness: float | None = 0.1,
                concerns: list[str] | None = None) -> LabelScreen:
    return LabelScreen(
        specimen=specimen, voxels=1000, volume_mm3=volume_mm3, components=1,
        debris=0, cohesion=1.0, surface_ratio=0.3, enclosed=0, punctures=0,
        lateral_asymmetry=None, boundary_sharpness=sharpness,
        concerns=list(concerns or []),
    )


class TestRanking:
    def _cohort(self):
        return [
            make_screen("a", 1.00), make_screen("b", 1.02), make_screen("c", 0.99),
            make_screen("d", 1.01), make_screen("e", 1.00),
            make_screen("odd", 3.00),
        ]

    def test_an_absolute_concern_outranks_being_merely_unusual(self):
        cohort = self._cohort() + [make_screen("broken", 1.0, concerns=["fragmented"])]
        assert rank_cohort(cohort)[0].specimen == "broken"

    def test_a_cohort_outlier_is_surfaced(self):
        top = [r.specimen for r in rank_cohort(self._cohort())[:1]]
        assert top == ["odd"]

    def test_relative_concerns_are_suppressed_below_three_specimens(self):
        """Two points have a standard deviation but it means nothing."""
        pair = [make_screen("a", 1.0), make_screen("b", 9.0)]
        assert all(not r.relative_concerns for r in rank_cohort(pair))

    def test_a_blunt_boundary_is_surfaced_relative_to_the_cohort(self):
        cohort = [make_screen(s, 1.0, sharpness=0.10) for s in "abcde"]
        cohort.append(make_screen("blunt", 1.0, sharpness=0.01))
        assert rank_cohort(cohort)[0].specimen == "blunt"

    def test_an_empty_cohort_returns_nothing(self):
        assert rank_cohort([]) == []

    def test_ordering_is_stable_when_suspicion_ties(self):
        cohort = [make_screen(s, 1.0) for s in ("b", "a", "c")]
        assert [r.specimen for r in rank_cohort(cohort)] == ["a", "b", "c"]

    def test_a_clean_cohort_still_describes_each_specimen(self):
        for review in rank_cohort([make_screen(s, 1.0) for s in "abc"]):
            assert isinstance(review, CohortReview)
            assert "nothing unusual" in review.describe()


class TestRender:
    def test_it_writes_a_file(self, tmp_path):
        mask = ball()
        out = tmp_path / "panel.png"
        panel = render_panel(image_for(mask), vol(mask), out, specimen="s1")
        assert out.exists() and out.stat().st_size > 0
        assert panel.specimen == "s1"

    def test_several_labels_get_distinct_colours(self, tmp_path):
        mask, inner = ball(), ball(radius=9.0)
        panel = render_panel(
            image_for(mask), {"propagated": vol(mask), "refined": vol(inner)},
            tmp_path / "two.png",
        )
        assert panel.labels == ["propagated", "refined"]
        assert len(set(panel.colours.values())) == 2

    def test_slices_land_inside_the_label_extent(self, tmp_path):
        """A structure occupying part of the field must not be reviewed
        through empty tissue."""
        mask = np.zeros(SHAPE, dtype="uint8")
        mask[30:44, 10:38, 10:38] = ball(radius=14.0)[30:44, 10:38, 10:38]
        mask[30:44, 18:30, 18:30] = 1
        panel = render_panel(image_for(mask), vol(mask), tmp_path / "e.png")
        for index in panel.slices[0]:
            assert 30 <= index < 44

    def test_an_empty_label_is_refused(self, tmp_path):
        empty = vol(np.zeros(SHAPE, dtype="uint8"))
        with pytest.raises(ValueError, match="empty"):
            render_panel(image_for(ball()), empty, tmp_path / "x.png")

    def test_no_labels_at_all_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="at least one"):
            render_panel(image_for(ball()), {}, tmp_path / "x.png")

    def test_the_caption_names_the_labels_and_slices(self, tmp_path):
        mask = ball()
        panel = render_panel(image_for(mask), {"refined": vol(mask)},
                             tmp_path / "c.png", specimen="specimen-07")
        caption = panel.caption()
        assert "specimen-07" in caption and "refined" in caption
        assert "no anatomical reorientation" in caption

    def test_mismatched_geometry_is_refused(self, tmp_path):
        other = Volume(ball().astype("float32"), (10.0, 10.0, 10.0), "um",
                       AFFINE, "different")
        with pytest.raises(Exception):
            render_panel(image_for(ball()), other, tmp_path / "x.png")
