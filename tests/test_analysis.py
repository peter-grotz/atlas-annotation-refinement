"""Tests for image description, cohort comparison, boundary profiling, anatomy."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from atlas_refine.analysis import (
    KNOWN,
    NeighbourRelation,
    StructureContext,
    associate_with_parameter,
    context_for,
    describe_structure,
    next_to_annotate,
    otsu_separability,
    position_in_cohort,
    profile_boundary,
    profile_image,
    screen_operations,
    screening_brief,
)
from atlas_refine.io.volumes import Volume

SHAPE = (32, 32, 32)
SPACING = (20.0, 20.0, 20.0)
AFFINE = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])


def volume(data: np.ndarray, frame: str = "test") -> Volume:
    return Volume(data.astype("float32"), SPACING, "um", AFFINE, frame)


def specimen(contrast: float = 2.0, shading: float = 0.0, noise: float = 0.0,
             seed: int = 0) -> Volume:
    """A bright body on a background, with optional shading and noise."""
    rng = np.random.default_rng(seed)
    data = np.full(SHAPE, 1000.0)
    zz, yy, xx = np.mgrid[0:32, 0:32, 0:32]
    body = ((zz - 16) / 10.0) ** 2 + ((yy - 16) / 9.0) ** 2 + ((xx - 16) / 9.0) ** 2 <= 1
    data[body] = 1000.0 * contrast
    if shading:
        data *= 1.0 + shading * (zz / 31.0 - 0.5)
    if noise:
        data += rng.normal(0, noise * 1000.0, SHAPE)
    return volume(data)


def body_mask(radius_scale: float = 1.0) -> np.ndarray:
    zz, yy, xx = np.mgrid[0:32, 0:32, 0:32]
    r = ((zz - 16) / (10.0 * radius_scale)) ** 2 + ((yy - 16) / (9.0 * radius_scale)) ** 2 \
        + ((xx - 16) / (9.0 * radius_scale)) ** 2
    return (r <= 1).astype("uint8")


class TestImageProfile:
    def test_reports_contrast(self):
        assert profile_image(specimen(contrast=2.0)).contrast == pytest.approx(2.0, abs=0.15)

    def test_higher_contrast_reads_higher(self):
        low = profile_image(specimen(contrast=1.4)).contrast
        high = profile_image(specimen(contrast=2.5)).contrast
        assert high > low

    def test_shading_is_detected(self):
        flat = profile_image(specimen(shading=0.0)).shading
        shaded = profile_image(specimen(shading=0.4)).shading
        assert shaded > flat

    def test_noise_is_detected(self):
        clean = profile_image(specimen(noise=0.0)).noise
        noisy = profile_image(specimen(noise=0.15)).noise
        assert noisy > clean

    def test_tissue_fraction_is_a_proportion(self):
        assert 0.0 < profile_image(specimen()).tissue_fraction < 1.0

    def test_rejects_a_volume_with_no_tissue(self):
        with pytest.raises(ValueError, match="no tissue"):
            profile_image(volume(np.full(SHAPE, 1000.0)))


class TestSeparability:
    def test_two_clean_classes_score_high(self):
        values = np.concatenate([np.full(500, 1000.0), np.full(500, 3000.0)])
        assert otsu_separability(values) > 0.9

    def test_a_single_class_scores_below_two_clean_classes(self):
        """A unimodal distribution still yields a best split explaining roughly
        two thirds of its variance, so the meaningful comparison is relative."""
        rng = np.random.default_rng(0)
        unimodal = otsu_separability(rng.normal(2000, 100, 1000))
        separated = otsu_separability(
            np.concatenate([np.full(500, 1000.0), np.full(500, 3000.0)])
        )
        assert unimodal < separated
        assert separated > 0.9

    def test_is_bounded(self):
        rng = np.random.default_rng(0)
        for values in (np.full(100, 5.0), rng.normal(0, 1, 500),
                       np.concatenate([np.zeros(50), np.ones(50)])):
            score = otsu_separability(values)
            assert np.isnan(score) or 0.0 <= score <= 1.0


class TestCohort:
    def _cohort(self):
        return {
            "a": profile_image(specimen(contrast=2.0, seed=1)),
            "b": profile_image(specimen(contrast=2.05, seed=2)),
            "c": profile_image(specimen(contrast=2.1, seed=3)),
            "odd": profile_image(specimen(contrast=1.3, shading=0.5, seed=4)),
        }

    def test_identifies_the_unusual_specimen(self):
        assert position_in_cohort(self._cohort())[0].specimen == "odd"

    def test_orders_by_deviation(self):
        positions = position_in_cohort(self._cohort())
        assert positions[0].deviation >= positions[-1].deviation

    def test_names_the_responsible_field(self):
        assert position_in_cohort(self._cohort())[0].driven_by

    def test_next_to_annotate_avoids_what_is_done(self):
        chosen = next_to_annotate(self._cohort(), annotated=["a", "b"])
        assert chosen.specimen not in ("a", "b")

    def test_next_to_annotate_prefers_the_furthest(self):
        assert next_to_annotate(self._cohort(), annotated=["a", "b"]).specimen == "odd"

    def test_with_nothing_annotated_takes_the_most_atypical(self):
        assert next_to_annotate(self._cohort(), annotated=[]).specimen == "odd"

    def test_returns_none_when_all_are_annotated(self):
        cohort = self._cohort()
        assert next_to_annotate(cohort, annotated=list(cohort)) is None

    def test_needs_at_least_two_specimens(self):
        with pytest.raises(ValueError, match="at least two"):
            position_in_cohort({"a": profile_image(specimen())})


class TestAssociation:
    def test_small_samples_are_marked_uninterpretable(self):
        profiles = {"a": profile_image(specimen(contrast=1.5, seed=1)),
                    "b": profile_image(specimen(contrast=2.5, seed=2))}
        for association in associate_with_parameter(profiles, {"a": 0.3, "b": 0.5}):
            assert association.n == 2
            assert not association.interpretable
            assert "not \ninterpretable" in association.describe().replace(" ", " ") or \
                   "not interpretable" in association.describe().replace("\n", " ")

    def test_needs_two_specimens_in_common(self):
        profiles = {"a": profile_image(specimen())}
        with pytest.raises(ValueError, match="at least two"):
            associate_with_parameter(profiles, {"a": 0.3})


class TestBoundaryProfile:
    def test_rises_across_a_boundary_into_brighter_tissue(self):
        """A dark structure inside bright tissue: intensity rises going outward."""
        data = np.full(SHAPE, 3000.0)
        mask = body_mask()
        data[mask > 0] = 1000.0
        profile = profile_boundary(volume(data), volume(mask), min_voxels=20)
        assert profile.sharpness_at_boundary > 0
        assert profile.is_placeable_by_intensity

    def test_polarity_is_carried_by_the_step_not_the_sharpness(self):
        """Sharpness is a magnitude; the direction of the step is separate."""
        bright_inside = np.full(SHAPE, 1000.0)
        mask = body_mask()
        bright_inside[mask > 0] = 3000.0
        dark_inside = np.full(SHAPE, 3000.0)
        dark_inside[mask > 0] = 1000.0

        bright = profile_boundary(volume(bright_inside), volume(mask), min_voxels=20)
        dark = profile_boundary(volume(dark_inside), volume(mask), min_voxels=20)

        assert bright.sharpness_at_boundary > 0
        assert dark.sharpness_at_boundary > 0
        assert bright.contrast_across_boundary < 0      # outside is darker
        assert dark.contrast_across_boundary > 0        # outside is brighter

    def test_a_uniform_volume_is_not_placeable(self):
        """No intensity difference across the boundary means no threshold can
        place it, whatever the parameter."""
        profile = profile_boundary(volume(np.full(SHAPE, 2000.0)),
                                   volume(body_mask()), min_voxels=20)
        assert not profile.is_placeable_by_intensity

    def test_normalisation_is_polarity_agnostic(self):
        """Both polarities normalise to 0 inside and 1 outside."""
        for inside, outside in ((1000.0, 3000.0), (3000.0, 1000.0)):
            data = np.full(SHAPE, outside)
            mask = body_mask()
            data[mask > 0] = inside
            profile = profile_boundary(volume(data), volume(mask), min_voxels=20)
            reported = [v for v in profile.intensity if v is not None]
            assert min(reported) == pytest.approx(0.0, abs=0.35)
            assert max(reported) == pytest.approx(1.0, abs=0.35)

    def test_rejects_an_empty_label(self):
        with pytest.raises(ValueError, match="empty"):
            profile_boundary(specimen(), volume(np.zeros(SHAPE, dtype="uint8")))


class TestAnatomy:
    def test_a_brighter_neighbour_rules_out_expansion(self):
        """The finding that expansion admits white matter, encoded."""
        screen = screen_operations(context_for("isocortex"))
        assert "dilate_then_threshold" in screen.ruled_out
        assert "brighter" in screen.ruled_out["dilate_then_threshold"]

    def test_opposite_polarity_boundaries_support_a_band(self):
        assert "band_threshold" in screen_operations(context_for("isocortex")).supported

    def test_a_single_boundary_cautions_against_a_band(self):
        screen = screen_operations(context_for("ventricles"))
        assert "band_threshold" not in screen.supported
        assert any("second parameter" in c for c in screen.cautions)

    def test_indistinguishable_neighbours_rule_out_thresholding(self):
        context = StructureContext(
            name="embedded",
            boundaries=(NeighbourRelation("surrounding tissue", "similar"),),
        )
        screen = screen_operations(context)
        assert "threshold" in screen.ruled_out
        assert any("registration-limited" in c for c in screen.cautions)

    def test_a_thin_structure_cautions_against_erosion(self):
        context = StructureContext(
            name="thin",
            boundaries=(NeighbourRelation("tissue", "brighter"),),
            typical_thickness=3,
        )
        assert any("erosion will" in c for c in screen_operations(context).cautions)

    def test_an_undescribed_structure_gives_guidance_rather_than_failing(self):
        brief = screening_brief("not_yet_described")
        assert "has not been described" in brief
        assert "describe_structure" in brief

    def test_describing_a_structure_registers_it(self):
        name = "test_only_structure"
        KNOWN.pop(name, None)
        assert context_for(name) is None
        describe_structure(
            name,
            [NeighbourRelation("neighbour", "darker", source="measured")],
            typical_thickness=10,
        )
        try:
            assert context_for(name) is not None
            assert "threshold" in screen_operations(context_for(name)).supported
        finally:
            KNOWN.pop(name, None)

    def test_every_known_relation_states_its_source(self):
        """A relation without provenance cannot be checked when it misleads."""
        for context in KNOWN.values():
            for boundary in context.boundaries:
                assert boundary.source != "unspecified", context.name
