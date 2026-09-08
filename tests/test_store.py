"""Tests for filename parsing, provenance, and admissibility rules."""

from __future__ import annotations

import numpy as np
import pytest

from atlas_refine.algorithms import REGISTRY
from atlas_refine.evaluate import AdmissibilityError, check_admissible, dice, scores
from atlas_refine.io.store import BASIS_KIND, IntakeError, parse_filename
from atlas_refine.io.volumes import GeometryError, Volume


def make_volume(shape=(8, 8, 8), value=1.0, frame="specimen", spacing=(20.0, 20.0, 20.0)):
    return Volume(
        data=np.full(shape, value, dtype="float32"),
        spacing=spacing,
        unit="um",
        affine=np.diag([-spacing[0], -spacing[1], spacing[2], 1.0]),
        frame=frame,
    )


class TestFilename:
    def test_parses_all_fields(self):
        fields = parse_filename("100001__isocortex__pushed__AB__2026-01-15.seg.nrrd")
        assert fields == {
            "specimen": "100001",
            "structure": "isocortex",
            "basis": "pushed",
            "annotator": "AB",
            "annotated": "2026-01-15",
        }

    def test_allows_underscores_inside_structure(self):
        fields = parse_filename("100001__cerebellar_cortex__scratch__ABC__2026-01-15.seg.nrrd")
        assert fields["structure"] == "cerebellar_cortex"
        assert fields["basis"] == "scratch"

    @pytest.mark.parametrize(
        "name",
        [
            "100001_isocortex_pushed_AB_2026-01-15.seg.nrrd",  # single separators
            "100001__isocortex__guessed__AB__2026-01-15.seg.nrrd",  # invalid basis
            "100001__isocortex__pushed__AB__15-01-2026.seg.nrrd",  # wrong date order
            "100001__isocortex__pushed__AB.seg.nrrd",  # missing date
            "100001__isocortex__pushed__AB__2026-01-15.nrrd",  # wrong extension
        ],
    )
    def test_rejects_malformed(self, name):
        with pytest.raises(IntakeError):
            parse_filename(name)


class TestBasisKind:
    def test_editing_an_algorithm_output_is_derived(self):
        assert BASIS_KIND["refined"] == "derived"

    @pytest.mark.parametrize("basis", ["pushed", "scratch"])
    def test_other_bases_are_independent(self, basis):
        assert BASIS_KIND[basis] == "independent"


class TestVolume:
    def test_rejects_implausible_spacing(self):
        with pytest.raises(GeometryError, match="plausible"):
            make_volume(spacing=(0.02, 0.02, 0.02))

    def test_unit_conversion_preserves_physical_size(self):
        micron = make_volume(spacing=(80.0, 80.0, 80.0))
        millimetre = micron.to_unit("mm")
        assert millimetre.unit == "mm"
        assert np.allclose(millimetre.spacing, (0.08, 0.08, 0.08))
        assert np.allclose(micron.spacing_um, millimetre.spacing_um)

    def test_rejects_frame_mismatch(self):
        a, b = make_volume(frame="specimen"), make_volume(frame="atlas")
        with pytest.raises(GeometryError, match="frame mismatch"):
            a.assert_compatible(b)

    def test_allows_frame_mismatch_when_disabled(self):
        a, b = make_volume(frame="specimen"), make_volume(frame="atlas")
        a.assert_compatible(b, same_frame=False)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(GeometryError, match="shape mismatch"):
            make_volume((8, 8, 8)).assert_compatible(make_volume((8, 8, 9)))

    def test_voxel_volume(self):
        volume = make_volume(spacing=(100.0, 100.0, 100.0))
        assert volume.voxel_volume_mm3 == pytest.approx(1e-3)


class TestMetrics:
    def test_dice_of_identical_masks(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        assert dice(mask, mask) == pytest.approx(1.0)

    def test_dice_of_disjoint_masks(self):
        a = np.zeros((4, 4, 4), dtype=bool)
        b = np.zeros((4, 4, 4), dtype=bool)
        a[0, 0, 0] = True
        b[3, 3, 3] = True
        assert dice(a, b) == 0.0

    def test_empty_inputs_do_not_raise(self):
        empty = np.zeros((4, 4, 4), dtype=bool)
        assert dice(empty, empty) == 0.0
        assert scores(empty, empty)["precision"] == 0.0

    def test_recall_and_precision_directions(self):
        truth = np.zeros((4, 4, 4), dtype=bool)
        truth[:2] = True
        over = np.ones((4, 4, 4), dtype=bool)
        result = scores(over, truth)
        assert result["recall"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(0.5)


class TestAdmissibility:
    def test_rejects_more_parameters_than_annotations(self):
        algorithm = REGISTRY.create("band_threshold")  # two parameters
        with pytest.raises(AdmissibilityError, match="free parameters"):
            check_admissible(algorithm, ["100001"])

    def test_accepts_when_annotations_suffice(self):
        algorithm = REGISTRY.create("band_threshold")
        check_admissible(algorithm, ["100001", "100002"])

    def test_rejects_empty_annotation_set(self):
        algorithm = REGISTRY.create("contrast_threshold")
        with pytest.raises(AdmissibilityError, match="no independent"):
            check_admissible(algorithm, [])

    def test_single_parameter_family_needs_one_annotation(self):
        algorithm = REGISTRY.create("contrast_threshold")
        check_admissible(algorithm, ["100001"])


class TestRegistry:
    def test_built_ins_are_registered(self):
        names = REGISTRY.names()
        assert {"contrast_threshold", "band_threshold", "hysteresis_threshold"} <= set(names)

    def test_unknown_algorithm_lists_alternatives(self):
        with pytest.raises(KeyError, match="Available"):
            REGISTRY.create("does_not_exist")

    def test_grid_size_matches_parameter_product(self):
        algorithm = REGISTRY.create("band_threshold")
        expected = 1
        for parameter in algorithm.parameters:
            expected *= len(parameter.values)
        assert len(list(algorithm.grid())) == expected

    def test_defaults_lie_inside_the_search_space(self):
        for name in REGISTRY.names():
            algorithm = REGISTRY.create(name)
            defaults = algorithm.defaults()
            for parameter in algorithm.parameters:
                assert min(parameter.values) <= defaults[parameter.name] <= max(parameter.values)
