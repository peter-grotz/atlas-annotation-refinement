"""Tests for region-wise threshold fitting."""

from __future__ import annotations

import numpy as np
import pytest

from atlas_refine.algorithms import REGISTRY
from atlas_refine.algorithms.regional import shell_interior_partition
from atlas_refine.io.volumes import Volume

SHAPE = (40, 40, 40)
SPACING = (20.0, 20.0, 20.0)
AFFINE = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])

BACKGROUND = 1000.0
STRUCTURE = 2000.0
NEIGHBOUR = 3200.0          # brighter than the structure, like white matter


def vol(data: np.ndarray) -> Volume:
    return Volume(data.astype("float32"), SPACING, "um", AFFINE, "test")


def sandwich(structure_value: float = STRUCTURE):
    """A slab bordered by darker material below and brighter material above.

    This is the arrangement the family exists for: one boundary a threshold can
    act on, one it cannot.
    """
    data = np.full(SHAPE, BACKGROUND)
    data[26:, :, :] = NEIGHBOUR
    data[14:26, :, :] = structure_value
    mask = np.zeros(SHAPE, dtype="uint8")
    mask[14:26, :, :] = 1
    return vol(data), vol(mask)


def algorithm():
    return REGISTRY.create("regional_threshold")


class TestPartition:
    def test_a_slab_between_dark_and_bright_splits_in_two(self):
        image, label = sandwich()
        regions, names = shell_interior_partition(image, label, STRUCTURE, BACKGROUND)
        assert names == ["shell", "interior"]
        assert (regions == 1).sum() > 0
        assert (regions == 2).sum() > 0
        assert (regions > 0).sum() == int(label.data.sum())

    def test_the_shell_is_the_side_facing_the_darker_material(self):
        image, label = sandwich()
        regions, _ = shell_interior_partition(image, label, STRUCTURE, BACKGROUND)
        shell_z = np.where((regions == 1).any(axis=(1, 2)))[0]
        interior_z = np.where((regions == 2).any(axis=(1, 2)))[0]
        assert shell_z.mean() < interior_z.mean()   # dark side is at low z

    def test_a_structure_with_only_dark_neighbours_is_all_shell(self):
        data = np.full(SHAPE, BACKGROUND)
        data[14:26, 14:26, 14:26] = STRUCTURE
        mask = np.zeros(SHAPE, dtype="uint8")
        mask[14:26, 14:26, 14:26] = 1
        regions, _ = shell_interior_partition(vol(data), vol(mask), STRUCTURE, BACKGROUND)
        assert (regions == 2).sum() == 0
        assert (regions == 1).sum() == int(mask.sum())

    def test_an_empty_label_partitions_to_nothing(self):
        image, _ = sandwich()
        empty = vol(np.zeros(SHAPE, dtype="uint8"))
        regions, _ = shell_interior_partition(image, empty, STRUCTURE, BACKGROUND)
        assert regions.sum() == 0

    def test_the_partition_never_sees_an_annotation(self):
        """Region assignment depends only on the image and the propagated
        label. Were it to use the reference, every score against that reference
        would be inflated."""
        import inspect
        signature = inspect.signature(shell_interior_partition)
        assert list(signature.parameters) == ["image", "label", "interior", "background"]


class TestApplication:
    def test_it_declares_two_parameters(self):
        assert algorithm().n_parameters == 2

    def test_zero_leaves_a_region_untouched(self):
        image, label = sandwich()
        alg = algorithm()
        with alg.specimen_context(image, label):
            out = alg.apply(image, label, fraction_shell=0.0, fraction_interior=0.0)
        assert int(out.data.sum()) == int(label.data.sum())

    def test_the_regions_can_be_thresholded_independently(self):
        """A dim shell and a bright interior: thresholding only the shell must
        remove shell voxels and leave the interior whole."""
        data = np.full(SHAPE, BACKGROUND)
        data[26:, :, :] = NEIGHBOUR
        data[14:20, :, :] = 1200.0      # dim half of the structure
        data[20:26, :, :] = 2400.0      # bright half
        mask = np.zeros(SHAPE, dtype="uint8")
        mask[14:26, :, :] = 1
        image, label = vol(data), vol(mask)

        alg = algorithm()
        with alg.specimen_context(image, label):
            sizes = alg.region_sizes(image, label)
            trimmed = alg.apply(image, label, fraction_shell=0.4, fraction_interior=0.0)
            untouched = alg.apply(image, label, fraction_shell=0.0, fraction_interior=0.0)

        assert sizes["shell"] > 0 and sizes["interior"] > 0
        assert int(trimmed.data.sum()) < int(untouched.data.sum())

    def test_an_empty_label_yields_an_empty_result(self):
        image, _ = sandwich()
        empty = vol(np.zeros(SHAPE, dtype="uint8"))
        alg = algorithm()
        out = alg.apply(image, empty, fraction_shell=0.4, fraction_interior=0.4)
        assert int(out.data.sum()) == 0

    def test_the_result_is_binary_and_on_the_input_grid(self):
        image, label = sandwich()
        alg = algorithm()
        with alg.specimen_context(image, label):
            out = alg.apply(image, label, fraction_shell=0.3, fraction_interior=0.2)
        assert out.data.shape == label.data.shape
        assert set(np.unique(out.data)).issubset({0, 1})
        assert out.frame == label.frame

    def test_it_never_adds_voxels(self):
        """Every family here is subtractive; adding material would mean the
        label had grown outside what registration proposed."""
        image, label = sandwich()
        alg = algorithm()
        with alg.specimen_context(image, label):
            out = alg.apply(image, label, fraction_shell=0.3, fraction_interior=0.3)
        assert not (out.data.astype(bool) & ~label.data.astype(bool)).any()


class TestCaching:
    def test_the_partition_is_computed_once_per_specimen(self):
        image, label = sandwich()
        alg = algorithm()
        with alg.specimen_context(image, label):
            assert "regions" in alg.context
            first = alg.context["regions"]
            alg.apply(image, label, fraction_shell=0.2, fraction_interior=0.0)
            assert alg.context["regions"] is first

    def test_it_still_works_without_a_context(self):
        image, label = sandwich()
        out = algorithm().apply(image, label, fraction_shell=0.0, fraction_interior=0.0)
        assert int(out.data.sum()) == int(label.data.sum())


class TestBudget:
    def test_it_is_refused_with_only_one_independent_annotation(self):
        from atlas_refine.evaluate import AdmissibilityError, check_admissible

        with pytest.raises(AdmissibilityError):
            check_admissible(algorithm(), ["only-one"])

    def test_it_is_admissible_with_two(self):
        from atlas_refine.evaluate import check_admissible

        check_admissible(algorithm(), ["a", "b"])


class TestCaliberPartition:
    def test_a_structure_of_varying_width_splits_by_caliber(self):
        from atlas_refine.algorithms.regional import caliber_partition

        mask = np.zeros(SHAPE, dtype="uint8")
        mask[10:30, 10:30, 10:30] = 1          # thick block
        mask[10:30, 19:21, 32:38] = 1          # thin limb
        image = vol(np.full(SHAPE, BACKGROUND))
        regions, names = caliber_partition(image, vol(mask), STRUCTURE, BACKGROUND)
        assert names == ["thin", "thick"]
        assert (regions == 1).sum() > 0
        assert (regions == 2).sum() > 0
        assert (regions > 0).sum() == int(mask.sum())

    def test_the_thin_limb_is_entirely_thin(self):
        from atlas_refine.algorithms.regional import caliber_partition

        mask = np.zeros(SHAPE, dtype="uint8")
        mask[18:22, 18:22, 5:35] = 1           # a 4-voxel-wide tube
        image = vol(np.full(SHAPE, BACKGROUND))
        regions, _ = caliber_partition(image, vol(mask), STRUCTURE, BACKGROUND)
        assert (regions == 2).sum() == 0       # nothing reaches a real core

    def test_the_two_families_partition_differently(self):
        image, label = sandwich()
        shell = REGISTRY.create("regional_threshold")
        caliber = REGISTRY.create("caliber_threshold")
        with shell.specimen_context(image, label):
            a = shell.region_sizes(image, label)
        with caliber.specimen_context(image, label):
            b = caliber.region_sizes(image, label)
        assert set(a) == {"shell", "interior"}
        assert set(b) == {"thin", "thick"}

    def test_caliber_declares_its_own_parameter_names(self):
        names = [p.name for p in REGISTRY.create("caliber_threshold").parameters]
        assert names == ["fraction_thin", "fraction_thick"]
