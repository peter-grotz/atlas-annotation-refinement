"""Tests for the contribution gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atlas_refine.algorithms import REGISTRY
from atlas_refine.algorithms.base import Parameter, RefinementAlgorithm
from atlas_refine.algorithms.contribute import (
    ContributionRejected,
    Rationale,
    benchmark,
    load_candidate,
    scaffold,
    validate,
)
from atlas_refine.io.volumes import Volume


def make_image(shape=(12, 12, 12)) -> Volume:
    """Volume with a bright core inside a dimmer margin, above a background."""
    data = np.full(shape, 1000.0, dtype="float32")
    data[2:10, 2:10, 2:10] = 1400.0
    data[4:8, 4:8, 4:8] = 2000.0
    spacing = (20.0, 20.0, 20.0)
    return Volume(
        data=data,
        spacing=spacing,
        unit="um",
        affine=np.diag([-spacing[0], -spacing[1], spacing[2], 1.0]),
        frame="test",
    )


def make_label(image: Volume) -> Volume:
    mask = np.zeros(image.shape, dtype="uint8")
    mask[2:10, 2:10, 2:10] = 1
    return image.with_data(mask)


def make_reference(image: Volume) -> Volume:
    mask = np.zeros(image.shape, dtype="uint8")
    mask[4:8, 4:8, 4:8] = 1
    return image.with_data(mask)


def loader(_specimen: str):
    image = make_image()
    return image, make_label(image), make_reference(image)


class TestRationale:
    def test_requires_substantive_statements(self):
        with pytest.raises(ContributionRejected, match="substantive"):
            Rationale(
                target_signature="short",
                why_incumbents_insufficient="also short",
                limitations="short too",
            )

    def test_accepts_full_statements(self):
        rationale = Rationale(
            target_signature="Over-coverage forming a surface shell with intensity "
            "separation above 0.3 from the structure interior.",
            why_incumbents_insufficient="A plain threshold removes the shell but also "
            "erodes genuine low-intensity structure at the boundary.",
            limitations="Should not be used where the structure borders material of "
            "comparable intensity, since no separation exists to exploit.",
        )
        assert rationale.references == ()


class TestScaffold:
    def test_writes_a_template(self, tmp_path: Path):
        path = scaffold("my_family", tmp_path)
        assert path.exists()
        text = path.read_text()
        assert "RATIONALE" in text
        assert "my_family" in text
        assert "class MyFamily" in text

    def test_rejects_existing_registry_name(self, tmp_path: Path):
        with pytest.raises(ContributionRejected, match="already registered"):
            scaffold("contrast_threshold", tmp_path)

    def test_rejects_bad_name(self, tmp_path: Path):
        with pytest.raises(ContributionRejected, match="lowercase"):
            scaffold("MyFamily", tmp_path)

    def test_rejects_overwrite(self, tmp_path: Path):
        scaffold("my_family", tmp_path)
        with pytest.raises(ContributionRejected, match="already exists"):
            scaffold("my_family", tmp_path)

    def test_scaffolded_module_loads_and_is_importable(self, tmp_path: Path):
        path = scaffold("loadable_family", tmp_path)
        algorithm, rationale = load_candidate(path)
        assert algorithm.name == "loadable_family"
        assert isinstance(rationale, Rationale)


class TestValidation:
    def test_accepts_a_well_behaved_family(self):
        image = make_image()
        report = validate(REGISTRY.create("contrast_threshold"), image, make_label(image))
        assert report.passed, report.failures

    def test_rejects_a_family_that_mutates_its_input(self):
        class Mutating(RefinementAlgorithm):
            name = "mutating"
            description = "Writes into the caller's array, which corrupts shared state."

            @property
            def parameters(self):
                return (Parameter("x", [1.0], "unused"),)

            def apply(self, image, label, **params):
                label.data[0, 0, 0] = 1
                return label

        image = make_image()
        report = validate(Mutating(), image, make_label(image))
        assert not report.passed
        assert any("does_not_mutate_input" in f for f in report.failures)

    def test_rejects_a_non_binary_output(self):
        class NonBinary(RefinementAlgorithm):
            name = "non_binary"
            description = "Returns label values rather than a binary mask."

            @property
            def parameters(self):
                return ()

            def apply(self, image, label, **params):
                return label.with_data((label.data > 0).astype("uint8") * 3)

        image = make_image()
        report = validate(NonBinary(), image, make_label(image))
        assert not report.passed
        assert any("binary_output" in f for f in report.failures)

    def test_rejects_a_family_that_raises_on_an_empty_label(self):
        class Fragile(RefinementAlgorithm):
            name = "fragile"
            description = "Divides by the label size without guarding against zero."

            @property
            def parameters(self):
                return ()

            def apply(self, image, label, **params):
                mask = label.data > 0
                if not mask.any():
                    raise ValueError("empty label")
                return label.with_data(mask.astype("uint8"))

        image = make_image()
        report = validate(Fragile(), image, make_label(image))
        assert not report.passed
        assert any("handles_empty_label" in f for f in report.failures)

    def test_rejects_a_thin_description(self):
        class Terse(RefinementAlgorithm):
            name = "terse"
            description = "does stuff"

            @property
            def parameters(self):
                return ()

            def apply(self, image, label, **params):
                return label.with_data((label.data > 0).astype("uint8"))

        image = make_image()
        report = validate(Terse(), image, make_label(image))
        assert not report.passed
        assert any("has_description" in f for f in report.failures)


class TestBenchmark:
    def test_compares_against_admissible_incumbents_only(self):
        candidate = REGISTRY.create("contrast_threshold")
        report = benchmark(candidate, ["s1"], loader, incumbents=REGISTRY.names())
        # Two-parameter families are inadmissible on a single annotation.
        assert "band_threshold" not in report.incumbent_mean_dice
        assert "morphological_cleanup" not in report.incumbent_mean_dice

    def test_reports_improvement_against_the_best_incumbent(self):
        candidate = REGISTRY.create("contrast_threshold")
        report = benchmark(candidate, ["s1", "s2"], loader)
        assert report.candidate == "contrast_threshold"
        assert report.best_incumbent is not None
        assert report.improvement == pytest.approx(
            report.candidate_mean_dice - report.incumbent_mean_dice[report.best_incumbent]
        )

    def test_a_family_duplicating_an_incumbent_does_not_qualify(self):
        """A new family must beat the incumbents, not merely equal them."""
        from atlas_refine.algorithms.intensity import ContrastNormalisedThreshold

        class Duplicate(ContrastNormalisedThreshold):
            name = "duplicate_of_contrast_threshold"
            description = "Behaviourally identical to an existing family."

        report = benchmark(Duplicate(), ["s1", "s2"], loader,
                           incumbents=["contrast_threshold"])
        assert report.improvement == pytest.approx(0.0, abs=1e-9)
        assert not report.qualifies

    def test_a_strictly_better_family_qualifies(self):
        """A family that recovers the reference exactly clears the margin."""

        class Oracle(RefinementAlgorithm):
            name = "oracle_for_test"
            description = "Selects the brightest tier, which is the reference here."

            @property
            def parameters(self):
                return (Parameter("cut", [1700.0], "Absolute cut, test only."),)

            def apply(self, image, label, **params):
                keep = (label.data > 0) & (image.data > params["cut"])
                return label.with_data(keep.astype("uint8"))

        report = benchmark(Oracle(), ["s1", "s2"], loader,
                           incumbents=["contrast_threshold"])
        assert report.candidate_mean_dice == pytest.approx(1.0)
        assert report.improvement > 0
        assert report.qualifies
