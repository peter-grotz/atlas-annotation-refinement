"""Tests for registration and label propagation.

Small synthetic volumes with a known displacement between them, so the recovered
transform can be checked against the truth rather than only for plausibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

ants = pytest.importorskip("ants")

from atlas_refine.io.volumes import Volume  # noqa: E402
from atlas_refine.registration import (  # noqa: E402
    MetricTerm,
    Registration,
    RegistrationConfig,
    propagate_labels,
    register,
)

SHAPE = (32, 32, 32)
SPACING = (20.0, 20.0, 20.0)
AFFINE = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])
BACKGROUND, SIGNAL = 1000.0, 2000.0
#: Phantom registration accuracy is sensitive to contrast in ways that do not
#: reflect real acquisitions: at 32 voxels a side there are few informative
#: samples per resolution level, and raising the signal can worsen the fit.
#: The assertions below therefore check that alignment improves substantially,
#: which is the property that matters, rather than pinning an exact recovery.
SHIFT = 3

#: Few iterations: these tests check that the machinery is wired correctly and
#: recovers a coarse displacement, not that convergence is tight.
FAST = RegistrationConfig(iterations=(10, 5, 0))


def volume(data: np.ndarray, frame: str) -> Volume:
    return Volume(data.astype("float32"), SPACING, "um", AFFINE, frame)


#: A smooth textured body filling most of the field, as an acquisition does.
#: Registration is driven by image content, so a small object in a large empty
#: volume is an adversarial test rather than a representative one: its interior
#: carries no gradient and the metric has few informative samples.
_N = SHAPE[0]
_RNG = np.random.default_rng(0)
_FIELD = ndi.gaussian_filter(_RNG.normal(0, 1, SHAPE), 2.0)
_FIELD = SIGNAL + 900.0 * (_FIELD - _FIELD.mean()) / _FIELD.std()
_ZZ, _YY, _XX = np.mgrid[0:_N, 0:_N, 0:_N]


def _ellipsoid(offset: int) -> np.ndarray:
    return (
        ((_ZZ - (15 + offset)) / 11.0) ** 2
        + ((_YY - 15) / 9.0) ** 2
        + ((_XX - 15) / 9.0) ** 2
    ) <= 1


def cube(offset: int = 0) -> np.ndarray:
    """Textured body at a given displacement along the first axis."""
    data = np.full(SHAPE, BACKGROUND, dtype="float32")
    interior = _ellipsoid(offset)
    data[interior] = np.roll(_FIELD, offset, axis=0)[interior]
    return data


def cube_mask(offset: int = 0, value: int = 1) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype="uint32")
    mask[_ellipsoid(offset)] = value
    return mask


def centroid(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask > 0)
    return points.mean(axis=0) if points.size else np.array([np.nan] * 3)


@pytest.fixture
def pair() -> tuple[Volume, Volume]:
    """Source and target differing by a known translation along the first axis."""
    return volume(cube(0), "source"), volume(cube(SHIFT), "target")


@pytest.fixture
def registration(pair) -> Registration:
    source, target = pair
    return register(moving=source, fixed=target, config=FAST)


class TestRegister:
    def test_records_both_frames(self, pair, registration):
        source, target = pair
        assert registration.source_frame == source.frame
        assert registration.target_frame == target.frame

    def test_returns_forward_and_inverse_chains(self, registration):
        assert registration.forward and registration.inverse
        assert all(Path(p).exists() for p in registration.forward)

    def test_recovers_a_known_translation(self, pair, registration):
        """The whole method rests on this: a label moved through the transform
        must land where the target's anatomy is."""
        source, target = pair
        warped = ants.apply_transforms(
            fixed=ants.from_numpy(target.data, spacing=SPACING),
            moving=ants.from_numpy(source.data, spacing=SPACING),
            transformlist=registration.forward,
        ).numpy()
        threshold = (BACKGROUND + SIGNAL) / 2
        before = abs(centroid(source.data > threshold)[0] - centroid(target.data > threshold)[0])
        after = abs(centroid(warped > threshold)[0] - centroid(target.data > threshold)[0])
        assert after < before / 2, f"displacement {before:.2f} -> {after:.2f}, not improved"
        assert after < 1.0

    def test_is_reproducible(self, pair):
        """Repeated runs on identical inputs must agree.

        The similarity metric samples voxels stochastically and the reduction is
        multithreaded, so neither a seed argument nor default settings are
        sufficient on their own.
        """
        source, target = pair
        first = register(moving=source, fixed=target, config=FAST)
        second = register(moving=source, fixed=target, config=FAST)

        def warped(reg: Registration) -> np.ndarray:
            return ants.apply_transforms(
                fixed=ants.from_numpy(target.data, spacing=SPACING),
                moving=ants.from_numpy(source.data, spacing=SPACING),
                transformlist=reg.forward,
            ).numpy()

        assert np.allclose(warped(first), warped(second))

    def test_background_subtraction_does_not_change_the_geometry(self, pair):
        """The offset is removed from both volumes, so it must not shift the
        recovered transform."""
        source, target = pair
        with_subtraction = register(moving=source, fixed=target, config=FAST)
        without = register(
            moving=source, fixed=target, config=FAST, subtract_background=False
        )
        def warped_centroid(reg: Registration) -> float:
            out = ants.apply_transforms(
                fixed=ants.from_numpy(target.data, spacing=SPACING),
                moving=ants.from_numpy(source.data, spacing=SPACING),
                transformlist=reg.forward,
            ).numpy()
            return centroid(out > (BACKGROUND + SIGNAL) / 2)[0]

        assert warped_centroid(with_subtraction) == pytest.approx(
            warped_centroid(without), abs=1.5
        )

    def test_accepts_an_additional_metric_term(self, pair):
        source, target = pair
        term = MetricTerm(
            fixed=volume(cube_mask(SHIFT).astype("float32"), "target"),
            moving=volume(cube_mask(0).astype("float32"), "source"),
            weight=1.0,
        )
        result = register(moving=source, fixed=target, extra_metrics=[term], config=FAST)
        assert result.forward


class TestPropagateLabels:
    def test_places_labels_at_the_target_anatomy(self, pair, registration):
        """Overlap with the target's own label must improve substantially.

        Overlap is a stricter check than centroid agreement: a transform can
        centre a label correctly while distorting its extent.
        """
        source, target = pair
        labels = Volume(cube_mask(0), SPACING, "um", AFFINE, "source")
        truth = cube_mask(SHIFT) > 0
        warped = propagate_labels(labels, reference=target, registration=registration)

        def dice(a, b):
            return 2 * int((a & b).sum()) / (int(a.sum()) + int(b.sum()))

        before = dice(cube_mask(0) > 0, truth)
        after = dice(warped.data > 0, truth)
        assert after > before, f"overlap {before:.3f} -> {after:.3f}, not improved"
        assert after > 0.85

    def test_output_adopts_the_reference_grid_and_frame(self, pair, registration):
        source, target = pair
        labels = Volume(cube_mask(0), SPACING, "um", AFFINE, "source")
        warped = propagate_labels(labels, reference=target, registration=registration)
        assert warped.frame == target.frame
        assert warped.shape == target.shape
        assert np.allclose(warped.affine, target.affine)

    def test_label_values_stay_discrete(self, pair, registration):
        """Linear interpolation of a label image invents values corresponding to
        no label, so propagation must not introduce any."""
        source, target = pair
        labels = Volume(cube_mask(0, value=7), SPACING, "um", AFFINE, "source")
        warped = propagate_labels(labels, reference=target, registration=registration)
        assert set(np.unique(warped.data).tolist()) <= {0, 7}

    def test_multiple_labels_are_preserved(self, pair, registration):
        source, target = pair
        data = cube_mask(0, value=1)
        data[(data > 0) & (_YY < 15)] = 2
        labels = Volume(data, SPACING, "um", AFFINE, "source")
        warped = propagate_labels(labels, reference=target, registration=registration)
        assert set(np.unique(warped.data).tolist()) <= {0, 1, 2}
        assert (warped.data == 2).any()

    def test_values_filter_retains_only_the_requested_labels(self, pair, registration):
        source, target = pair
        data = cube_mask(0, value=1)
        data[(data > 0) & (_YY < 15)] = 2
        labels = Volume(data, SPACING, "um", AFFINE, "source")
        warped = propagate_labels(labels, reference=target, registration=registration, values=[2])
        assert set(np.unique(warped.data).tolist()) <= {0, 2}

    def test_rejects_labels_from_the_wrong_frame(self, pair, registration):
        _, target = pair
        stray = Volume(cube_mask(0), SPACING, "um", AFFINE, "some-other-frame")
        with pytest.raises(ValueError, match="frame"):
            propagate_labels(stray, reference=target, registration=registration)


class TestSave:
    def test_writes_forward_and_inverse_transforms(self, registration, tmp_path: Path):
        saved = registration.save(tmp_path, prefix="spec")
        written = sorted(p.name for p in tmp_path.iterdir())
        assert written, "no transform files were written"
        assert any("forward" in name for name in written)
        assert any("inverse" in name for name in written)
        assert set(saved) == {"forward", "inverse"}

    def test_every_transform_in_each_chain_is_recoverable(self, registration, tmp_path: Path):
        """The returned mapping must let a caller rebuild the transform list.

        A chain is typically a warp field plus an affine; recording only one of
        them makes the saved registration unusable.
        """
        saved = registration.save(tmp_path, prefix="spec")
        for label, chain in (("forward", registration.forward), ("inverse", registration.inverse)):
            assert len(saved[label]) == len(chain), (
                f"{label} chain has {len(chain)} transform(s) but "
                f"{len(saved[label])} were reported as saved"
            )
            assert all(Path(p).exists() for p in saved[label])
