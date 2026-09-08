"""Atlas-to-specimen registration and label propagation.

Registration proceeds in two stages. An affine is solved first on image
intensity alone and without spatial masking, then supplied as the initialisation
for a deformable stage. Deformable optimisation in ANTs does not itself estimate
an affine, and restricting the affine stage to masked regions removes most of
the samples that constrain it, so both are handled explicitly.

Where manual annotations exist for a structure, they can be supplied as
additional metric terms. This constrains the deformation using anatomy that
intensity alone does not localise well, and improves label placement for the
whole cohort rather than correcting each label afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import ants
import numpy as np

from ..io.volumes import Volume, background_level


@dataclass(frozen=True)
class MetricTerm:
    """An additional similarity term evaluated during the deformable stage.

    Attributes:
        fixed: Volume in the target frame.
        moving: Corresponding volume in the source frame.
        weight: Relative contribution against the image metric, which is 1.0.
        metric: ANTs metric name.
    """

    fixed: Volume
    moving: Volume
    weight: float = 1.0
    metric: str = "MeanSquares"


@dataclass(frozen=True)
class RegistrationConfig:
    transform: str = "SyNOnly"
    iterations: tuple[int, ...] = (40, 20, 10)
    gradient_step: float = 0.2
    flow_sigma: float = 3.0
    total_sigma: float = 0.0
    image_metric: str = "mattes"


@dataclass(frozen=True)
class Registration:
    """Transforms mapping the source frame onto the target frame."""

    forward: list[str]
    inverse: list[str]
    target_frame: str
    source_frame: str

    def save(self, directory: str | Path, prefix: str) -> dict[str, str]:
        import shutil

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        saved: dict[str, str] = {}
        for label, paths in (("forward", self.forward), ("inverse", self.inverse)):
            for path in paths:
                name = f"{prefix}_{label}_{Path(path).name}"
                shutil.copy2(path, directory / name)
                saved.setdefault(label, str(directory / name))
        return saved


def _to_ants(volume: Volume) -> ants.ANTsImage:
    """Wrap array data with explicit spacing.

    Constructed from the array rather than read from disk so that spacing comes
    from the validated :class:`Volume` and not from header metadata.
    """
    return ants.from_numpy(np.ascontiguousarray(volume.data.astype("float32")),
                           spacing=tuple(float(s) for s in volume.spacing))


def register(
    moving: Volume,
    fixed: Volume,
    *,
    extra_metrics: Sequence[MetricTerm] = (),
    config: RegistrationConfig | None = None,
    subtract_background: bool = True,
) -> Registration:
    """Register ``moving`` onto ``fixed``.

    Args:
        moving: Source volume, typically an atlas template.
        fixed: Target volume, typically a specimen acquisition.
        extra_metrics: Additional similarity terms for the deformable stage.
        config: Deformable stage settings.
        subtract_background: Offset both volumes by their modal background so
            the intensity metric is not dominated by the background plateau.

    Returns:
        The forward and inverse transform chains.
    """
    config = config or RegistrationConfig()

    def prepare(volume: Volume) -> ants.ANTsImage:
        data = volume.data
        if subtract_background:
            data = np.clip(data - background_level(volume), 0, None)
        return _to_ants(volume.with_data(data))

    fixed_image, moving_image = prepare(fixed), prepare(moving)

    affine = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform="Affine")

    multivariate = [
        (term.metric, _to_ants(term.fixed), _to_ants(term.moving), float(term.weight), 0)
        for term in extra_metrics
    ]

    result = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=config.transform,
        initial_transform=affine["fwdtransforms"][0],
        multivariate_extras=multivariate or None,
        reg_iterations=config.iterations,
        grad_step=config.gradient_step,
        flow_sigma=config.flow_sigma,
        total_sigma=config.total_sigma,
        aff_metric=config.image_metric,
    )
    return Registration(
        forward=list(result["fwdtransforms"]),
        inverse=list(result["invtransforms"]),
        target_frame=fixed.frame,
        source_frame=moving.frame,
    )


def propagate_labels(
    labels: Volume,
    reference: Volume,
    registration: Registration,
    *,
    values: Iterable[int] | None = None,
) -> Volume:
    """Resample a label volume into the reference frame.

    The full transform chain is applied in a single resampling step, and nearest
    equivalent interpolation is used throughout. Interpolating label values
    linearly produces intensities that correspond to no label.

    Args:
        labels: Label volume in the registration's source frame.
        reference: Volume defining the output grid.
        registration: Transforms produced by :func:`register`.
        values: Label values to retain. All non-zero values are kept if omitted.
    """
    if labels.frame != registration.source_frame:
        raise ValueError(
            f"labels are in frame '{labels.frame}' but the registration maps from "
            f"'{registration.source_frame}'"
        )
    data = labels.data
    if values is not None:
        keep = np.isin(data, list(values))
        data = np.where(keep, data, 0)

    warped = ants.apply_transforms(
        fixed=_to_ants(reference),
        moving=_to_ants(labels.with_data(data.astype("float32"))),
        transformlist=registration.forward,
        interpolator="genericLabel",
    ).numpy()

    return Volume(
        data=np.rint(warped).astype("uint32"),
        spacing=reference.spacing,
        unit=reference.unit,
        affine=reference.affine,
        frame=reference.frame,
    )
