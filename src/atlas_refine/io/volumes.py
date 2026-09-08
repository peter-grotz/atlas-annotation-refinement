"""Volume container with explicit units and coordinate frame.

Voxel spacing carries a unit and every volume declares the frame its array axes
are expressed in. Both are stored rather than inferred, and operations that
combine volumes assert compatibility. Header metadata in NIfTI and NRRD is
frequently inconsistent with the stored voxel sizes; silently trusting it
produces geometrically wrong output without raising.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import nibabel as nib
import numpy as np
import nrrd

Unit = Literal["um", "mm"]
_TO_UM = {"um": 1.0, "mm": 1000.0}

#: Header unit codes are unreliable across writers. A spacing outside this range
#: for the declared unit indicates the header disagrees with the data.
_PLAUSIBLE_UM = (0.05, 500.0)


class GeometryError(ValueError):
    """Raised when volume geometry is missing, implausible, or incompatible."""


@dataclass(frozen=True)
class Volume:
    """A 3-D array with physical spacing, a declared unit, and an affine.

    Attributes:
        data: Array of shape ``(x, y, z)``.
        spacing: Voxel size along each axis, in ``unit``.
        unit: Physical unit of ``spacing``.
        affine: 4x4 voxel-to-world matrix in RAS.
        frame: Free-form name of the coordinate frame, used to prevent
            accidental combination of volumes that are not co-registered.
    """

    data: np.ndarray
    spacing: tuple[float, float, float]
    unit: Unit
    affine: np.ndarray
    frame: str

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise GeometryError(f"expected a 3-D array, got shape {self.data.shape}")
        if len(self.spacing) != 3:
            raise GeometryError(f"expected 3 spacing values, got {self.spacing}")
        lo, hi = _PLAUSIBLE_UM
        um = self.spacing_um
        if not all(lo <= s <= hi for s in um):
            raise GeometryError(
                f"spacing {self.spacing} {self.unit} is {um} um, outside the plausible "
                f"range {lo}-{hi} um. The header unit is likely wrong; pass an explicit "
                f"`spacing` and `unit` when loading."
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def spacing_um(self) -> tuple[float, float, float]:
        f = _TO_UM[self.unit]
        return tuple(s * f for s in self.spacing)  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing_um)) / 1e9

    def to_unit(self, unit: Unit) -> "Volume":
        if unit == self.unit:
            return self
        f = _TO_UM[self.unit] / _TO_UM[unit]
        return replace(self, spacing=tuple(s * f for s in self.spacing), unit=unit)

    def with_data(self, data: np.ndarray) -> "Volume":
        """Return a copy carrying new data on the same geometry."""
        return replace(self, data=data)

    def assert_compatible(self, other: "Volume", *, same_frame: bool = True) -> None:
        """Raise unless ``other`` shares this volume's sampling grid."""
        if self.shape != other.shape:
            raise GeometryError(f"shape mismatch: {self.shape} vs {other.shape}")
        if not np.allclose(self.spacing_um, other.spacing_um, rtol=1e-3):
            raise GeometryError(
                f"spacing mismatch: {self.spacing_um} um vs {other.spacing_um} um"
            )
        if same_frame and self.frame != other.frame:
            raise GeometryError(
                f"frame mismatch: '{self.frame}' vs '{other.frame}'. Transform one "
                f"into the other's frame before combining them."
            )
        if not np.allclose(self.affine, other.affine, atol=1e-3):
            raise GeometryError("affine mismatch between volumes on the same grid")


def load_nifti(
    path: str | Path,
    *,
    frame: str,
    spacing: Sequence[float] | None = None,
    unit: Unit | None = None,
    dtype: str = "float32",
) -> Volume:
    """Load a NIfTI volume.

    Args:
        path: File to read.
        frame: Name of the coordinate frame the result belongs to.
        spacing: Voxel size override. Supply this whenever the header is known
            to be unreliable; the stored pixdim is used otherwise.
        unit: Unit of ``spacing``. Required if ``spacing`` is given, otherwise
            inferred from the header's unit code.
        dtype: Array dtype to load as.
    """
    img = nib.load(str(path))
    if spacing is None:
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        unit = unit or _unit_from_header(img.header)
    elif unit is None:
        raise GeometryError("`unit` is required when `spacing` is supplied")
    return Volume(
        data=np.asanyarray(img.dataobj).astype(dtype),
        spacing=tuple(float(s) for s in spacing),  # type: ignore[arg-type]
        unit=unit,
        affine=np.asarray(img.affine, dtype=float),
        frame=frame,
    )


def _unit_from_header(header) -> Unit:
    try:
        code = header.get_xyzt_units()[0]
    except Exception:
        code = "unknown"
    if code in ("micron", "um"):
        return "um"
    if code in ("mm", "millimeter"):
        return "mm"
    raise GeometryError(
        f"header declares spatial unit '{code}'; supply `spacing` and `unit` explicitly"
    )


def load_nrrd_mask(path: str | Path, reference: Volume) -> Volume:
    """Load a segmentation mask and verify it lies on ``reference``'s grid.

    Segmentation editors may export a cropped bounding box rather than the full
    field of view. Such a file cannot be placed without its extent offset, so it
    is rejected rather than silently misaligned.
    """
    data, header = nrrd.read(str(path))
    offset = str(header.get("Segmentation_ReferenceImageExtentOffset", "0 0 0"))
    if offset.replace(" ", "") not in ("000", ""):
        raise GeometryError(
            f"mask was exported on a cropped extent (offset {offset}); "
            f"re-export at the full field of view"
        )
    mask = Volume(
        data=(data > 0).astype("uint8"),
        spacing=reference.spacing,
        unit=reference.unit,
        affine=reference.affine,
        frame=reference.frame,
    )
    reference.assert_compatible(mask)
    return mask


def save_mask(volume: Volume, path: str | Path, reference: Volume | None = None) -> Path:
    """Write a binary mask as uint8 NIfTI with scaling disabled."""
    ref = reference or volume
    header = nib.Nifti1Header()
    header.set_data_dtype(np.uint8)
    header["scl_slope"], header["scl_inter"] = 1.0, 0.0
    img = nib.Nifti1Image((volume.data > 0).astype("uint8"), ref.affine, header)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def background_level(volume: Volume) -> float:
    """Modal intensity of the non-zero histogram.

    Cleared-tissue volumes have a non-zero background plateau inside the field
    of view, with true zeros only outside the acquisition box. Sampling corners
    therefore reports zero rather than the background level.
    """
    values = volume.data[volume.data > 0]
    if values.size == 0:
        raise GeometryError("volume is entirely zero")
    counts, edges = np.histogram(values, bins=512)
    i = int(counts.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))
