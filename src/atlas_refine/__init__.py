"""Atlas annotation propagation and ground-truth-anchored refinement."""

from .io.store import GroundTruthStore, IntakeError, Provenance
from .io.volumes import GeometryError, Volume, background_level, load_nifti, load_nrrd_mask, save_mask

__version__ = "0.1.0"

__all__ = [
    "GroundTruthStore",
    "IntakeError",
    "Provenance",
    "GeometryError",
    "Volume",
    "background_level",
    "load_nifti",
    "load_nrrd_mask",
    "save_mask",
    "__version__",
]
