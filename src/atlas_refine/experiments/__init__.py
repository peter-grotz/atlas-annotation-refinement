"""Record of correction attempts, indexed by measured error signature."""

from .trials import (
    COMPARISON_FIELDS,
    Trial,
    TrialLog,
    render_index,
    signature_distance,
    signature_vector,
)

__all__ = [
    "COMPARISON_FIELDS",
    "Trial",
    "TrialLog",
    "render_index",
    "signature_distance",
    "signature_vector",
]
