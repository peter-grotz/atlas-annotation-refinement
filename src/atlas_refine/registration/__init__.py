"""Registration and label propagation."""

from .register import (
    MetricTerm,
    Registration,
    RegistrationConfig,
    propagate_labels,
    register,
)

__all__ = [
    "MetricTerm",
    "Registration",
    "RegistrationConfig",
    "propagate_labels",
    "register",
]
