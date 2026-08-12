"""Closed-form SEW-Mimic retargeting for seven-DoF humanoid arms."""

from .solver import (
    Serial7DoF,
    align_axis,
    align_wrist_parallel,
    alignment_diagnostics,
    sew_mimic,
    sp1,
    sp2,
    sp4,
)
from .utility import (
    DegenerateGeometryError,
    JointLimitError,
    SEWMimicError,
    as_vec3,
    equivalent_angles_in_limits,
    is_rotation_matrix,
    make_frame,
    rot,
    skew,
    unit,
    wrap_to_pi,
)

__all__ = [
    "DegenerateGeometryError",
    "JointLimitError",
    "SEWMimicError",
    "Serial7DoF",
    "align_axis",
    "align_wrist_parallel",
    "alignment_diagnostics",
    "as_vec3",
    "equivalent_angles_in_limits",
    "is_rotation_matrix",
    "make_frame",
    "rot",
    "sew_mimic",
    "skew",
    "sp1",
    "sp2",
    "sp4",
    "unit",
    "wrap_to_pi",
]
