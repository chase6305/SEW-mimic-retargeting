import math

import numpy as np
import pytest

from sew_mimic import (
    DegenerateGeometryError,
    as_vec3,
    equivalent_angles_in_limits,
    is_rotation_matrix,
    make_frame,
    rot,
    unit,
)


def test_as_vec3_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        as_vec3([0.0, np.nan, 1.0])


def test_unit_rejects_degenerate_vector():
    with pytest.raises(DegenerateGeometryError):
        unit([0.0, 0.0, 0.0])


def test_rotation_validation():
    assert is_rotation_matrix(rot([0.0, 0.0, 1.0], 0.3))
    assert not is_rotation_matrix(np.diag([1.0, 1.0, -1.0]))
    assert not is_rotation_matrix(np.full((3, 3), np.nan))


def test_equivalent_angles_are_sorted_around_reference():
    values = equivalent_angles_in_limits(0.2, -3 * math.pi, 3 * math.pi, 6.4)
    assert values == pytest.approx([0.2 + 2 * math.pi, 0.2, 0.2 - 2 * math.pi])


def test_make_frame_is_orthonormal():
    rotation, origin = make_frame([0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0])
    assert is_rotation_matrix(rotation)
    assert origin == pytest.approx([0.0, 0.0, 1.0])
