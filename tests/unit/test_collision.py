import numpy as np
import pytest

from sew_mimic import (
    Capsule,
    capsule_contact,
    closest_points_on_segments,
    cpp_backend_available,
    minimum_capsule_distance,
)
from sew_mimic.collision import _capsule_distances_unchecked
from sew_mimic.safety import CapsuleRadii, SafetyFilterConfig


def test_closest_points_for_crossing_segments():
    point_a, point_b, parameter_a, parameter_b = closest_points_on_segments(
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    )
    assert point_a == pytest.approx([0.0, 0.0, 0.0])
    assert point_b == pytest.approx([0.0, 0.0, 0.0])
    assert parameter_a == pytest.approx(0.5)
    assert parameter_b == pytest.approx(0.5)


def test_capsule_signed_distance_and_contact_points():
    first = Capsule(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.2)
    second = Capsule(np.array([0.0, 0.5, 0.0]), np.array([1.0, 0.5, 0.0]), 0.2)
    contact = capsule_contact(first, second)
    assert contact.distance == pytest.approx(0.1)
    assert not contact.colliding
    assert contact.normal == pytest.approx([0.0, 1.0, 0.0])

    colliding = capsule_contact(
        first,
        Capsule(np.array([0.0, 0.3, 0.0]), np.array([1.0, 0.3, 0.0]), 0.2),
    )
    assert colliding.distance == pytest.approx(-0.1)
    assert colliding.colliding


def test_vectorized_capsule_distances_match_scalar_kernel():
    rng = np.random.default_rng(4)
    starts_a = rng.normal(size=(100, 3))
    ends_a = starts_a + rng.normal(size=(100, 3))
    starts_b = rng.normal(size=(100, 3))
    ends_b = starts_b + rng.normal(size=(100, 3))
    radii_a = rng.uniform(0.01, 0.2, size=100)
    radii_b = rng.uniform(0.01, 0.2, size=100)
    batched = _capsule_distances_unchecked(starts_a, ends_a, starts_b, ends_b, radii_a, radii_b)
    scalar = np.array(
        [
            capsule_contact(Capsule(a0, a1, ra), Capsule(b0, b1, rb)).distance
            for a0, a1, b0, b1, ra, rb in zip(starts_a, ends_a, starts_b, ends_b, radii_a, radii_b)
        ]
    )
    assert np.allclose(batched, scalar, atol=1e-12)


@pytest.mark.skipif(not cpp_backend_available(), reason="native extension is not built")
def test_cpp_minimum_capsule_distance_matches_python():
    config = SafetyFilterConfig(
        radii=CapsuleRadii(0.15, 0.07, 0.06, 0.05),
        torso_start=np.array([0.0, 0.0, 0.3]),
        torso_end=np.array([0.0, 0.0, 1.2]),
    )
    rng = np.random.default_rng(8)
    for _ in range(100):
        points = rng.normal(size=(8, 3)) * 0.3 + np.array([0.3, 0.0, 0.8])
        python_distance = minimum_capsule_distance(points, config, backend="python")
        cpp_distance = minimum_capsule_distance(points, config, backend="cpp")
        assert cpp_distance == pytest.approx(python_distance, abs=1e-12)
