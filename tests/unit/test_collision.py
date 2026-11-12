import numpy as np
import pytest

from sew_mimic import (
    Capsule,
    capsule_contact,
    closest_points_on_segments,
    cpp_backend_available,
    minimum_capsule_distance,
)
from sew_mimic.backends import get_cpp_collision_backend
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


def _assert_capsule_distances(endpoints, radius, expected, scale):
    starts = np.asarray([endpoints[0], endpoints[2]], dtype=float)
    ends = np.asarray([endpoints[1], endpoints[3]], dtype=float)
    tolerance = dict(rel=1e-10, abs=1e-15 * scale)
    first = Capsule(starts[0], ends[0], radius)
    second = Capsule(starts[1], ends[1], radius)
    assert capsule_contact(first, second).distance == pytest.approx(expected, **tolerance)
    with np.errstate(all="raise"):
        batched = _capsule_distances_unchecked(
            starts[:1], ends[:1], starts[1:], ends[1:], np.array([radius]), np.array([radius])
        )
    assert batched[0] == pytest.approx(expected, **tolerance)
    if cpp_backend_available():
        native = get_cpp_collision_backend().minimum_distance(
            starts, ends, np.array([radius, radius]), np.array([[0, 1]], dtype=np.int32)
        )
        assert native == pytest.approx(expected, **tolerance)


@pytest.mark.parametrize("scale", [1e-5, 0.01, 1.0, 1e3])
def test_small_crossing_segments_have_negative_clearance_at_every_scale(scale):
    half_length = 1e-3 * scale
    endpoints = half_length * np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])
    point_a, point_b, parameter_a, parameter_b = closest_points_on_segments(*endpoints)
    np.testing.assert_allclose(point_a, np.zeros(3), atol=1e-15 * half_length)
    np.testing.assert_allclose(point_b, np.zeros(3), atol=1e-15 * half_length)
    assert parameter_a == pytest.approx(0.5)
    assert parameter_b == pytest.approx(0.5)
    _assert_capsule_distances(endpoints, half_length / 10, -half_length / 5, half_length)


@pytest.mark.parametrize("half_length", [1e-8, 1e-6, 1e-3])
@pytest.mark.parametrize("swap", [False, True])
def test_short_nonzero_segment_keeps_its_interior_minimum(half_length, swap):
    endpoints = half_length * np.array([[-1, 0, 0], [1, 0, 0], [0, 0.25, 0], [0, 0.25, 0]])
    if swap:
        endpoints = endpoints[[2, 3, 0, 1]]
    _assert_capsule_distances(endpoints, half_length / 10, half_length / 20, half_length)


@pytest.mark.parametrize("angle", [1e-4, 1e-6, 1e-8, 1e-12])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("reverse_a,reverse_b,swap", [(False, False, False), (True, True, True)])
def test_nearly_parallel_crossings_keep_the_interior_intersection(
    angle, scale, reverse_a, reverse_b, swap
):
    endpoints = scale * np.array([[-1, 0, 0], [1, 0, 0], [-1, -angle, 0], [1, angle, 0]])
    if reverse_a:
        endpoints[[0, 1]] = endpoints[[1, 0]]
    if reverse_b:
        endpoints[[2, 3]] = endpoints[[3, 2]]
    if swap:
        endpoints = endpoints[[2, 3, 0, 1]]
    _assert_capsule_distances(endpoints, 1e-4 * scale, -2e-4 * scale, scale)


@pytest.mark.parametrize("angle", [0.3, 1e-6, 1e-10])
def test_skew_capsules_keep_known_clearance_after_rigid_transform(angle):
    # The centerlines' XY projections intersect at their midpoints, while
    # their constant Z separation is 0.02 m. Rotate and translate this known
    # minimum so neither the ordinary nor near-parallel kernel can depend on
    # special world-axis coordinates or coincident closest points.
    rotation = np.array([[2, -2, 1], [1, 2, 2], [-2, -1, 2]], dtype=float) / 3.0
    endpoints = np.array([[-1, 0, 0], [1, 0, 0], [-1, -angle, 0.02], [1, angle, 0.02]])
    endpoints = endpoints @ rotation.T + np.array([0.2, -0.3, 0.4])
    _assert_capsule_distances(endpoints, 0.005, 0.01, 1.0)


@pytest.mark.parametrize("reverse_a", [False, True])
@pytest.mark.parametrize("reverse_b", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_parallel_capsule_distance_is_invariant_to_endpoint_order(reverse_a, reverse_b, swap):
    endpoints = 1e-3 * np.array([[0, 0, 0], [2, 0, 0], [1, 0.5, 0], [3, 0.5, 0]])
    if reverse_a:
        endpoints[[0, 1]] = endpoints[[1, 0]]
    if reverse_b:
        endpoints[[2, 3]] = endpoints[[3, 2]]
    if swap:
        endpoints = endpoints[[2, 3, 0, 1]]
    _assert_capsule_distances(endpoints, 1e-4, 3e-4, 1e-3)


@pytest.mark.parametrize("scale", [1e-5, 0.01, 1.0, 1e3])
def test_vectorized_capsule_distances_match_scalar_kernel(scale):
    rng = np.random.default_rng(4)
    starts_a = rng.normal(size=(100, 3))
    ends_a = starts_a + rng.normal(size=(100, 3))
    starts_b = rng.normal(size=(100, 3))
    ends_b = starts_b + rng.normal(size=(100, 3))
    radii_a = rng.uniform(0.01, 0.2, size=100)
    radii_b = rng.uniform(0.01, 0.2, size=100)
    ends_a[::5] = starts_a[::5]
    ends_b[::7] = starts_b[::7]
    ends_b[1::5] = starts_b[1::5] + ends_a[1::5] - starts_a[1::5]
    starts_a, ends_a, starts_b, ends_b, radii_a, radii_b = (
        value * scale for value in (starts_a, ends_a, starts_b, ends_b, radii_a, radii_b)
    )
    with np.errstate(all="raise"):
        batched = _capsule_distances_unchecked(starts_a, ends_a, starts_b, ends_b, radii_a, radii_b)
    scalar = np.array(
        [
            capsule_contact(Capsule(a0, a1, ra), Capsule(b0, b1, rb)).distance
            for a0, a1, b0, b1, ra, rb in zip(starts_a, ends_a, starts_b, ends_b, radii_a, radii_b)
        ]
    )
    assert np.allclose(batched, scalar, atol=1e-12)


@pytest.mark.parametrize(
    "a0,a1,b0,b1,distance",
    [
        ([0, 0, 0], [0, 0, 0], [0, 2, 0], [0, 2, 0], 2.0),
        ([2, 0, 0], [2, 0, 0], [0, 0, 0], [1, 0, 0], 1.0),
        ([0, 0, 0], [1, 0, 0], [-1, 1, 0], [-1, 1, 0], np.sqrt(2)),
        ([0, 0, 0], [1, 0, 0], [2, -1, 0], [2, 1, 0], 1.0),
        ([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], 1.0),
        ([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], 0.0),
    ],
)
def test_capsule_kernels_agree_on_degenerate_parallel_and_endpoint_contacts(
    a0, a1, b0, b1, distance
):
    first, second = Capsule(a0, a1, 0.1), Capsule(b0, b1, 0.2)
    contact = capsule_contact(first, second)
    assert contact.distance == pytest.approx(distance - 0.3)
    assert np.linalg.norm(contact.normal) == pytest.approx(1.0)
    starts = np.array([a0, b0], dtype=float)
    ends = np.array([a1, b1], dtype=float)
    vectorized = _capsule_distances_unchecked(
        starts[:1], ends[:1], starts[1:], ends[1:], np.array([0.1]), np.array([0.2])
    )
    assert vectorized[0] == pytest.approx(contact.distance)
    if cpp_backend_available():
        native = get_cpp_collision_backend().minimum_distance(
            starts, ends, np.array([0.1, 0.2]), np.array([[0, 1]], dtype=np.int32)
        )
        assert native == pytest.approx(contact.distance)


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
