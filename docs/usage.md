# Usage

See the [README](../README.md) for installation and the Marvin demos.
The solver matches limb directions and hand orientation; limb lengths do not
set absolute end-effector position.

## Input frames

- Joint vectors and limits use radians; positions use metres.
- All shoulder/elbow/wrist positions and the hand orientation in one solve use
  the same arm-base frame.
- Shoulder-to-elbow and elbow-to-wrist vectors must be nonzero.
- Hand orientation is a finite `(3, 3)` SO(3) rotation matrix.

For Marvin, arm bases are `left_arm_base` and `right_arm_base`. Inspect
`arm.base_link`, `arm.ee_link`, and `arm.joint_names` for any registered robot.
Marvin's joint order is shoulder pitch, shoulder roll, elbow pitch, elbow yaw,
wrist pitch, wrist yaw, wrist roll.

Convert world inputs using the arm's world transform:

```python
R_arm_world = R_world_arm.T
shoulder_arm = R_arm_world @ (shoulder_world - p_world_arm)
elbow_arm = R_arm_world @ (elbow_world - p_world_arm)
wrist_arm = R_arm_world @ (wrist_world - p_world_arm)
hand_orientation_arm = R_arm_world @ hand_orientation_world
```

## Wrist orientation

`align_wrist_orientation()` solves J5–J7 while holding J1–J4 fixed.
`align_wrist_parallel` remains an alias for compatibility. Both Python and C++
use the same geometric construction:

```text
T(q) = R07(q) @ R_7T_local @ R_align
R07_des = H @ R_align.T @ R_7T_local.T
```

The solver aligns `R07_des @ axes_local[6]`, then recovers the remaining
rotation about J7. Aligning J7 and the independent J6 direction determines
the complete rotation. Thus the tracked tool's X axis need not coincide with J7.
Consecutive joint-axis orthogonality, valid calibration, and joint limits still
apply; this is not support for arbitrary seven-axis geometries. Near singular
configurations, numerical thresholds and branch selection affect the result.

In the bundled URDFs, using the configured tool frame and default `R_align`,
the absolute dot product between the tool X direction and J7 is approximately
1 for Marvin's `*_ee` and 0 for OpenArm's `*_hand_base`, on both sides.
These describe the selected coordinate frames, not a classification of every
hardware variant or a conclusion drawn from the joint names.

This implementation generalizes Algorithm 3's axis alignment. It does not
implement the separate Euler decomposition in
[Appendix C of the paper](https://arxiv.org/html/2602.01632v1#A3).
Joint-limit filtering selects among the analytical branches considered by the
solver; it is not an exhaustive search over all arm configurations.

## Backends and batches

`solve(..., backend="python"|"cpp")` selects the implementation explicitly.
Omitting `backend` uses `SEW_MIMIC_BACKEND`, defaulting to Python.
Selecting C++ fails if the native extension is unavailable; `sew_mimic()`
remains the direct Python API.

```bash
SEW_MIMIC_BUILD_CPP=1 pip install -e '.[cpp]'
sew-mimic-info --json
sew-mimic-info --validate-robots
```

`backend_status()` reports availability without initializing a native robot.
For CMake-based integration:

```bash
cmake -S src/cpp -B build/cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/cpp --parallel
```

Ordered trajectories share one API:

```python
from sew_mimic import solve_batch

q_trajectory = solve_batch(
    robot,
    q_initial,
    shoulders,  # (N, 3)
    elbows,  # (N, 3)
    wrists,  # (N, 3)
    hand_orientations,  # (N, 3, 3)
    backend="cpp",
)
```

The result is `(N, 7)`. Each frame starts from the previous solution, preserving
single-frame branch selection. Both backends follow this contract.

C++ caches up to 16 calibrated robot solvers; `get_backend("cpp").clear_cache()`
releases them. Native computation releases the GIL after input validation.
Applications own worker threads and scheduling. A safety filter's selected
backend applies to solving, prepared FK, capsule checks, and XPBD projection.
See [architecture](architecture.md) for model ownership.

## Self-collision filtering

Install `.[safety]` for mesh-based OOBB capsule estimation. Construct a filter
once and pass current and desired bimanual commands:

```python
from sew_mimic.robots import create_robot_safety_filter

safety_filter = create_robot_safety_filter("marvin", padding=1.05, backend="python")
result = safety_filter.filter(q_left_current, q_right_current, q_left_target, q_right_target)
q_left_command, q_right_command = result.q_left, result.q_right
```

The filter is also callable. `result.safe`, `result.status.value`,
`result.desired_minimum_distance`, `result.command_minimum_distance`, and
`result.iterations` expose the decision. Distances use metres; negative values
mean capsule penetration.

| Status              | Meaning                                            |
| ------------------- | -------------------------------------------------- |
| `accepted`          | Desired command passed collision checks.           |
| `corrected`         | XPBD correction passed final FK validation.        |
| `xpbd_failed`       | Projection did not reach the configured clearance. |
| `ik_failed`         | Corrected targets had no valid SEW solution.       |
| `validation_failed` | Reconstructed joint pose remained unsafe.          |

On failure the returned command is the current pose. This does not make an
already unsafe current pose safe.

The pipeline computes desired keypoints with FK, samples the path from the
current pose, projects capsule constraints with XPBD, restores link lengths,
solves corrected directions, and validates the reconstructed pose with FK.
OOBB fitting is cached by URDF path and padding during initialization.

Pass a `SafetyFilterConfig` through `config=` to customize radii, collision
pairs, thresholds, compliance, or iterations. Thresholds must satisfy
`minimum_distance <= activation_distance <= release_distance`. Default pairs
exclude adjacent same-arm capsules and torso/upper-arm attachments.
See [robot integration](robot-integration.md) for geometry setup.

## Tracking and command filtering

Use `BimanualPoseFilter` for One Euro keypoint smoothing and shortest-arc
quaternion filtering, and `JointRateLimiter` for actuator velocity limits.
The online order is tracking filtering, SEW solving, collision filtering, then
joint-rate limiting. Recheck collisions when command limiting changes the
corrected pose. Reset histories after tracking loss, an emergency stop, or
controller reconnection.

The [VR integration guide](vr-integration.md) covers calibration, independent
position/rotation tuning, stream configuration, motion rejection, and replay.
Hardware-free examples:

```bash
python -m examples.demo_tracking_filtering --rate 90
python -m examples.demo_tracking_calibration
python -m examples.demo_tracking_input --robot marvin --record /tmp/tracking-demo.jsonl
python -m examples.demo_tracking_input --replay /tmp/tracking-demo.jsonl
```

## Logging and metrics

The package uses standard-library logging without installing handlers on import.
`configure_logging("INFO")` enables setup and summary messages; `"DEBUG"`
includes per-frame stages and changes timing results. Disable logging for benchmarks.

The motion demo reports solver-only arm solves/s, sequential pose pairs/s,
and ms/arm. The collision demo reports a complete bimanual filter call, including
FK and final validation, in frames/s and mean/P95 ms/frame. Neither includes
rendering, transport, nor hardware I/O. Reference data and reproduction commands
are in [performance measurements](performance-2026-09-05.md).
