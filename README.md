# SEW-Mimic Retargeting

[![CI](https://github.com/chase6305/SEW-mimic-retargeting/actions/workflows/ci.yml/badge.svg)](https://github.com/chase6305/SEW-mimic-retargeting/actions/workflows/ci.yml)

A Python reference and high-performance C++ implementation of the closed-form geometric
retargeting method
from *A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot
Teleoperation*. The repository includes the Marvin M6 robot assets, single- and
dual-arm retargeting, interactive viser demos, and the paper's capsule/XPBD
bimanual self-collision safety-filter pipeline.

![Marvin M6 viser demo](docs/media/marvin_viser.gif)

## Highlights

- Closed-form seven-DoF arm retargeting without iterative numerical IK.
- Explicitly selectable Python reference and native C++ backends.
- IK-Geo Subproblems 1, 2, and 4.
- Parallel-wrist `AlignWrist` and joint-limit-aware analytical solution selection.
- Marvin M6 and OpenArm left/right arm extraction from bundled URDFs.
- Continuous 60-second dual-arm human-like motion demo.
- Viser visualization of robot meshes, keypoints, bones, and coordinate frames.
- Capsule parameters estimated from collision-mesh oriented bounding boxes.
- Continuous-path collision sampling and XPBD bimanual safety filtering.
- Final FK validation before a corrected command is accepted.
- Stage logging and clearly defined solve-rate/latency metrics.
- Unit, integration, formatting, linting, wheel-build, and asset-packaging checks.

SEW-Mimic matches limb directions and hand orientation rather than absolute
human limb lengths:

```text
upper-arm direction = normalize(elbow - shoulder)
forearm direction   = normalize(wrist - elbow)
end-effector target = hand_orientation
```

## Installation

Python 3.10 or newer is required.

Core solver only:

```bash
pip install -e .
```

Viser demos:

```bash
pip install -e '.[visualization]'
```

OOBB capsule fitting and safety-filter configuration:

```bash
pip install -e '.[safety]'
```

Complete development environment:

```bash
pip install -e '.[dev,visualization,safety]'
```

C++ development, including the pinned formatter:

```bash
pip install -e '.[dev,cpp,cpp-dev]'
clang-format -i src/cpp/sew_mimic_cpp.cpp
```

### Python/C++ backend selection

The public `solve()` API requires the application layer to choose `python` or
`cpp`. The default is `python`; there is no implicit fallback between implementations:

```python
from sew_mimic import get_backend, solve

q = solve(robot, q0, shoulder, elbow, wrist, hand_orientation, backend="cpp")
print(get_backend("cpp").name)
```

Deployment diagnostics are available without initializing a native robot model:

```python
from sew_mimic import backend_status

print(backend_status())
# {'default': 'python', 'cpp_available': True, 'cpp_implementation': 'native'}
```

The installed command also reports package version, platform, registered
robots, and resolved asset paths:

```bash
sew-mimic-info
sew-mimic-info --json
sew-mimic-info --validate-robots
```

A process-wide default can be selected with `SEW_MIMIC_BACKEND`. Build the
optional extension with:

```bash
SEW_MIMIC_BUILD_CPP=1 pip install -e '.[cpp]'
export SEW_MIMIC_BACKEND=cpp  # python | cpp
```

For CMake-based Jetson or system integration builds:

```bash
cmake -S src/cpp -B build/cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/cpp --parallel
```

Selecting `cpp` fails immediately if the extension is unavailable. Existing
`sew_mimic()` calls remain the direct Python API.
The native C++17 backend implements SP1/SP2/SP4, axis and wrist alignment,
joint-limit handling, rotation/FK operations, and the complete single-arm
SEW-Mimic solve. It uses the same calibrated NumPy-array model data, so robot
adapters and application code are shared by both implementations.

The native code is organized around focused classes:

- `RobotModel` owns calibrated axes, fixed rotations, limits, and rotational FK;
- `SewMimicSolver` owns axis alignment, wrist recovery, and the complete solve;
- `CapsuleCollisionDetector` owns segment closest-point and signed-distance queries.
- `XpbdCollisionProjector` owns native constraint multipliers and point projection.

The Python boundary mirrors those responsibilities with `CppBackend` for arm
solving and `CppCollisionBackend` for collision/XPBD. Safety code does not import
the private pybind module directly, which keeps native binding details isolated.

`CppBackend` caches one persistent native `SewMimicSolver` for each calibrated
`Serial7DoF` object, so axes, limits, and fixed rotations are parsed once rather
than once per frame. The cache is LRU-bounded to 16 robot objects and can be
released explicitly with `get_backend("cpp").clear_cache()`. Ordered trajectories
can cross the binding once:

```python
from sew_mimic import solve_batch

q_trajectory = solve_batch(
    robot,
    q_initial,
    shoulders,          # (N, 3)
    elbows,             # (N, 3)
    wrists,             # (N, 3)
    hand_orientations,  # (N, 3, 3)
    backend="cpp",
)
```

Batch solving is stateful within the call: each frame starts from the previous
frame's solution, matching the real-time single-frame API and branch-selection
behavior. The Python backend implements the same contract.

Native computation releases the Python GIL after array validation. Independent
left/right arm batches, capsule queries, and XPBD projections can therefore run
concurrently in application-managed worker threads. On the current two-worker
test, two 100,000-frame batches improved from `0.617 s` sequential to `0.308 s`
parallel (approximately `2.0x`). The backend itself does not create threads, so
thread ownership and real-time scheduling remain explicit at the application layer.

Selecting `backend="cpp"` on a robot safety filter applies to SEW solving,
capsule contacts/minimum-distance checks, XPBD multiplier updates, collision
gradient projection, link-length projection, and convergence testing. Python
retains an independent implementation of the complete path as the reference.

See [Performance metrics](#performance-metrics) for the reference hardware,
measurement scope, Python/C++ results, and the exact reproduction command.

The Marvin URDF, collision meshes, and visual meshes are included in built
wheels as data files. An alternative URDF can be supplied to every Marvin API
and demo with an explicit path.

## Quick start

### Solve one arm frame

```python
import numpy as np

from sew_mimic import sew_mimic
from sew_mimic.robots import load_marvin_arm

arm = load_marvin_arm(side="left")

q_current = np.zeros(7)  # radians
shoulder = np.array([0.0, 0.0, 0.0])  # metres, left_arm_base frame
elbow = np.array([0.20, 0.20, 0.05])
wrist = np.array([0.45, 0.25, 0.10])
hand_orientation = np.eye(3)  # valid SO(3) rotation matrix

q_command = sew_mimic(
    arm.robot,
    q_current,
    shoulder,
    elbow,
    wrist,
    hand_orientation,
)
```

Select OpenArm with the corresponding adapter; the solver API is unchanged:

```python
from sew_mimic.robots import load_openarm_arm

openarm_left = load_openarm_arm(side="left")
openarm_right = load_openarm_arm(side="right")
```

Applications that select robots dynamically can use the unified registry API:

```python
from sew_mimic.robots import available_robots, load_robot_arm

print(available_robots())  # ("marvin", "openarm")
arm = load_robot_arm("openarm", side="left")
```

Aliases such as `m6`, `marvin-m6`, and `open-arm` are accepted by the Python
registry. Applications can register additional adapters and aliases at runtime;
a custom URDF path can be passed as the third argument.

## Integrating a new robot

Integration is deliberately split into three levels. Complete level 1 before
adding visualization, and validate visualization before enabling collision
correction. This keeps kinematic convention errors separate from safety-model
errors.

### 1. Check the kinematic contract

Each arm exposed to SEW-Mimic must be a serial seven-DoF revolute chain ordered
from shoulder to wrist. The adapter must provide:

| Field                  | Meaning                                                                     |
| ---------------------- | --------------------------------------------------------------------------- |
| `axes_local`           | Seven joint axes expressed in their URDF joint frames.                      |
| `R_local`              | Seven fixed parent-to-joint origin rotations, in chain order.               |
| `q_min`, `q_max`       | Finite lower/upper joint limits in radians.                                 |
| `R_7T_local`           | Fixed rotation from joint-seven child frame to the tracked hand/tool frame. |
| `R_align`              | Rotation aligning the human hand convention with the robot tool convention. |
| `joint_names`          | Seven URDF actuated-joint names in the exact solver order.                  |
| `base_link`, `ee_link` | Arm reference link and tracked hand/tool link used by FK and Viser.         |

Do not infer the hand frame from its visual mesh. Follow the fixed-joint chain
from joint seven to the intended control landmark. A hand-base link is often a
better retargeting landmark than a fingertip TCP. All shoulder, elbow, wrist,
and hand-orientation inputs passed to one solve must use the same arm-base
coordinate frame and SI units.

The returned arm object is structural: it may be any dataclass that satisfies
the `RobotArm` protocol. `MarvinArm` and `OpenArmArm` are working examples.

### 2. Implement and register the arm adapter

Create a robot module outside the core package first. `SerialArmSpec` keeps the
common case declarative: it parses the URDF, verifies that each set of seven
named joints forms one continuous chain, extracts axes/origin rotations/limits,
accumulates the fixed tool rotation, and constructs the solver model. Use the
lower-level `load_serial_7dof_arm()` helper only when a robot needs a custom arm
wrapper.

```python
from pathlib import Path

import numpy as np

from sew_mimic.robots import (
    RobotAdapter,
    SerialArmSpec,
    register_robot_adapter,
)

ARM_SPEC = SerialArmSpec(
    left_joint_names=(
        "left_j1",
        "left_j2",
        "left_j3",
        "left_j4",
        "left_j5",
        "left_j6",
        "left_j7",
    ),
    right_joint_names=(
        "right_j1",
        "right_j2",
        "right_j3",
        "right_j4",
        "right_j5",
        "right_j6",
        "right_j7",
    ),
    left_ee_link="left_hand_base",
    right_ee_link="right_hand_base",
)


def my_reference_trajectory(side, duration, fps, cycle_duration=None):
    """Return times and a smooth, joint-limit-valid (N, 7) reference path."""
    times = np.linspace(0.0, duration, int(np.ceil(duration * fps)) + 1)
    reference = np.zeros((len(times), 7))
    return times, reference


register_robot_adapter(
    RobotAdapter(
        name="my-robot",
        display_name="My Robot",
        default_urdf=Path("assets/MyRobot/robot.urdf"),
        load_arm=ARM_SPEC.load,
        trajectory_profile=my_reference_trajectory,
        keypoint_profile="urdf",
    ),
    aliases=("myrobot",),
)
```

Registration is process-local and thread-safe. Names use lowercase hyphenated
form; duplicate names and aliases are rejected unless an application explicitly
uses `replace=True`. A callable `trajectory_profile` lets the generic Viser demo
support a new robot without adding robot-specific branches to the demo.
`keypoint_profile="urdf"` draws physical URDF landmarks; use `"solver"` when
the visualization should show the abstract SEW direction targets instead.

Because registration must happen before CLI argument parsing, use a small
launcher when the adapter lives outside this repository:

```python
import my_robot_adapter  # performs register_robot_adapter(...)

from examples.demo_robot_viser import main

main()
```

Then run the launcher with the normal arguments:

```bash
python my_robot_viser.py --robot my-robot --backend cpp --side both
```

### 3. Validate retargeting before safety

Run the generic adapter acceptance check first:

```python
from sew_mimic.robots import validate_robot_adapter

report = validate_robot_adapter("my-robot")
print(report)
```

It loads both arms, chooses the closest-to-zero joint-limit-valid neutral pose,
evaluates bimanual FK, validates finite keypoints and both tool SO(3) matrices,
summarizes the narrowest joint range, and reports the largest consecutive-axis
dot product. The same check is available through `sew-mimic-info --validate-robots`.

Use known robot configurations as ground truth. For each test pose, obtain the
shoulder/elbow/wrist directions and tool orientation from FK, solve from a
nearby `q0`, and check `alignment_diagnostics()`. Include neutral, bent-elbow,
near-extension, asymmetric, joint-limit, and left/right mirrored poses. A useful
adapter test suite should verify:

- the seven-joint topology and exact joint order;
- FK at zero and at one non-zero configuration against a trusted URDF library;
- `R_7T_local` and `R_align` tool-orientation agreement;
- Python/C++ solutions within numerical tolerance;
- batch and single-frame solutions produce the same stateful trajectory;
- every generated demo reference remains inside joint limits.

If shoulder and elbow positions look correct but the hand twists, fix the tool
frame or `R_align`; do not compensate by altering the human trajectory. If the
elbow bends on the wrong branch, check joint ordering, axis signs, limits, and
the initial `q0`.

### 4. Add collision geometry and a safety wrapper

Safety integration is robot-specific because mesh names, torso placement, and
valid collision pairs differ. Fit conservative capsules from collision-mesh
OOBBs by declaring a `CapsuleMeshSpec` and calling
`estimate_capsule_config_from_meshes()`. The helper fits every unique mesh once,
uses the longest OOBB axis for each capsule, selects the largest radius in each
link group, and transforms the torso capsule into the URDF root frame.

```python
from sew_mimic.robots import CapsuleMeshSpec, estimate_capsule_config_from_meshes

mesh_spec = CapsuleMeshSpec(
    torso="torso/torso.stl",
    upper_arm=("arm/upper_left.stl", "arm/upper_right.stl"),
    lower_arm=("arm/lower_left.stl", "arm/lower_right.stl"),
    hand=("hand/left.stl", "hand/right.stl"),
    torso_link="torso",
)
my_capsule_config = estimate_capsule_config_from_meshes(
    robot_urdf,
    robot_urdf.parent / "collision",
    mesh_spec,
    padding=1.05,
)
```

Keep adjacent-link and physical attachment pairs excluded when customizing the
resulting `SafetyFilterConfig.collision_pairs`.

Use `RobotSafetyFilter` for the shared orchestration. A new robot only needs to
provide its left/right `RobotArm` instances, parsed `URDFKinematics`, capsule
configuration, and one bimanual FK function returning `BimanualPose` in the
URDF root frame:

```python
from sew_mimic.robots import RobotSafetyFilter, URDFKinematics, urdf_bimanual_pose

kinematics = URDFKinematics(robot_urdf)
safety_filter = RobotSafetyFilter(
    robot_urdf,
    left=load_my_robot_arm(robot_urdf, "left"),
    right=load_my_robot_arm(robot_urdf, "right"),
    kinematics=kinematics,
    config=my_capsule_config,
    pose_function=urdf_bimanual_pose,
    backend="cpp",
)
```

`RobotSafetyFilter` owns base-frame conversion, left/right SEW callbacks,
backend dispatch, failure handling, and the callable `filter()` interface. A
named robot class may subclass it to package asset loading and OOBB estimation;
`MarvinSafetyFilter` and `OpenArmSafetyFilter` follow this pattern.
Their real-time FK path uses `URDFBimanualPoseEvaluator`, which resolves the
shoulder, elbow, wrist, tool, and required URDF branches once during filter
construction. `URDFKinematics` also caches an ordered FK execution plan for
those branches, so unrelated fingers, sensors, and fixed links are not visited
on every frame. Custom filters can reuse the same evaluator when they follow
the standard J1/J4/J6 landmark convention.
`urdf_bimanual_pose()` uses the standard J1/J4/J6 shoulder/elbow/wrist
landmarks; pass different landmark indices or a custom pose function when a
robot's mechanical convention differs.

To make the generic collision demo discover the integration, attach the filter
class and four validation endpoints to `RobotAdapter`:

```python
from sew_mimic.robots import CollisionDemoProfile

collision_profile = CollisionDemoProfile(
    neutral_left=q_left_safe,
    neutral_right=q_right_safe,
    colliding_left=q_left_crossed,
    colliding_right=q_right_crossed,
)

register_robot_adapter(
    RobotAdapter(
        name="my-robot",
        display_name="My Robot",
        default_urdf=robot_urdf,
        load_arm=load_my_robot_arm,
        trajectory_profile=my_reference_trajectory,
        safety_filter_factory=MyRobotSafetyFilter,
        collision_profile=collision_profile,
    ),
    replace=True,
)
```

The collision endpoints must be finite seven-joint vectors. The generated
minimum-jerk trajectory moves continuously from neutral through the colliding
pose and back without pausing. Once registered, `demo_robot_collision_avoidance.py`
uses these capabilities without a robot-name branch.

Cache URDF parsing, zero-pose base transforms, OOBB results, collision-pair
indices, and capsule radii during initialization. In the per-frame path, request
only shoulder, elbow, wrist, and tool links from
`URDFKinematics.link_transforms(..., required_links=...)`.

Validate the safety layer with a continuous safe-to-colliding-to-safe motion,
not only isolated poses. Record target and filtered minimum distance, result
status, XPBD iterations, mean/P95 latency, and verify that failure returns the
current command. Only after these tests should the robot be added to the generic
collision demo.

### Integration acceptance checklist

- Solver inputs and FK outputs use metres, radians, and documented frames.
- Both arms pass topology, FK, orientation, joint-limit, and mirror tests.
- Python and C++ backends agree on representative poses and trajectories.
- Viser frames clearly show base, shoulder, elbow, wrist, and tool conventions.
- Capsule padding and every enabled/excluded collision pair are reviewed.
- A continuous collision trajectory demonstrates correction and recovery.
- Benchmarks are recorded on the deployment hardware with logging disabled.
- Asset paths work both from a source checkout and an installed wheel.

The returned Marvin joint order is:

```text
SHOULDER_PITCH
SHOULDER_ROLL
ELBOW_PITCH
ELBOW_YAW
WRIST_PITCH
WRIST_YAW
WRIST_ROLL
```

### Filter one bimanual command

```python
import numpy as np

from sew_mimic.robots import create_robot_safety_filter

safety_filter = create_robot_safety_filter("marvin", padding=1.05, backend="cpp")

result = safety_filter.filter(
    q_left_current=np.zeros(7),
    q_right_current=np.zeros(7),
    q_left_desired=q_left_target,
    q_right_desired=q_right_target,
)

q_left_command = result.q_left
q_right_command = result.q_right

print(result.safe)
print(result.status.value)
print(result.minimum_distance)  # metres; negative means capsule penetration
print(result.desired_minimum_distance)  # unfiltered target clearance
print(result.command_minimum_distance)  # clearance of returned command
print(result.iterations)
```

The returned filter is callable directly and delegates to `filter()`. The same
entry point accepts any registered robot that supplies `safety_filter_factory`.

Safety-filter statuses:

| Status              | Meaning                                                             |
| ------------------- | ------------------------------------------------------------------- |
| `accepted`          | The desired command was already collision-free.                     |
| `corrected`         | XPBD generated a corrected command that passed final FK validation. |
| `xpbd_failed`       | XPBD did not achieve the configured safety distance.                |
| `ik_failed`         | Corrected keypoints had no valid joint-limit-aware SEW solution.    |
| `validation_failed` | The reconstructed joint pose remained unsafe.                       |

For every failure status, the filter returns the current pose rather than the
unsafe desired pose.

## Input conventions

- `q_current`, `q_command`, and joint limits use radians.
- Left-arm inputs use `left_arm_base`; right-arm inputs use `right_arm_base`.
- Shoulder, elbow, wrist, and hand orientation must share one frame.
- Positions conventionally use metres, although limb length does not affect the
  direction-only SEW objective.
- Shoulder-to-elbow and elbow-to-wrist vectors must not be near zero.
- `hand_orientation` must be a finite 3-by-3 SO(3) matrix.

To convert world-frame human inputs into an arm-base frame:

```python
R_arm_world = R_world_arm.T

shoulder_arm = R_arm_world @ (shoulder_world - p_world_arm)
elbow_arm = R_arm_world @ (elbow_world - p_world_arm)
wrist_arm = R_arm_world @ (wrist_world - p_world_arm)
hand_orientation_arm = R_arm_world @ hand_orientation_world
```

## Real-time filtering

Noisy tracking input and actuator command limits are handled outside the
closed-form solver. `BimanualPoseFilter` applies a One Euro filter to all eight
keypoints and shortest-arc quaternion filtering to both tool orientations, so
its orientation outputs remain valid SO(3) matrices. `OneEuroFilter` is also
available independently for scalar or Euclidean observations.

```python
from sew_mimic import BimanualPoseFilter, JointRateLimiter

tracking_filter = BimanualPoseFilter(min_cutoff=1.0, beta=0.02)
filtered_pose = tracking_filter.update(timestamp, measured_bimanual_pose)

command_filter = JointRateLimiter(max_velocity=robot_velocity_limits)
command_filter.reset(current_joint_command, timestamp)
limited_command = command_filter.update(next_timestamp, desired_joint_command)
```

Keep rotation validation enabled for external tracking data. If the pose comes
directly from a validated FK implementation, use
`BimanualPoseFilter(validate_rotations=False)` to avoid two redundant SO(3)
checks per frame; this is a trusted-input optimization and must not be used to
accept arbitrary matrices.

The recommended online order is tracking/keypoint filtering, SEW retargeting,
self-collision filtering, and joint-rate limiting. If rate limiting materially
changes a collision-corrected command, perform a final collision check before
sending it to hardware. Call `reset()` after tracking loss, emergency stop, or
controller reconnection; timestamps must be finite and strictly increasing.

## Demos

### Synthetic solver consistency

```bash
python examples/demo_toy.py
```

This runs FK from known joint angles, reconstructs corresponding SEW targets,
and solves them again. Direction and tool-orientation errors should be close to
floating-point precision.

### Robot-selectable dual-arm motion

```bash
python examples/demo_robot_viser.py --robot marvin --side both
```

Default behavior:

```text
Trajectory duration    60 s
Trajectory rate        60 Hz
Playback speed         1.75x
Arms                   left + right
Viser port             8080
Loop playback          enabled
```

Example with explicit settings:

```bash
python examples/demo_robot_viser.py \
  --robot marvin \
  --side both \
  --duration 60 \
  --motion-cycle 60 \
  --fps 60 \
  --playback-speed 1.75 \
  --port 8080 \
  --log-level INFO
```

The motion program contains chest expansion/crossing, asymmetric arm swings,
alternating punches, level-wrist push/pull motion, and running-style arm swings.
The interface includes playback controls, fourteen arm joint sliders, human
keypoints, bone segments, compact joint frames, enlarged end-effector frames,
and closed-form solver performance metrics.

Run the same visualization with the bundled OpenArm model:

```bash
python examples/demo_robot_viser.py \
  --robot openarm \
  --backend cpp \
  --side both \
  --duration 60 \
  --fps 60 \
  --port 8080
```

OpenArm uses Cartesian-fitted, joint-limit-safe keyframes designed for its
asymmetric left/right shoulder ranges. Its program includes readable chest
opening/crossing, opposing vertical swings, alternating reaches, synchronized
push/pull, and running-style arm swings. `demo_marvin_viser.py` is only a
backward-compatible Marvin entry point; new integrations should use
`demo_robot_viser.py`.

### Bimanual self-collision avoidance

```bash
python examples/demo_robot_collision_avoidance.py \
  --robot marvin \
  --backend cpp \
  --duration 12 \
  --fps 30 \
  --port 8081 \
  --padding 1.05 \
  --log-level INFO
```

The red translucent robot follows a continuous chest-height target that passes
through a colliding cross-arm pose. The solid robot displays the command allowed
by the safety filter. The target never pauses: only the command temporarily
holds its last safe pose while the target crosses the unsafe region, then
resumes tracking after the target becomes safe.

Run the same collision-filter visualization with OpenArm and its own STL OOBB
capsule model:

```bash
python examples/demo_robot_collision_avoidance.py \
  --robot openarm \
  --backend cpp \
  --duration 12 \
  --fps 30 \
  --port 8081
```

`demo_marvin_collision_avoidance.py` remains only as a backward-compatible
Marvin command shim.

The GUI reports:

- unsafe target signed capsule distance in metres;
- filtered command signed capsule distance in metres;
- `SAFE` or `BLOCKED / HOLDING LAST SAFE POSE`;
- complete safety-filter solve rate in Hz;
- mean and P95 safety latency in milliseconds per bimanual frame; and
- the selected trajectory sample's solve time.

## Self-collision safety filter

The safety layer follows the structure of paper Algorithms 4 and 9-12:

1. Compute desired bimanual shoulder/elbow/wrist/tool keypoints with FK.
1. Approximate torso, upper arms, lower arms, and hands with capsules.
1. Interpolate from the current pose to find the first activated collision.
1. Apply XPBD collision constraints to the capsule keypoints.
1. Project modified links back to their original lengths.
1. Recover tool directions and solve corrected SEW targets.
1. Recompute FK and reject any reconstructed pose that remains unsafe.

For Marvin M6, OOBB fitting uses the longest oriented-box dimension as the
capsule axis. Half of the larger transverse dimension becomes the radius, with
configurable padding. Fitting is cached by `(URDF path, padding)` and is not part
of the per-frame real-time path.

Default parameters can be inspected or replaced:

```python
from sew_mimic import CapsuleIndex, SafetyFilterConfig
from sew_mimic.robots import MarvinSafetyFilter, estimate_marvin_capsule_config

default_config = estimate_marvin_capsule_config(padding=1.05)

custom_config = SafetyFilterConfig(
    radii=default_config.radii,
    torso_start=default_config.torso_start,
    torso_end=default_config.torso_end,
    minimum_distance=0.01,
    activation_distance=0.03,
    release_distance=0.04,
    compliance=1e-6,
    iterations=20,
    collision_pairs=(
        (CapsuleIndex.LEFT_HAND, CapsuleIndex.RIGHT_HAND),
        (CapsuleIndex.LEFT_LOWER_ARM, CapsuleIndex.RIGHT_LOWER_ARM),
    ),
)

safety_filter = MarvinSafetyFilter(config=custom_config)
```

The distance thresholds must satisfy:

```text
minimum_distance <= activation_distance <= release_distance
```

Adjacent same-arm capsules and torso-to-upper-arm attachment pairs are excluded
from the default collision-pair set.

## Performance metrics

### Metric definitions

The regular Marvin demo measures only the closed-form solver:

| Metric                  | Unit    | Scope                                         |
| ----------------------- | ------- | --------------------------------------------- |
| Single-arm solve rate   | Hz      | Individual SEW arm solves per second.         |
| Dual-arm pose-pair rate | pairs/s | Sequential left+right solve pairs per second. |
| Mean solve latency      | ms/arm  | Mean time for one arm solve.                  |

The collision demo measures one complete bimanual safety-filter call, including
FK, capsule checking, continuous sampling, XPBD, optional SEW reconstruction,
and final validation. It excludes trajectory generation, logging, Viser scene
updates, networking, and browser rendering.

`Hz`, `solves/s`, and `frames/s` all mean completed operations per second.
Latency is reported in milliseconds per operation; lower latency and higher
throughput are better. A dual-arm frame is one left-arm plus right-arm update,
not two independently counted arm solves.

### Reference hardware

The following results were measured on 2026-08-15. They are reference values,
not guaranteed real-time deadlines.

| Component        | Reference system                                |
| ---------------- | ----------------------------------------------- |
| CPU              | AMD Ryzen 9 9950X, 16 cores / 32 threads        |
| Memory           | 60 GiB available system RAM                     |
| Operating system | Ubuntu Linux, kernel 6.8.0-106-generic, x86-64  |
| Compiler         | GCC/G++ 11.4.0, C++17 release extension (`-O3`) |
| Python stack     | Python 3.10.0, NumPy 2.2.6, pybind11 3.1.0      |

The benchmark runs in one Python process with logging disabled. It uses 20,000
timed calls for each microbenchmark and three passes over the included
12-second, 30 Hz collision trajectory for the full safety test. Reported values
are medians across three repeats; the command also prints the observed latency
range. CPU frequency scaling and other system load were not pinned, so
small run-to-run variation is expected.

### Reference results

| Workload                                |  Python throughput |  Python latency |     C++ throughput |     C++ latency | C++ speedup |
| --------------------------------------- | -----------------: | --------------: | -----------------: | --------------: | ----------: |
| Single-arm SEW solve                    |   1,050.8 solves/s | 0.9516 ms/solve | 223,628.9 solves/s | 0.0045 ms/solve |      212.8x |
| Seven-capsule bimanual minimum distance | 20,222.0 queries/s | 0.0495 ms/query | 61,311.6 queries/s | 0.0163 ms/query |        3.0x |
| Complete bimanual safety frame          |     580.5 frames/s | 1.7226 ms/frame |   8,306.5 frames/s | 0.1204 ms/frame |       14.3x |

At a 60 Hz control target, the complete safety pipeline provides approximately
`9.7x` real-time throughput with the Python backend and `138.4x` with the C++
backend on this reference system. These figures exclude visualization, robot
communication, and sensor preprocessing. The full safety result also includes
Python-side URDF FK and orchestration, so it does not represent only native C++
execution time.

Reproduce the table after installing the project with its native extension:

```bash
python -m benchmarks.benchmark_backends --iterations 20000 --trajectory-fps 30 --repeats 3
```

### Jetson Orin NX status

Jetson Orin NX performance has not yet been measured in this repository. The
desktop results above must not be used as an Orin NX latency estimate: CPU
architecture, power mode, clock policy, memory bandwidth, compiler, and thermal
state differ substantially. Run the same command on the target and record the
JetPack version, Orin NX memory variant, `nvpmodel` mode, clock policy, and
cooling state alongside the output. A future measured row should use this form:

| Platform       | Power/clock mode | Backend      | SEW solve | Safety frame | Status       |
| -------------- | ---------------- | ------------ | --------: | -----------: | ------------ |
| Jetson Orin NX | To be recorded   | Python / C++ |         — |            — | Not measured |

Rendering performance is intentionally separate from solver performance because
Viser update rate depends on browser, network, mesh complexity, and scene size.

### Performance optimization methodology

Optimize against the deployment pipeline, not only the closed-form kernel. Use
the following loop for every performance change:

1. Select a control target such as 60, 100, or 200 Hz and define what one frame
   includes: input conversion, FK, collision sampling, XPBD, SEW recovery, final
   validation, and command serialization.
1. Profile a representative continuous trajectory with logging disabled. Keep
   initialization/OOBB fitting outside the timed region and report mean, P95,
   worst-case latency, active-constraint count, and failure status.
1. Remove repeated parsing, topology traversal, immutable geometry construction,
   and temporary arrays before moving more code to C++.
1. Compare Python and C++ results after every native optimization. Numerical
   agreement and safety status are release requirements, not optional checks.
1. Repeat measurements across several processes/runs and record compiler flags,
   CPU governor, power mode, clocks, temperature, and background load.

The next high-value native optimization is a robot-specific or generic C++ FK
plan that consumes the 14 bimanual joint values and emits only the eight safety
keypoints and two tool rotations. This would remove most remaining Python matrix
and dictionary work from the C++ safety path. After that, consider one native
`filter_frame()` boundary that performs FK, capsule queries, XPBD, reconstruction,
and validation without round trips through Python. Preserve the current Python
implementation as the readable reference and parity oracle.

For batch/offline retargeting, prefer `solve_batch()` and preallocated contiguous
`float64` arrays. For online control, reuse robot models, safety configurations,
FK plans, collision pairs, and output buffers. Do not share mutable scratch
buffers across control threads unless ownership is explicit; allocation removal
must not introduce data races.

On Jetson Orin NX, benchmark native AArch64 builds in each intended `nvpmodel`
mode with the production cooling setup. Optimize worst-case full safety-frame
latency before solver throughput. GPU offload is unlikely to help individual
seven-DoF closed-form solves; it becomes relevant only when upstream perception
or large batched workloads already reside on the GPU.

## Logging

The package uses standard-library `logging` and does not configure handlers on
import.

```python
from sew_mimic import configure_logging

configure_logging("INFO")  # model loading, OOBB initialization, summaries
# configure_logging("DEBUG")  # per-frame FK, collision, XPBD, and SEW stages
```

Use `INFO` or `WARNING` for runtime applications. `DEBUG` intentionally emits
detailed per-stage/per-iteration messages and affects benchmark results.

## Repository layout

```text
SEW-mimic-retargeting/
├── src/sew_mimic/
│   ├── __init__.py                    Public API
│   ├── solver.py                      Closed-form SEW solver and robot model
│   ├── utility.py                     Rotation, validation, and angle utilities
│   ├── collision.py                   Capsule geometry and OOBB fitting
│   ├── safety.py                      Continuous collision and XPBD filter
│   └── robots/
│       ├── marvin_m6.py               Marvin adapter and high-level filter
│       ├── openarm.py                  OpenArm seven-DoF adapter
│       ├── arm_loader.py               Generic serial seven-DoF URDF extraction
│       ├── capsule_config.py            Generic mesh-OOBB capsule estimation
│       ├── registry.py                 Unified robot selection API
│       ├── demo_profiles.py            Registered collision validation paths
│       ├── pose.py                     Generic bimanual URDF landmark extraction
│       ├── safety_filter.py            Shared bimanual safety orchestration
│       └── urdf.py                    Dependency-free general URDF FK
├── examples/
│   ├── demo_toy.py
│   ├── demo_robot_viser.py             Multi-robot implementation
│   ├── demo_marvin_viser.py            Legacy Marvin command shim
│   ├── demo_robot_collision_avoidance.py Multi-robot collision demo
│   └── demo_marvin_collision_avoidance.py Legacy Marvin command shim
├── tests/
│   ├── unit/
│   │   ├── test_solver.py
│   │   ├── test_utility.py
│   │   ├── test_collision.py
│   │   └── test_safety.py
│   └── integration/test_marvin_m6.py
├── assets/Marvin_M6_S_CCS_696_V4.0/
│   ├── robot_with_ee.urdf
│   ├── collision/
│   └── visual/
├── assets/OpenArm/                     OpenArm URDF and mesh assets
├── docs/media/marvin_viser.gif
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Development

Run all tests:

```bash
pytest -q
```

Run test groups independently:

```bash
pytest -q tests/unit
pytest -q tests/integration
```

Format and lint:

```bash
ruff format src examples benchmarks scripts tests setup.py
mdformat README.md
clang-format -i src/cpp/sew_mimic_cpp.cpp
ruff check src examples benchmarks scripts tests setup.py
clang-format --dry-run --Werror src/cpp/sew_mimic_cpp.cpp
```

Build a wheel containing the bundled Marvin and OpenArm assets:

```bash
python -m build --wheel
python scripts/verify_wheel.py dist/*.whl --assets assets --expect-pure
```

After installing the wheel into a virtual environment, run the installed-package
smoke test from outside the repository:

```bash
cd /tmp
/path/to/venv/bin/python /path/to/repository/scripts/smoke_test_install.py
```

GitHub Actions checks Python, C++, and Markdown formatting, runs the reference
and native parity tests, verifies packaged robot assets, and builds Python and
native wheels on every push and pull request.

## Current limitations

- The core solver implements the paper's parallel-wrist path; the perpendicular-
  wrist Euler-decomposition appendix path is not implemented.
- The core objective matches directions and orientation, not absolute tool
  position.
- The safety filter handles configured robot self-collision capsules; it does
  not model environment obstacles.
- Capsule/OOBB geometry is an approximation of the original mesh geometry.
- Like the paper's proposed filter, the XPBD safety layer is heuristic and does
  not provide a formal collision-avoidance guarantee.
- The included demo trajectories are procedural rather than motion-capture data.

## Paper reproduction notes

The implementation resolves several notation and printing inconsistencies in
the paper pseudocode:

- SP4 follows the IK-Geo argument order `SP4(h, p, k, d)`.
- `MakeFrame` returns `[ux, uy, uz]`, correcting the duplicated `uy` in print.
- `AlignAxis` uses the frame-consistent expression `R^(i-2,i-1) h_(i-1)`.
- Analytical angles are expanded by equivalent `2*pi` rotations, filtered by
  joint limits, and selected by distance from `q_current`.
- Wrist alignment uses `R07_des @ h7`, supporting different local-axis conventions.

## Citation

If this repository or the underlying method is useful in your work, cite the
original paper:

```bibtex
@article{kong2026closed_form_geometric_retargeting,
  author  = {Kong, Chuizheng and Cho, Yunho and Jung, Wonsuhk and
             Wibowo, Idris and Shinde, Parth and Vinodh-Sangeetha, Sundhar and
             Chung, Long Kiu and Chen, Zhenyang and Mattei, Andrew and
             Nidumukkala, Advaith and Elias, Alexander and Xu, Danfei and
             Higgins, Taylor and Kousik, Shreyas},
  title   = {A Closed-Form Geometric Retargeting Solver for Upper Body
             Humanoid Robot Teleoperation},
  journal = {arXiv preprint arXiv:2602.01632},
  year    = {2026},
  month   = {February},
  doi     = {10.48550/arXiv.2602.01632}
}
```

DOI: [10.48550/arXiv.2602.01632](https://doi.org/10.48550/arXiv.2602.01632)
