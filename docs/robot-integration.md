# Robot integration

Validate kinematics before visualization, and visualization before collision correction.
See [usage](usage.md) for the input contract and [architecture](architecture.md) for prepared FK.

## 1. Check the kinematic contract

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

## 2. Implement and register the arm adapter

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

## 3. Validate retargeting before safety

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

## 4. Add collision geometry and a safety wrapper

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
Use `URDFBimanualPoseEvaluator` to prepare the real-time FK path once.
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
