# SEW-Mimic Retargeting

A Python/NumPy implementation of the closed-form geometric retargeting method
from **A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot
Teleoperation**, with a repository-local Marvin M6 model, dual-arm trajectory
demo, and interactive viser visualization.

The solver retargets human upper-arm and forearm directions together with hand
orientation to a seven-DoF robot arm. It uses closed-form geometric subproblems
instead of iterative numerical optimization, making it suitable for low-latency
teleoperation and real-time motion-retargeting prototypes.

## Demo video

[![Marvin M6 viser demo](docs/media/marvin_viser.gif)](docs/media/marvin_viser.mp4)

## Features

- Rodrigues rotation formula.
- IK-Geo geometric Subproblems 1, 2, and 4.
- Closed-form two-joint `AlignAxis`.
- Parallel-wrist `AlignWrist`.
- Single-arm, seven-DoF SEW-Mimic solver.
- Joint-limit filtering and closest analytical-solution selection.
- Automatic Marvin M6 left/right arm adaptation from URDF.
- Continuous dual-arm trajectory generation and frame-by-frame retargeting.
- viser visualization of the URDF, human keypoints, bones, and coordinate frames.
- Solver throughput and latency measurements.

The core solver matches directions and orientation rather than absolute human
limb lengths:

```text
upper-arm direction = normalize(elbow - shoulder)
forearm direction   = normalize(wrist - elbow)
end-effector target = hand_orientation
```

## Repository layout

```text
SEW-mimic-retargeting/
├── src/sew_mimic/
│   ├── __init__.py                    Public Python API
│   ├── solver.py                      Closed-form SEW-Mimic solver
│   └── robots/
│       ├── __init__.py
│       └── marvin_m6.py               Marvin M6 URDF adapter
├── examples/
│   ├── demo_toy.py                    Synthetic FK-to-SEW example
│   └── demo_marvin_viser.py           Dual-arm viser demo
├── tests/
│   ├── unit/test_solver.py            Geometric solver unit tests
│   └── integration/test_marvin_m6.py  URDF and dual-arm integration tests
├── assets/Marvin_M6_S_CCS_696_V4.0/
│   ├── robot_with_ee.urdf             Default complete robot model
│   ├── collision/                     Collision meshes
│   └── visual/                        GLB and STL visual meshes
├── docs/media/
│   ├── marvin_viser.gif               README demo preview
│   └── marvin_viser.mp4               Higher-quality demo video
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Installation

Python 3.10 or newer is required.

Install only the core package:

```bash
pip install -e .
```

Install test dependencies:

```bash
pip install -e '.[test]'
```

Install viser visualization dependencies:

```bash
pip install -e '.[visualization]'
```

Install everything needed for development:

```bash
pip install -e '.[dev,visualization]'
```

## Formatting and linting

The repository uses [Ruff](https://docs.astral.sh/ruff/) for Python formatting,
import sorting, and lightweight linting. The configuration is stored in
`pyproject.toml`; general editor whitespace rules are stored in `.editorconfig`.

Format all Python code:

```bash
ruff format src examples tests
```

Check formatting without changing files:

```bash
ruff format --check src examples tests
```

Run lint checks and apply safe automatic fixes:

```bash
ruff check --fix src examples tests
```

Run the same checks without modifying files:

```bash
ruff check src examples tests
```

## Core API

### Load a Marvin arm

```python
from sew_mimic.robots import load_marvin_arm

left_arm = load_marvin_arm(side="left")
right_arm = load_marvin_arm(side="right")
```

The adapter reads the repository-local
`assets/Marvin_M6_S_CCS_696_V4.0/robot_with_ee.urdf` and extracts:

- the ordered seven-joint arm chain;
- local joint axes and zero-configuration URDF RPY rotations;
- lower and upper joint limits;
- parent/child chain topology; and
- the fixed `LEFT_EE` or `RIGHT_EE` transform.

An alternative URDF can be supplied explicitly:

```python
arm = load_marvin_arm("/path/to/robot_with_ee.urdf", side="left")
```

### Solve one frame

```python
import numpy as np

from sew_mimic import sew_mimic
from sew_mimic.robots import load_marvin_arm

arm = load_marvin_arm(side="left")

q0 = np.zeros(7)                       # Current joint angles [rad]
shoulder = np.array([0.0, 0.0, 0.0])
elbow = np.array([0.20, 0.20, 0.05])
wrist = np.array([0.45, 0.25, 0.10])
hand_orientation = np.eye(3)           # Valid SO(3) rotation matrix

q = sew_mimic(
    arm.robot,
    q0,
    shoulder,
    elbow,
    wrist,
    hand_orientation,
)
```

The returned joint order is:

```text
SHOULDER_PITCH
SHOULDER_ROLL
ELBOW_PITCH
ELBOW_YAW
WRIST_PITCH
WRIST_YAW
WRIST_ROLL
```

### Input conventions

- Left-arm inputs must be expressed in `left_arm_base`.
- Right-arm inputs must be expressed in `right_arm_base`.
- Shoulder, elbow, wrist, and hand orientation must share one coordinate frame.
- Position inputs should normally use meters. Absolute human limb length does
  not affect the direction-only arm objective.
- `q0` and returned joint angles are in radians.
- Shoulder-to-elbow and elbow-to-wrist vectors must not be near zero.
- `hand_orientation` must be a valid 3-by-3 SO(3) rotation matrix.

For human inputs expressed in a world frame:

```python
R_arm_world = R_world_arm.T

shoulder_arm = R_arm_world @ (shoulder_world - p_world_arm)
elbow_arm = R_arm_world @ (elbow_world - p_world_arm)
wrist_arm = R_arm_world @ (wrist_world - p_world_arm)
hand_orientation_arm = R_arm_world @ hand_orientation_world
```

## Examples

### Synthetic consistency demo

```bash
python examples/demo_toy.py
```

This demo performs FK from known joint angles, constructs corresponding SEW
targets, and recovers the joint angles through the closed-form solver. Upper-arm,
forearm, and tool-orientation errors should be near floating-point precision.

### Marvin M6 dual-arm viser demo

```bash
python examples/demo_marvin_viser.py --side both
```

Default settings:

```text
Trajectory duration    60 s
Trajectory rate        60 Hz
Playback speed         1.75x
Arms                   left + right
viser port             8080
Loop playback          enabled
```

All options can be overridden:

```bash
python examples/demo_marvin_viser.py \
  --side both \
  --duration 60 \
  --motion-cycle 60 \
  --fps 60 \
  --playback-speed 1.75 \
  --port 8080
```

The generated motion program contains two repetitions of each group:

1. Chest-front crossing and chest expansion.
2. A large asymmetric swing with the left arm high and right arm low.
3. Alternating punches while the opposite hand remains in guard.
4. Synchronized zombie-style push/pull: wrists remain level and in fixed
   lateral lanes while the elbows flex and extend.
5. Alternating running-style arm swings.

The viser interface provides:

- the complete Marvin URDF model;
- fourteen left/right arm joint sliders;
- colored shoulder, elbow, and wrist target keypoints;
- upper-arm and forearm bone segments;
- arm-base, joint, and enlarged end-effector coordinate frames; and
- play, pause, loop, restart, timeline, and playback-speed controls.

The default URDF and meshes are resolved relative to this repository. To use a
different model path:

```bash
python examples/demo_marvin_viser.py --urdf /path/to/robot_with_ee.urdf
```

## Performance metrics

The viser panel reports:

```text
Single-arm solve rate [Hz]
Dual-arm pose-pair rate [pairs/s]
Mean solve latency [ms/arm]
```

- `Hz` is the number of individual arm solves completed per second.
- `pairs/s` is the number of complete dual-arm pose pairs per second when the
  left and right arms are solved sequentially.
- `ms/arm` is the mean latency of one single-arm solve.

These metrics time only the closed-form SEW-Mimic call. They exclude URDF FK,
viser scene updates, network communication, and browser rendering.

## Tests

Run the complete suite:

```bash
pytest -q
```

Run test groups independently:

```bash
pytest -q tests/unit
pytest -q tests/integration
```

The suite covers:

- numerical correctness of SP1, SP2, and SP4;
- randomized FK-to-SEW-to-solver consistency;
- Marvin left/right URDF extraction and axis morphology;
- joint-limit compliance;
- dual-arm trajectory continuity and retargeting accuracy; and
- human keypoint limb lengths.

## Current limitations

- The core implementation currently supports the paper's parallel-wrist path.
- The perpendicular-wrist Euler-decomposition appendix path is not implemented.
- The XPBD/capsule safety filter is not implemented.
- The demo does not enforce robot self-collision or environment collision.
- The core method retargets directions and orientation; it is not an absolute
  end-effector position IK solver.
- The included trajectory is procedurally generated and is not motion-capture data.

## Notes on the paper pseudocode

The implementation resolves several notation and printing inconsistencies in
the paper pseudocode:

- SP4 follows the IK-Geo argument order `SP4(h, p, k, d)`.
- `MakeFrame` returns `[ux, uy, uz]`, correcting the duplicated `uy` in print.
- `AlignAxis` uses the frame-consistent expression
  `R^(i-2,i-1) h_(i-1)`.
- Analytical angles are expanded by equivalent `2*pi` rotations, filtered by
  limits, and selected by distance from `q0`.
- Wrist alignment uses the coordinate-free expression `R07_des @ h7`, allowing
  different local axis conventions.

## Citation

If this repository or the underlying method is useful in your work, please cite
the original paper:

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
