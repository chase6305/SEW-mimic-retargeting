# Solver optimization measurements (2026-09-05)

The comparison uses commit `7b4872d` as the baseline and the optimized working
tree. Both native extensions were compiled with the same GCC toolchain and
`-O3` settings. Measurements ran sequentially on the same AMD Ryzen 9 9950X
machine using Python 3.10.0 and NumPy 2.2.6. CPU affinity and frequency were
not pinned; small differences should be treated as measurement noise.

Each value is the median of five repeats after warmup. Single-frame and
relative-FK queries use 1,000 calls per repeat. Ordered solve batches contain
256 frames; each batch starts from zero and carries the previous solution
forward. Safety measurements use the bundled 12-second Marvin collision
trajectory sampled at 10 Hz. Timings exclude visualization and robot I/O.

| Workload                                        | Before (ms/operation) | After (ms/operation) | Speedup |
| ----------------------------------------------- | --------------------: | -------------------: | ------: |
| Python relative FK, frame 3 to frame 5          |              0.073485 |             0.018790 |   3.91x |
| Python single-arm solve                         |              0.960132 |             0.499276 |   1.92x |
| Python ordered batch, per arm frame             |              0.955602 |             0.495202 |   1.93x |
| C++ single-arm solve                            |              0.004415 |             0.002672 |   1.65x |
| C++ ordered batch, per arm frame                |              0.003112 |             0.001290 |   2.41x |
| Python complete bimanual safety frame           |              1.687964 |             1.543384 |   1.09x |
| C++ complete bimanual safety frame              |              0.120568 |             0.121216 |   0.99x |
| Python capsule query, including pose validation |              0.051098 |             0.050362 |   1.01x |
| C++ capsule query, including pose validation    |              0.016392 |             0.016592 |   0.99x |

The main gain comes from eliminating repeated base-to-joint FK in axis
alignment. Local axes can be transformed using fixed joint rotations because
rotation about a joint leaves its own axis unchanged. Relative FK now evaluates
only the joints between the requested frames. No model cache or invalidation
requirement was added to the Python model.

The C++ distance-only path also avoids constructing a contact normal. The
capsule-query and complete native safety measurements remain effectively
unchanged: their Python validation and orchestration costs still matter.

Reproduce the workloads with the extended benchmark:

```bash
SEW_MIMIC_BUILD_CPP=1 python setup.py build_ext --inplace
PYTHONPATH=src python -m benchmarks.benchmark_backends \
  --iterations 1000 --trajectory-fps 10 --repeats 5 --batch-size 256
```

Use `--backends python` for a pure-Python installation and `--urdf PATH` to
select the Marvin asset path explicitly. By default the benchmark measures
the installed backends; explicitly requesting an unavailable backend fails
before any timings are collected.

Correctness checks cover randomized local joint-frame conventions, forward
and reverse relative rotations, Python/C++ parity, ordered trajectories, and
parallel native calls. Additional regressions verify that rejected bimanual
poses do not advance filter history, copied safety geometry cannot be changed
through caller-owned arrays, invalid configuration/batch inputs are rejected,
and native interpolation clamps sample counts before integer conversion.

Input contracts are stricter: joint limits, projection compliance, and tolerance
must be finite; iteration counts must be positive 32-bit integers; capsule pair
indices must be integers. Direct backend batch calls now perform the same
shape and finite-value checks as `solve_batch()`.

## Follow-up: shared FK preparation and safety hot paths

This second comparison uses the completed first optimization pass as its
baseline. FK timings use 2,000 calls per repeat; safety timings use the
12-second Marvin trajectory at 30 Hz. The table reports five-repeat medians
with warmup, on the same machine. Benchmarks ran after correctness tests.

| Workload                            | First pass (ms) | Refactored (ms) | Speedup |
| ----------------------------------- | --------------: | --------------: | ------: |
| Marvin Python bimanual FK           |        0.052795 |        0.025843 |   2.04x |
| Marvin C++ bimanual FK              |        0.052397 |        0.007384 |   7.10x |
| OpenArm Python bimanual FK          |        0.052762 |        0.025681 |   2.05x |
| OpenArm C++ bimanual FK             |        0.052486 |        0.007402 |   7.09x |
| Marvin Python complete safety frame |        1.556609 |        1.224810 |   1.27x |
| Marvin C++ complete safety frame    |        0.123175 |        0.043670 |   2.82x |

The [architecture guide](architecture.md) describes the common preparation
stage and per-call execution. The Python executor constructs local rotations
in one NumPy batch; the native executor evaluates the same indexed tree.
Scalar contact calculations avoid NumPy scalar dispatch and defer surface
point construction to the public contact-report API. Tool orientation recovery
uses the cross product's sine directly and handles opposite directions without
normalizing a degenerate axis.

Reproduce the Marvin workloads with:

```bash
PYTHONPATH=src python -m benchmarks.benchmark_backends \
  --iterations 2000 --trajectory-fps 30 --repeats 5 --backends python cpp
```

OpenArm FK uses the same joint-vector inputs to isolate the evaluator cost:

```python
from benchmarks.benchmark_backends import timed
from examples.demo_robot_collision_avoidance import collision_test_trajectory
from sew_mimic.robots import OpenArmSafetyFilter

_, left, right = collision_test_trajectory(12.0, 30.0)
for backend in ("python", "cpp"):
    robot = OpenArmSafetyFilter(backend=backend)
    print(backend, timed(lambda: robot.forward_kinematics(left[0], right[0]), 2000, 5))
```
