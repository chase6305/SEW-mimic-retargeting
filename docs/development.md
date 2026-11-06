# Development

Install the development tools and optional demo/safety dependencies:

```bash
pip install -e '.[dev,visualization,safety,cpp-dev]'
```

## Checks

```bash
ruff format --check src examples benchmarks scripts tests setup.py
ruff check src examples benchmarks scripts tests setup.py
mdformat --check README.md docs/*.md
clang-format --dry-run --Werror src/cpp/*.cpp src/cpp/*.h
pytest -q
```

Build the optional extension before running native parity tests:

```bash
SEW_MIMIC_BUILD_CPP=1 pip install -e '.[cpp]'
pytest -q
```

The native formatter is pinned by the `cpp-dev` extra. Formatting commands without
`--check` or `--dry-run --Werror` update their inputs.

## Packaging

Build and verify a pure wheel, including bundled Marvin and OpenArm assets:

```bash
python -m build --wheel
python scripts/verify_wheel.py dist/*.whl --assets assets --expect-pure
```

For a native wheel, set `SEW_MIMIC_BUILD_CPP=1` during the build and verify with
`--expect-native`. Use separate output directories for pure and native builds.

Install the wheel into a fresh virtual environment, then run the smoke test from
outside the checkout:

```bash
cd /tmp
/path/to/venv/bin/python /path/to/repository/scripts/smoke_test_install.py
```

Add `--expect-native` for a native wheel. GitHub Actions runs these checks for both
wheel variants; reference tests cover Python 3.10 and 3.12.

## Paper reproduction notes

The implementation resolves several notation and printing inconsistencies in
the paper pseudocode:

- SP4 follows the IK-Geo argument order `SP4(h, p, k, d)`.
- `MakeFrame` returns `[ux, uy, uz]`, correcting the duplicated `uy` in print.
- `AlignAxis` uses the frame-consistent expression `R^(i-2,i-1) h_(i-1)`.
- Analytical angles are expanded by equivalent `2*pi` rotations, filtered by
  joint limits, and selected by distance from `q_current`.
- Wrist alignment uses `R07_des @ h7`, supporting different local-axis conventions.

See [performance measurements](performance-2026-09-05.md) for benchmark scope and
reproduction commands.
