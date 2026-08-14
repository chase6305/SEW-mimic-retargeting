"""Installed-package diagnostics for deployments and issue reports."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict

from . import __version__, backend_status
from .robots import available_robots, get_robot_adapter, validate_robot_adapter


def deployment_info(*, validate_robots: bool = False) -> dict[str, object]:
    """Return serializable runtime, backend, robot, and asset diagnostics."""
    robots = {}
    for name in available_robots():
        adapter = get_robot_adapter(name)
        urdf = adapter.default_urdf
        robots[name] = {
            "display_name": adapter.display_name,
            "urdf": str(urdf),
            "urdf_exists": urdf.is_file(),
            "safety_filter": adapter.safety_filter_factory is not None,
        }
    info = {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prefix": sys.prefix,
        "backend": backend_status(),
        "robots": robots,
    }
    if validate_robots:
        info["validation"] = {
            name: asdict(validate_robot_adapter(name)) for name in available_robots()
        }
    return info


def _human_readable(info: dict[str, object]) -> str:
    backend = info["backend"]
    robots = info["robots"]
    assert isinstance(backend, dict) and isinstance(robots, dict)
    lines = [
        f"SEW-Mimic {info['version']}",
        f"Python: {info['python']}",
        f"Platform: {info['platform']}",
        f"Install prefix: {info['prefix']}",
        f"Default backend: {backend['default']}",
        f"C++ backend: {backend['cpp_implementation']}",
        "Robots:",
    ]
    for name, details in robots.items():
        assert isinstance(details, dict)
        status = "ok" if details["urdf_exists"] else "missing"
        lines.append(f"  {name}: {details['display_name']} [{status}] {details['urdf']}")
    validation = info.get("validation")
    if isinstance(validation, dict):
        lines.append("Validation:")
        for name, report in validation.items():
            assert isinstance(report, dict)
            lines.append(
                f"  {name}: finite={report['keypoints_finite']} "
                f"min_joint_range={report['minimum_joint_range']:.3f} rad "
                f"axis_dot={report['maximum_consecutive_axis_dot']:.3g}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--validate-robots",
        action="store_true",
        help="load both arms and run generic URDF/FK validation",
    )
    args = parser.parse_args()
    info = deployment_info(validate_robots=args.validate_robots)
    print(json.dumps(info, indent=2, sort_keys=True) if args.json else _human_readable(info))
    missing = [name for name, details in info["robots"].items() if not details["urdf_exists"]]
    if missing:
        parser.exit(1, f"Missing robot assets: {missing}\n")


if __name__ == "__main__":
    main()
