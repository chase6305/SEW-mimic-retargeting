"""Backward-compatible Marvin collision-avoidance entry point."""

from __future__ import annotations

import sys

if __package__:
    from .demo_robot_collision_avoidance import main
else:
    from demo_robot_collision_avoidance import main


if __name__ == "__main__":
    if "--robot" not in sys.argv:
        sys.argv[1:1] = ["--robot", "marvin"]
    main()
