"""Backward-compatible Marvin M6 entry point.

Use ``demo_robot_viser.py --robot marvin`` for new integrations.
"""

from __future__ import annotations

import sys

if __package__:
    from .demo_robot_viser import main
else:
    from demo_robot_viser import main


if __name__ == "__main__":
    if "--robot" not in sys.argv:
        sys.argv[1:1] = ["--robot", "marvin"]
    main()
