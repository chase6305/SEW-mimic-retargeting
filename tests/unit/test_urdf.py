from pathlib import Path

import numpy as np
import pytest

from sew_mimic.robots.urdf import URDFKinematics


def _branched_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "branched.urdf"
    path.write_text(
        """<robot name="branched">
  <link name="base"/>
  <link name="arm_1"/>
  <link name="tool"/>
  <link name="unused"/>
  <joint name="arm_joint" type="revolute">
    <parent link="base"/><child link="arm_1"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="tool_joint" type="fixed">
    <parent link="arm_1"/><child link="tool"/><origin xyz="1 0 0"/>
  </joint>
  <joint name="unused_joint" type="prismatic">
    <parent link="base"/><child link="unused"/><origin xyz="0 2 0"/><axis xyz="0 2 0"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )
    return path


def test_required_link_fk_prunes_unrelated_branches(tmp_path):
    kinematics = URDFKinematics(_branched_urdf(tmp_path))
    positions = {"arm_joint": np.pi / 2.0}

    complete = kinematics.link_transforms(positions)
    pruned = kinematics.link_transforms(positions, ("tool",))

    assert set(complete) == {"base", "arm_1", "tool", "unused"}
    assert set(pruned) == {"base", "arm_1", "tool"}
    assert np.allclose(pruned["tool"], complete["tool"])


def test_required_link_fk_rejects_unknown_targets(tmp_path):
    kinematics = URDFKinematics(_branched_urdf(tmp_path))
    with pytest.raises(KeyError, match="missing"):
        kinematics.link_transforms({}, ("missing",))


def test_prismatic_axis_is_normalized_once_and_applied_in_fk(tmp_path):
    kinematics = URDFKinematics(_branched_urdf(tmp_path))
    transforms = kinematics.link_transforms({"unused_joint": 0.5}, ("unused",))
    assert transforms["unused"][:3, 3] == pytest.approx([0.0, 2.5, 0.0])
