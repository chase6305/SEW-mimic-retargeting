from pathlib import Path

import numpy as np
import pytest

from sew_mimic import Capsule
from sew_mimic.robots import CapsuleMeshSpec, estimate_capsule_config_from_meshes
from sew_mimic.robots import capsule_config as capsule_config_module


def test_mesh_capsule_estimation_fits_each_unique_mesh_once(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fake_fit(path, *, padding, name):
        calls.append(Path(path).name)
        radius = {"torso.obj": 0.2, "upper.obj": 0.1, "lower.obj": 0.08, "hand.obj": 0.06}[
            Path(path).name
        ]
        return Capsule(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), radius)

    class FakeKinematics:
        def __init__(self, path):
            self.path = path

        def link_transforms(self, positions, required_links):
            assert required_links == ("torso",)
            return {"torso": np.eye(4)}

    monkeypatch.setattr(capsule_config_module, "capsule_from_oobb", fake_fit)
    monkeypatch.setattr(capsule_config_module, "URDFKinematics", FakeKinematics)
    spec = CapsuleMeshSpec(
        torso="torso.obj",
        upper_arm=("upper.obj",),
        lower_arm=("lower.obj",),
        hand=("hand.obj", "torso.obj"),
        torso_link="torso",
    )

    config = estimate_capsule_config_from_meshes(
        tmp_path / "robot.urdf", tmp_path / "collision", spec
    )

    assert calls.count("torso.obj") == 1
    assert sorted(calls) == ["hand.obj", "lower.obj", "torso.obj", "upper.obj"]
    assert config.radii.torso == pytest.approx(0.2)
    assert config.radii.hand == pytest.approx(0.2)
