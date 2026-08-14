import json

from sew_mimic import __version__
from sew_mimic.cli import deployment_info, main


def test_deployment_info_reports_backends_and_robot_assets():
    info = deployment_info()
    assert info["version"] == __version__
    assert info["backend"]["default"] in ("python", "cpp")
    assert set(info["robots"]) == {"marvin", "openarm"}
    assert all(details["urdf_exists"] for details in info["robots"].values())
    assert all(details["safety_filter"] for details in info["robots"].values())


def test_info_cli_emits_machine_readable_json(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["sew-mimic-info", "--json"])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == __version__
    assert payload["robots"]["marvin"]["urdf_exists"]


def test_deployment_info_can_validate_all_registered_robots():
    info = deployment_info(validate_robots=True)
    assert set(info["validation"]) == {"marvin", "openarm"}
    for report in info["validation"].values():
        assert report["keypoints_finite"]
        assert report["minimum_joint_range"] > 0.0
        assert report["maximum_consecutive_axis_dot"] < 1e-4
