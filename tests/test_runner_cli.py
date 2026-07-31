import json
import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from simlab.runner import main

CONFIG_DICT = {
    "world": {
        "rng_seed": 42,
        "observation": {"private_event_rate": 0.5, "global_event_rate": 0.1},
        "truths": {0: True, 1: False},
        "noise": {"OBSERVE": 0.1, "HEAR": 0.15, "VERIFY": 0.05},
    },
    "agent": {
        "defaults": {
            "action_preference": {
                "IDLE": 0.0,
                "VERIFY": 0.9,
                "COMMUNICATE": 0.7,
                "BROADCAST": 0.5,
            },
            "action_cost": {
                "IDLE": 0.0,
                "VERIFY": 0.35,
                "COMMUNICATE": 0.15,
                "BROADCAST": 0.30,
            },
        },
        "profiles": [{"name": "default", "count": 5}],
    },
}


@pytest.fixture
def config_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(CONFIG_DICT, f)
        path = f.name
    try:
        yield path
    finally:
        os.unlink(path)


def test_cli_runs_headlessly_and_writes_artifacts(config_path, tmp_path):
    output_dir = str(tmp_path / "runs")

    with patch(
        "sys.argv",
        [
            "python -m simlab.runner",
            "--config",
            config_path,
            "--steps",
            "3",
            "--output-dir",
            output_dir,
            "--run-id",
            "cli-run",
        ],
    ):
        main()

    run_dir = os.path.join(output_dir, "cli-run")
    assert os.path.isfile(os.path.join(run_dir, "manifest.json"))
    assert os.path.isfile(os.path.join(run_dir, "summary.json"))
    assert os.path.isfile(os.path.join(run_dir, "trajectory.csv"))

    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["run_id"] == "cli-run"
    assert manifest["completed_steps"] == 3


def test_cli_refuses_overwrite_by_default(config_path, tmp_path):
    output_dir = str(tmp_path / "runs")
    argv = [
        "python -m simlab.runner",
        "--config",
        config_path,
        "--steps",
        "1",
        "--output-dir",
        output_dir,
        "--run-id",
        "dup-run",
    ]

    with patch("sys.argv", argv):
        main()

    with patch("sys.argv", argv):
        with pytest.raises(FileExistsError):
            main()


def test_cli_requires_config_steps_and_output_dir():
    with patch("sys.argv", ["python -m simlab.runner"]):
        with pytest.raises(SystemExit):
            main()
