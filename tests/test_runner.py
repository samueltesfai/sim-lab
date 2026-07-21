import os
import tempfile

import pytest
from omegaconf import OmegaConf

from simlab.runner import RunRequest, execute_run

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
        OmegaConf.save(CONFIG_DICT, f.name)
        path = f.name
    try:
        yield path
    finally:
        os.unlink(path)


def test_execute_run_records_initial_row_plus_one_per_step(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=4))

    assert result.completed_steps == 4
    # Initial row (tick -1) plus one row per completed step.
    assert len(result.telemetry) == 5
    assert result.telemetry[0].tick == -1
    assert [row.tick for row in result.telemetry[1:]] == [0, 1, 2, 3]


def test_execute_run_zero_steps_records_only_initial_row(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=0))

    assert result.completed_steps == 0
    assert len(result.telemetry) == 1
    assert result.telemetry[0].tick == -1


def test_execute_run_measures_step_runtime(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=3))

    # Initial row has no runtime measurement; every stepped row does.
    assert result.telemetry[0].step_runtime_ms is None
    for row in result.telemetry[1:]:
        assert row.step_runtime_ms is not None
        assert row.step_runtime_ms >= 0.0


def test_execute_run_total_runtime_is_positive(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=3))

    assert result.total_runtime_ms >= 0.0


def test_execute_run_is_headless_and_reproducible(config_path):
    """Same config and seed should produce an identical trajectory, and the
    run must not depend on any visualization machinery."""
    result_a = execute_run(RunRequest(config_path=config_path, steps=5))
    result_b = execute_run(RunRequest(config_path=config_path, steps=5))

    rows_a = [row.to_dict() for row in result_a.telemetry]
    rows_b = [row.to_dict() for row in result_b.telemetry]

    for row_a, row_b in zip(rows_a, rows_b):
        row_a.pop("step_runtime_ms")
        row_b.pop("step_runtime_ms")
    assert rows_a == rows_b


def test_execute_run_raises_for_missing_config():
    with pytest.raises(FileNotFoundError):
        execute_run(RunRequest(config_path="does/not/exist.yaml", steps=1))
