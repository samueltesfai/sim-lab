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

    assert result.metadata.completed_steps == 4
    # Initial row (tick -1) plus one row per completed step.
    assert len(result.telemetry) == 5
    assert result.telemetry[0].tick == -1
    assert [row.tick for row in result.telemetry[1:]] == [0, 1, 2, 3]


def test_execute_run_zero_steps_records_only_initial_row(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=0))

    assert result.metadata.completed_steps == 0
    assert len(result.telemetry) == 1
    assert result.telemetry[0].tick == -1


def test_run_request_rejects_negative_steps(config_path):
    with pytest.raises(ValueError):
        RunRequest(config_path=config_path, steps=-1)


def test_execute_run_measures_step_runtime(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=3))

    # Initial row has no runtime measurement; every stepped row does.
    assert result.telemetry[0].step_runtime_ms is None
    for row in result.telemetry[1:]:
        assert row.step_runtime_ms is not None
        assert row.step_runtime_ms >= 0.0


def test_execute_run_total_runtime_is_positive(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=3))

    assert result.summary.total_runtime_ms >= 0.0


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


def test_execute_run_populates_metadata(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=2))
    metadata = result.metadata

    assert metadata.schema_version
    assert metadata.run_id
    assert metadata.config_path == config_path
    assert metadata.world_seed == 42
    assert metadata.requested_steps == 2
    assert metadata.completed_steps == 2
    assert metadata.num_agents == 5
    assert metadata.num_claims == 2
    assert metadata.profile_counts == {"default": 5}
    assert metadata.resolved_config["world"]["rng_seed"] == 42
    assert metadata.resolved_config["agent"]["profiles"][0]["count"] == 5


def test_execute_run_generates_run_id_when_not_provided(config_path):
    result_a = execute_run(RunRequest(config_path=config_path, steps=1))
    result_b = execute_run(RunRequest(config_path=config_path, steps=1))

    assert result_a.metadata.run_id != result_b.metadata.run_id


def test_execute_run_uses_provided_run_id(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=1, run_id="my-run"))

    assert result.metadata.run_id == "my-run"


def test_execute_run_config_fingerprint_is_stable(config_path):
    result_a = execute_run(RunRequest(config_path=config_path, steps=1))
    result_b = execute_run(RunRequest(config_path=config_path, steps=1))

    assert result_a.metadata.config_fingerprint == result_b.metadata.config_fingerprint


def test_execute_run_config_fingerprint_changes_with_config(config_path):
    other_config = {**CONFIG_DICT, "world": {**CONFIG_DICT["world"], "rng_seed": 7}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(other_config, f.name)
        other_path = f.name

    try:
        result_a = execute_run(RunRequest(config_path=config_path, steps=1))
        result_b = execute_run(RunRequest(config_path=other_path, steps=1))
        assert (
            result_a.metadata.config_fingerprint != result_b.metadata.config_fingerprint
        )
    finally:
        os.unlink(other_path)


def test_execute_run_scenario_matches_world(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=1))
    scenario = result.scenario

    assert scenario["num_agents"] == 5
    assert scenario["num_claims"] == 2
    assert scenario["private_event_rate"] == pytest.approx(0.5)
    assert scenario["global_event_rate"] == pytest.approx(0.1)
    assert scenario["profile_count.default"] == 5
    assert scenario["profile_fraction.default"] == pytest.approx(1.0)
    assert scenario["graph.num_nodes"] == 5
    assert scenario["initial.belief_mean"] == result.telemetry[0].belief_mean


def test_execute_run_summary_matches_trajectory(config_path):
    result = execute_run(RunRequest(config_path=config_path, steps=5))
    final_row = result.telemetry[-1]

    assert result.summary.final_mean_truth_error == final_row.mean_abs_error_to_truth
    assert result.summary.final_mean_trust == final_row.mean_trust
    assert result.summary.total_observations == sum(
        row.num_observations for row in result.telemetry
    )
    assert result.summary.total_runtime_ms >= 0.0
