import csv
import json
import os
import tempfile

import pytest
from omegaconf import OmegaConf

import simlab.experiment_io as experiment_io
from simlab.experiment_io import write_run_artifacts
from simlab.runner import RunRequest, execute_run
from simlab.telemetry import TelemetryRow

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


@pytest.fixture
def run_result(config_path):
    return execute_run(RunRequest(config_path=config_path, steps=3, run_id="test-run"))


def test_write_run_artifacts_creates_expected_files(run_result, tmp_path):
    run_dir = write_run_artifacts(run_result, str(tmp_path))

    assert run_dir == os.path.join(str(tmp_path), "test-run")
    assert os.path.isfile(os.path.join(run_dir, "manifest.json"))
    assert os.path.isfile(os.path.join(run_dir, "summary.json"))
    assert os.path.isfile(os.path.join(run_dir, "trajectory.csv"))


def test_manifest_json_matches_metadata(run_result, tmp_path):
    run_dir = write_run_artifacts(run_result, str(tmp_path))

    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == run_result.metadata.schema_version
    assert manifest["run_id"] == "test-run"
    assert manifest["config_fingerprint"] == run_result.metadata.config_fingerprint
    assert manifest["world_seed"] == 42
    assert manifest["completed_steps"] == 3
    assert manifest["profile_counts"] == {"default": 5}
    assert manifest["resolved_config"]["world"]["rng_seed"] == 42


def test_summary_json_has_scenario_outcomes_and_labels(run_result, tmp_path):
    run_dir = write_run_artifacts(run_result, str(tmp_path))

    with open(os.path.join(run_dir, "summary.json")) as f:
        summary_doc = json.load(f)

    assert summary_doc["run_id"] == "test-run"
    assert summary_doc["scenario"] == run_result.scenario
    assert summary_doc["labels"] == {
        "converged": run_result.summary.converged,
        "final_consensus": run_result.summary.final_consensus,
        "final_truth_aligned": run_result.summary.final_truth_aligned,
        "final_false_consensus": run_result.summary.final_false_consensus,
    }
    # Outcomes should carry the continuous metrics but not duplicate labels.
    assert "final_mean_truth_error" in summary_doc["outcomes"]
    for label_field in summary_doc["labels"]:
        assert label_field not in summary_doc["outcomes"]


def test_trajectory_csv_matches_telemetry(run_result, tmp_path):
    run_dir = write_run_artifacts(run_result, str(tmp_path))

    with open(os.path.join(run_dir, "trajectory.csv"), newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    assert len(rows) == len(run_result.telemetry)
    assert columns == set(TelemetryRow.__annotations__.keys())
    assert rows[0]["tick"] == "-1"


def test_write_run_artifacts_refuses_overwrite_by_default(run_result, tmp_path):
    write_run_artifacts(run_result, str(tmp_path))

    with pytest.raises(FileExistsError):
        write_run_artifacts(run_result, str(tmp_path))


def test_write_run_artifacts_overwrite_replaces_existing_run(config_path, tmp_path):
    first = execute_run(RunRequest(config_path=config_path, steps=1, run_id="same-run"))
    write_run_artifacts(first, str(tmp_path))

    second = execute_run(
        RunRequest(config_path=config_path, steps=4, run_id="same-run")
    )
    run_dir = write_run_artifacts(second, str(tmp_path), overwrite=True)

    with open(os.path.join(run_dir, "trajectory.csv"), newline="") as f:
        rows = list(csv.DictReader(f))

    # 4 steps + 1 initial row, not the first run's 1 step + 1 initial row.
    assert len(rows) == 5


def test_write_run_artifacts_leaves_no_partial_output_on_failure(
    run_result, tmp_path, monkeypatch
):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(experiment_io, "_write_trajectory_csv", _boom)

    with pytest.raises(RuntimeError):
        write_run_artifacts(run_result, str(tmp_path))

    final_dir = os.path.join(str(tmp_path), "test-run")
    assert not os.path.exists(final_dir)
    # No stray temp directories left behind either.
    assert os.listdir(str(tmp_path)) == []


@pytest.mark.parametrize(
    "bad_run_id", [".", "..", "../evil", "sub/dir", "sub/../../evil"]
)
def test_write_run_artifacts_rejects_unsafe_run_id(config_path, tmp_path, bad_run_id):
    # An empty run_id can't reach here via execute_run: RunRequest.run_id is
    # falsy-or-generated, so it's covered directly against _validate_run_id
    # instead (see test_validate_run_id_rejects_empty_string).
    result = execute_run(
        RunRequest(config_path=config_path, steps=1, run_id=bad_run_id)
    )

    with pytest.raises(ValueError):
        write_run_artifacts(result, str(tmp_path))

    # Nothing was written outside (or even inside) tmp_path.
    parent = os.path.dirname(str(tmp_path))
    assert not os.path.exists(os.path.join(parent, "evil"))
    assert os.listdir(str(tmp_path)) == []


def test_validate_run_id_rejects_empty_string():
    with pytest.raises(ValueError):
        experiment_io._validate_run_id("")


def test_write_run_artifacts_detects_concurrent_writer_when_not_overwriting(
    run_result, tmp_path, monkeypatch
):
    """A second writer must not silently clobber a run that appeared between
    this call's existence check and its final rename, even though neither
    call requested overwrite=True."""
    final_dir = os.path.join(str(tmp_path), "test-run")
    original_write_csv = experiment_io._write_trajectory_csv

    def _write_then_simulate_concurrent_writer(rows, path):
        original_write_csv(rows, path)
        os.makedirs(final_dir)
        with open(os.path.join(final_dir, "sentinel.txt"), "w") as f:
            f.write("winner")

    monkeypatch.setattr(
        experiment_io, "_write_trajectory_csv", _write_then_simulate_concurrent_writer
    )

    with pytest.raises(FileExistsError):
        write_run_artifacts(run_result, str(tmp_path))

    # The "other writer's" artifact must survive untouched.
    assert os.path.isfile(os.path.join(final_dir, "sentinel.txt"))
