import csv
import json
import os
import tempfile

import pytest
import yaml

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
        yaml.dump(CONFIG_DICT, f)
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
    assert manifest["scenario_fingerprint"] == run_result.metadata.scenario_fingerprint
    assert manifest["run_spec_fingerprint"] == run_result.metadata.run_spec_fingerprint
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


def test_write_run_artifacts_overwrite_preserves_previous_run_on_rename_failure(
    config_path, tmp_path, mocker
):
    """If the final os.rename() fails after the previous run has already
    been cleared out of the way, the previous (complete, valid) run must
    still be there afterward -- not silently lost."""
    final_dir = os.path.join(str(tmp_path), "same-run")
    first = execute_run(RunRequest(config_path=config_path, steps=1, run_id="same-run"))
    write_run_artifacts(first, str(tmp_path))

    real_rename = os.rename
    renames_onto_final_dir = 0

    def _fail_only_on_first_rename_onto_final_dir(src, dst):
        nonlocal renames_onto_final_dir
        # The first rename() landing on final_dir is the tmp_dir -> final_dir
        # swap-in, which should fail here. The final_dir -> backup_dir
        # displacement lands elsewhere so is unaffected; a second rename
        # onto final_dir (the restore-on-failure moving backup_dir back)
        # must go through untouched.
        if dst == final_dir:
            renames_onto_final_dir += 1
            if renames_onto_final_dir == 1:
                raise OSError("simulated failure landing the replacement")
        return real_rename(src, dst)

    mocker.patch.object(
        os, "rename", side_effect=_fail_only_on_first_rename_onto_final_dir
    )

    second = execute_run(
        RunRequest(config_path=config_path, steps=4, run_id="same-run")
    )
    with pytest.raises(OSError, match="simulated failure landing the replacement"):
        write_run_artifacts(second, str(tmp_path), overwrite=True)

    # The original run's artifacts are still there, not deleted then lost.
    assert os.path.isdir(final_dir)
    with open(os.path.join(final_dir, "trajectory.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # the first run's 1 step + 1 initial row


def test_write_run_artifacts_leaves_no_partial_output_on_failure(
    run_result, tmp_path, mocker
):
    mocker.patch.object(
        experiment_io,
        "_write_trajectory_csv",
        side_effect=RuntimeError("simulated failure"),
    )

    with pytest.raises(RuntimeError):
        write_run_artifacts(run_result, str(tmp_path))

    final_dir = os.path.join(str(tmp_path), "test-run")
    assert not os.path.exists(final_dir)
    # No stray temp directories left behind either.
    assert os.listdir(str(tmp_path)) == []


def test_write_run_artifacts_rejects_non_finite_scenario_feature(config_path, tmp_path):
    """Two profiles with the same-sign, schema-valid-but-extreme action_cost
    silently overflow to inf during aggregation. json.dump's default
    allow_nan=True would otherwise write the non-standard token `Infinity`
    into summary.json -- most JSON parsers besides Python's own reject
    that. Writing must fail loudly instead, leaving no partial output."""
    overflow_config = {
        **CONFIG_DICT,
        "agent": {
            "defaults": {},
            "profiles": [
                {"name": "a", "count": 1, "action_cost": {"VERIFY": 1e308}},
                {"name": "b", "count": 1, "action_cost": {"VERIFY": 1e308}},
            ],
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(overflow_config, f)
        overflow_path = f.name

    try:
        result = execute_run(
            RunRequest(config_path=overflow_path, steps=1, run_id="overflow-run")
        )
        assert result.scenario["agent_action_cost.VERIFY_mean"] == float("inf")

        with pytest.raises(ValueError, match="JSON compliant"):
            write_run_artifacts(result, str(tmp_path))

        assert os.listdir(str(tmp_path)) == []
    finally:
        os.unlink(overflow_path)


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
    run_result, tmp_path, mocker
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

    mocker.patch.object(
        experiment_io,
        "_write_trajectory_csv",
        side_effect=_write_then_simulate_concurrent_writer,
    )

    with pytest.raises(FileExistsError):
        write_run_artifacts(run_result, str(tmp_path))

    # The "other writer's" artifact must survive untouched.
    assert os.path.isfile(os.path.join(final_dir, "sentinel.txt"))


def test_write_run_artifacts_does_not_mask_unrelated_rename_failure(
    run_result, tmp_path, mocker
):
    """An os.rename() failure that isn't actually a run_id collision (disk
    full, permissions, etc.) must surface as-is, not get reinterpreted as
    FileExistsError just because overwrite=False."""
    mocker.patch.object(
        os, "rename", side_effect=OSError("simulated unrelated failure")
    )

    with pytest.raises(OSError, match="simulated unrelated failure"):
        write_run_artifacts(run_result, str(tmp_path))
