import os
import tempfile

import pytest
import yaml

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
        yaml.dump(CONFIG_DICT, f)
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
    assert metadata.scenario_fingerprint
    assert metadata.run_spec_fingerprint
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


def test_execute_run_fingerprints_are_stable(config_path):
    result_a = execute_run(RunRequest(config_path=config_path, steps=1))
    result_b = execute_run(RunRequest(config_path=config_path, steps=1))

    assert (
        result_a.metadata.scenario_fingerprint == result_b.metadata.scenario_fingerprint
    )
    assert (
        result_a.metadata.run_spec_fingerprint == result_b.metadata.run_spec_fingerprint
    )


def test_execute_run_scenario_fingerprint_ignores_seed(config_path):
    """Different seeds are stochastic replicates of the same scenario, not
    different scenarios -- scenario_fingerprint must not depend on the seed,
    even though run_spec_fingerprint (which identifies a specific requested
    replicate) must."""
    other_config = {**CONFIG_DICT, "world": {**CONFIG_DICT["world"], "rng_seed": 7}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(other_config, f)
        other_path = f.name

    try:
        result_a = execute_run(RunRequest(config_path=config_path, steps=1))
        result_b = execute_run(RunRequest(config_path=other_path, steps=1))

        assert (
            result_a.metadata.scenario_fingerprint
            == result_b.metadata.scenario_fingerprint
        )
        assert (
            result_a.metadata.run_spec_fingerprint
            != result_b.metadata.run_spec_fingerprint
        )
    finally:
        os.unlink(other_path)


def test_execute_run_scenario_fingerprint_changes_with_behavioral_change(config_path):
    """A behavioral change (unlike the seed) must change scenario_fingerprint."""
    other_config = {
        **CONFIG_DICT,
        "world": {
            **CONFIG_DICT["world"],
            "noise": {"OBSERVE": 0.9, "HEAR": 0.9, "VERIFY": 0.9},
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(other_config, f)
        other_path = f.name

    try:
        result_a = execute_run(RunRequest(config_path=config_path, steps=1))
        result_b = execute_run(RunRequest(config_path=other_path, steps=1))

        assert (
            result_a.metadata.scenario_fingerprint
            != result_b.metadata.scenario_fingerprint
        )
    finally:
        os.unlink(other_path)


def test_execute_run_scenario_fingerprint_normalizes_negative_zero(config_path):
    """-0.0 and 0.0 are behaviorally identical; scenario_fingerprint must
    not depend on which sign of zero a config happens to use."""
    negative_zero_config = {
        **CONFIG_DICT,
        "agent": {
            **CONFIG_DICT["agent"],
            "defaults": {
                **CONFIG_DICT["agent"]["defaults"],
                "action_preference": {
                    **CONFIG_DICT["agent"]["defaults"]["action_preference"],
                    "IDLE": -0.0,
                },
            },
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(negative_zero_config, f)
        other_path = f.name

    try:
        result_a = execute_run(RunRequest(config_path=config_path, steps=1))
        result_b = execute_run(RunRequest(config_path=other_path, steps=1))

        assert (
            result_a.metadata.scenario_fingerprint
            == result_b.metadata.scenario_fingerprint
        )
    finally:
        os.unlink(other_path)


def test_execute_run_scenario_fingerprint_normalizes_integral_floats():
    """global_event_rate: 1 and global_event_rate: 1.0 are behaviorally
    identical (validation only range-checks, it doesn't require a float
    literal); scenario_fingerprint must not depend on which one a config
    happens to spell out."""
    integral_config = {
        **CONFIG_DICT,
        "world": {
            **CONFIG_DICT["world"],
            "observation": {
                **CONFIG_DICT["world"]["observation"],
                "global_event_rate": 1,
            },
        },
    }
    float_config = {
        **CONFIG_DICT,
        "world": {
            **CONFIG_DICT["world"],
            "observation": {
                **CONFIG_DICT["world"]["observation"],
                "global_event_rate": 1.0,
            },
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(integral_config, f)
        integral_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(float_config, f)
        float_path = f.name

    try:
        integral_result = execute_run(RunRequest(config_path=integral_path, steps=1))
        float_result = execute_run(RunRequest(config_path=float_path, steps=1))

        assert (
            integral_result.metadata.scenario_fingerprint
            == float_result.metadata.scenario_fingerprint
        )
    finally:
        os.unlink(integral_path)
        os.unlink(float_path)


def test_execute_run_run_spec_fingerprint_distinguishes_large_seeds():
    """Seeds above 2**53 lose exact integer precision if ever converted to
    float; two distinct large seeds must still produce distinct
    run_spec_fingerprints (fingerprint normalization must not blanket-cast
    ints to float)."""
    seed_a = 9007199254740992
    seed_b = 9007199254740993
    assert float(seed_a) == float(seed_b)  # precondition: floats collide

    config_a = {**CONFIG_DICT, "world": {**CONFIG_DICT["world"], "rng_seed": seed_a}}
    config_b = {**CONFIG_DICT, "world": {**CONFIG_DICT["world"], "rng_seed": seed_b}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_a, f)
        path_a = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_b, f)
        path_b = f.name

    try:
        result_a = execute_run(RunRequest(config_path=path_a, steps=1))
        result_b = execute_run(RunRequest(config_path=path_b, steps=1))

        assert result_a.metadata.world_seed == seed_a
        assert result_b.metadata.world_seed == seed_b
        assert (
            result_a.metadata.run_spec_fingerprint
            != result_b.metadata.run_spec_fingerprint
        )
    finally:
        os.unlink(path_a)
        os.unlink(path_b)


def test_execute_run_scenario_fingerprint_same_for_explicit_and_omitted_defaults(
    config_path,
):
    """Two configs that build identical simulations -- one omitting
    agent.defaults.observation/trust/social/learning, one spelling out the
    exact built-in Agent defaults for them -- must fingerprint identically
    and must actually produce identical simulation results."""
    explicit_config = {
        **CONFIG_DICT,
        "agent": {
            **CONFIG_DICT["agent"],
            "defaults": {
                **CONFIG_DICT["agent"]["defaults"],
                "observation": {"attention": 1.0, "bias": 0.0},
                "trust": {"default": 0.5},
                "social": {
                    "confidence_bound": 1.0,
                    "trust_update_rate": 0.0,
                    "update_trust_on_rejection": True,
                },
                "learning": {
                    "rate": 0.1,
                    "observe_weight": 0.6,
                    "hear_weight": 0.3,
                    "verify_weight": 1.0,
                },
            },
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(explicit_config, f)
        explicit_path = f.name

    try:
        omitted_result = execute_run(RunRequest(config_path=config_path, steps=1))
        explicit_result = execute_run(RunRequest(config_path=explicit_path, steps=1))

        assert (
            omitted_result.metadata.scenario_fingerprint
            == explicit_result.metadata.scenario_fingerprint
        )
        assert (
            omitted_result.metadata.run_spec_fingerprint
            == explicit_result.metadata.run_spec_fingerprint
        )
        assert (
            omitted_result.metadata.resolved_config
            == explicit_result.metadata.resolved_config
        )
        # Same effective config -> not just the same fingerprint, but the
        # actual simulation results must be identical too. step_runtime_ms
        # is real wall-clock timing, not simulation state, so it's excluded.
        omitted_rows = [row.to_dict() for row in omitted_result.telemetry]
        explicit_rows = [row.to_dict() for row in explicit_result.telemetry]
        for omitted_row, explicit_row in zip(omitted_rows, explicit_rows):
            omitted_row.pop("step_runtime_ms")
            explicit_row.pop("step_runtime_ms")
        assert omitted_rows == explicit_rows
    finally:
        os.unlink(explicit_path)


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
