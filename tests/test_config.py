import inspect

import pytest
import tempfile
import os
import yaml

from simlab.agent import Agent
from simlab.config import (
    _materialize_world_settings,
    _settings_to_agent_kwargs,
    _settings_to_world_kwargs,
    load_config,
    validate_config,
    expand_agent_specs,
    world_from_config,
)
from simlab.config_schema import AgentSettings
from simlab.kernel_types import ActionType, MemoryType, Snapshot
from simlab.world import World


def _build_valid_world(config_dict: dict) -> World:
    """Build a World from a config dict after validating it."""
    validate_config(config_dict)
    return world_from_config(config_dict)


def create_test_config_file(config_dict: dict) -> str:
    """Create a temporary YAML config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        return f.name


def test_load_config_success():
    """Test successful config loading."""
    config_dict = {
        "world": {
            "rng_seed": 42,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True, 1: False},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
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

    config_path = create_test_config_file(config_dict)

    try:
        cfg = load_config(config_path)
        # load_config returns a plain dict
        assert isinstance(cfg, dict)

        assert cfg["agent"]["profiles"][0]["count"] == 5
        assert cfg["world"]["rng_seed"] == 42
        assert cfg["world"]["observation"]["private_event_rate"] == 0.1
        assert cfg["world"]["truths"] == {0: True, 1: False}
        assert cfg["agent"]["defaults"]["action_preference"]["IDLE"] == 0.0
        assert cfg["agent"]["defaults"]["action_preference"]["VERIFY"] == 0.9
    finally:
        os.unlink(config_path)


def test_load_config_file_not_found():
    """Test loading config with non-existent file."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("non_existent_config.yaml")


def test_validate_config_success():
    """Test config validation with valid config."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.2, "global_event_rate": 0.0},
            "truths": {0: True, 1: False},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
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
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    # Should not raise any exceptions
    validate_config(config_dict)


def test_validate_config_invalid_profile_count():
    """Test config validation with a non-positive profile count."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.2, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
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
            "profiles": [{"name": "default", "count": 0}],
        },
    }

    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.count"):
        validate_config(config_dict)


def test_validate_config_non_integral_profile_count():
    """A non-integral count is rejected rather than silently floored."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
        },
        "agent": {
            "defaults": {},
            "profiles": [{"name": "default", "count": 2.9}],  # Invalid: not an int
        },
    }

    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.count"):
        validate_config(config_dict)


def test_validate_config_invalid_observation_rate():
    """Test config validation with invalid observation event rate."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {
                "private_event_rate": 1.5,
                "global_event_rate": 0.0,
            },  # Invalid: must be in [0, 1]
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
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
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    with pytest.raises(ValueError, match=r"world\.observation\.private_event_rate"):
        validate_config(config_dict)


def test_validate_config_invalid_global_event_rate():
    """Test config validation with out-of-range global event rate."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {
                "private_event_rate": 0.1,
                "global_event_rate": 1.5,  # Invalid: must be in [0, 1]
            },
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
        },
        "agent": {
            "defaults": {},
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    with pytest.raises(ValueError, match=r"world\.observation\.global_event_rate"):
        validate_config(config_dict)


def test_validate_config_invalid_observation_attention():
    """Test config validation with out-of-range observation attention."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
        },
        "agent": {
            "defaults": {
                "observation": {"attention": 1.5},  # Invalid: must be in [0, 1]
            },
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.observation\.attention"):
        validate_config(config_dict)


def test_validate_config_invalid_observation_bias():
    """Test config validation with out-of-range observation bias on a profile."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
        },
        "agent": {
            "defaults": {},
            "profiles": [
                {
                    "name": "extreme",
                    "count": 3,
                    "observation": {"bias": -1.5},  # Invalid: must be in [-1, 1]
                }
            ],
        },
    }

    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.observation\.bias"):
        validate_config(config_dict)


def test_validate_config_negative_noise():
    """Test config validation with negative noise values."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.2, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {
                "OBSERVE": -0.1,
                "HEAR": 0.1,
                "VERIFY": 0.05,
            },  # Invalid: negative noise
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
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    with pytest.raises(ValueError, match="world.noise values must be non-negative"):
        validate_config(config_dict)


def test_validate_config_invalid_action_preference():
    """Test config validation with invalid action preference."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.2, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
        },
        "agent": {
            "defaults": {
                "action_preference": {
                    "IDLE": 0.0,
                    "VERIFY": 1.5,
                    "COMMUNICATE": 0.7,
                    "BROADCAST": 0.5,
                },  # Invalid: > 1
                "action_cost": {
                    "IDLE": 0.0,
                    "VERIFY": 0.35,
                    "COMMUNICATE": 0.15,
                    "BROADCAST": 0.30,
                },
            },
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    with pytest.raises(
        ValueError, match=r"action_preference values must be in \[0, 1\]"
    ):
        validate_config(config_dict)


def test_validate_config_invalid_action_name():
    """Test config validation with invalid action name."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.2, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
        },
        "agent": {
            "defaults": {
                "action_preference": {
                    "IDLE": 0.0,
                    "VERIFY": 0.9,
                    "INVALID_ACTION": 0.7,
                    "BROADCAST": 0.5,
                },  # Invalid action name
                "action_cost": {
                    "IDLE": 0.0,
                    "VERIFY": 0.35,
                    "COMMUNICATE": 0.15,
                    "BROADCAST": 0.30,
                },
            },
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    with pytest.raises(
        ValueError, match=r"agent\.profiles\.0\.action_preference\.INVALID_ACTION"
    ):
        validate_config(config_dict)


def test_validate_config_negative_action_cost():
    """Test config validation with negative action cost."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.2, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
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
                    "VERIFY": -0.1,
                    "COMMUNICATE": 0.15,
                    "BROADCAST": 0.30,
                },  # Invalid: negative cost
            },
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    with pytest.raises(ValueError, match="action_cost values must be non-negative"):
        validate_config(config_dict)


def test_validate_config_invalid_truths():
    """Test config validation with invalid truth values."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.2, "global_event_rate": 0.0},
            "truths": {0: "not_boolean", 1: False},  # Invalid: not boolean
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
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
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    with pytest.raises(ValueError, match=r"world\.truths\.0"):
        validate_config(config_dict)


def test_validate_config_rejects_unknown_learning_field():
    """A typo like learning.ratee (instead of learning.rate) must be
    rejected -- otherwise it silently has zero effect on the simulation
    while still polluting resolved_config/the scenario fingerprint."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        },
        "agent": {
            "defaults": {"learning": {"ratee": 0.5}},
            "profiles": [{"name": "default", "count": 1}],
        },
    }

    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.learning\.ratee"):
        validate_config(config_dict)


def test_validate_config_rejects_unknown_top_level_setting():
    """An unrecognized top-level settings key (e.g. a misspelled section
    name) is rejected rather than silently ignored."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        },
        "agent": {
            "defaults": {"observaton": {"attention": 0.5}},
            "profiles": [{"name": "default", "count": 1}],
        },
    }

    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.observaton"):
        validate_config(config_dict)


def test_validate_config_rejects_unknown_setting_on_profile():
    """Unknown settings keys are also rejected on profile overrides, not
    just agent.defaults."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        },
        "agent": {
            "defaults": {},
            "profiles": [
                {"name": "default", "count": 1, "social": {"confidence_boundd": 0.5}}
            ],
        },
    }

    with pytest.raises(
        ValueError, match=r"agent\.profiles\.0\.social\.confidence_boundd"
    ):
        validate_config(config_dict)


def test_build_world():
    """Test building a World instance from configuration."""
    config_dict = {
        "world": {
            "rng_seed": 42,
            "observation": {"private_event_rate": 0.2, "global_event_rate": 0.0},
            "truths": {0: True, 1: False},
            "noise": {"OBSERVE": 0.1, "HEAR": 0.05, "VERIFY": 0.02},
        },
        "agent": {
            "defaults": {
                "action_preference": {
                    "IDLE": 0.1,
                    "VERIFY": 0.8,
                    "COMMUNICATE": 0.6,
                    "BROADCAST": 0.4,
                },
                "action_cost": {
                    "IDLE": 0.05,
                    "VERIFY": 0.3,
                    "COMMUNICATE": 0.2,
                    "BROADCAST": 0.25,
                },
            },
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    world = _build_valid_world(config_dict)

    # Check world properties
    assert len(world.agents) == 3
    # World doesn't store rng_seed as attribute
    assert world.private_event_rate == 0.2
    assert world.truths == {0: True, 1: False}
    assert world.noise[MemoryType.OBSERVE] == 0.1
    assert world.noise[MemoryType.HEAR] == 0.05
    assert world.noise[MemoryType.VERIFY] == 0.02

    # Check agents
    for i, agent in enumerate(world.agents):
        assert agent.id == i
        # Agent doesn't store rng_seed as attribute
        # assert agent.rng_seed == 42 + i + 1

        # Check action preferences
        assert agent.action_preference[ActionType.IDLE] == 0.1
        assert agent.action_preference[ActionType.VERIFY] == 0.8
        assert agent.action_preference[ActionType.COMMUNICATE] == 0.6
        assert agent.action_preference[ActionType.BROADCAST] == 0.4

        # Check action costs
        assert agent.action_cost[ActionType.IDLE] == 0.05
        assert agent.action_cost[ActionType.VERIFY] == 0.3
        assert agent.action_cost[ActionType.COMMUNICATE] == 0.2
        assert agent.action_cost[ActionType.BROADCAST] == 0.25


def test_build_world_partial_config():
    """Test building a World with partial configuration (defaults should be used)."""
    config_dict = {
        "world": {
            "rng_seed": 10,
            "observation": {"private_event_rate": 0.15, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.05},  # Missing HEAR and VERIFY
        },
        "agent": {
            "defaults": {
                "action_preference": {"VERIFY": 0.7},  # Missing other actions
                "action_cost": {"VERIFY": 0.4},  # Missing other actions
            },
            "profiles": [{"name": "default", "count": 2}],
        },
    }

    world = _build_valid_world(config_dict)

    # Defaults should be applied
    assert len(world.agents) == 2

    agent = world.agents[0]
    # Should have default values for missing action preferences
    assert agent.action_preference[ActionType.IDLE] == 0.0  # Default
    assert agent.action_preference[ActionType.VERIFY] == 0.7  # Custom
    assert agent.action_preference[ActionType.COMMUNICATE] == 0.7  # Default
    assert agent.action_preference[ActionType.BROADCAST] == 0.5  # Default

    # Should have default values for missing action costs
    assert agent.action_cost[ActionType.IDLE] == 0.0  # Default
    assert agent.action_cost[ActionType.VERIFY] == 0.4  # Custom
    assert agent.action_cost[ActionType.COMMUNICATE] == 0.15  # Default
    assert agent.action_cost[ActionType.BROADCAST] == 0.30  # Default

    # Should have default noise values
    assert world.noise[MemoryType.OBSERVE] == 0.05  # Custom
    assert world.noise[MemoryType.HEAR] == 0.0  # Default
    assert world.noise[MemoryType.VERIFY] == 0.0  # Default


def test_build_world_integration():
    """Test that built world works correctly (integration test)."""
    config_dict = {
        "world": {
            "rng_seed": 123,
            "observation": {
                "private_event_rate": 0.0,
                "global_event_rate": 0.0,
            },  # No random observations for deterministic test
            "truths": {0: True, 1: False},
            "noise": {
                "OBSERVE": 0.0,
                "HEAR": 0.0,
                "VERIFY": 0.0,
            },  # No noise for deterministic test
        },
        "agent": {
            "defaults": {
                "action_preference": {
                    "IDLE": 1.0,
                    "VERIFY": 0.0,
                    "COMMUNICATE": 0.0,
                    "BROADCAST": 0.0,
                },
                "action_cost": {
                    "IDLE": 0.0,
                    "VERIFY": 0.0,
                    "COMMUNICATE": 0.0,
                    "BROADCAST": 0.0,
                },
            },
            "profiles": [{"name": "default", "count": 3}],
        },
    }

    world = _build_valid_world(config_dict)

    # Test that world can run steps
    initial_tick = world.tick
    snapshot = world.step()

    assert world.tick == initial_tick + 1
    assert isinstance(snapshot, Snapshot)

    # Test that agents have expected behavior
    agent = world.agents[0]
    # With IDLE preference of 1.0 and 0 cost, should always choose IDLE
    action = agent.choose_action(world)
    assert action.type == ActionType.IDLE


def test_load_config_and_build_world_integration():
    """Test full integration: load config from file and build world."""
    config_dict = {
        "world": {
            "rng_seed": 999,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        },
        "agent": {
            "defaults": {
                "action_preference": {
                    "IDLE": 0.0,
                    "VERIFY": 1.0,
                    "COMMUNICATE": 0.0,
                    "BROADCAST": 0.0,
                },
                "action_cost": {
                    "IDLE": 0.0,
                    "VERIFY": 0.1,
                    "COMMUNICATE": 0.1,
                    "BROADCAST": 0.1,
                },
            },
            "profiles": [{"name": "default", "count": 2}],
        },
    }

    config_path = create_test_config_file(config_dict)

    try:
        # Load and build
        cfg = load_config(config_path)
        world = world_from_config(cfg)

        # Verify it works
        assert len(world.agents) == 2
        assert world.truths == {0: True}

        # Run a step to make sure everything is wired correctly
        snapshot = world.step()
        assert snapshot.tick == 0

    finally:
        os.unlink(config_path)


def _config(profiles: list[dict]) -> dict:
    """Build a canonical config with the given profiles and minimal defaults."""
    return {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
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
            "profiles": profiles,
        },
    }


def test_validate_config_rejects_non_integer_rng_seed():
    """A non-integral seed (e.g. 1.9) would silently seed the world's RNG
    with a different stream than the int world_seed later recorded in run
    metadata -- reject it instead of letting it through unnoticed."""
    cfg = _config([{"name": "default", "count": 1}])
    cfg["world"]["rng_seed"] = 1.9

    with pytest.raises(ValueError, match=r"world\.rng_seed"):
        validate_config(cfg)


def test_validate_config_rejects_bool_rng_seed():
    """bool is an int subclass; True/False are not meaningful seeds."""
    cfg = _config([{"name": "default", "count": 1}])
    cfg["world"]["rng_seed"] = True

    with pytest.raises(ValueError, match=r"world\.rng_seed"):
        validate_config(cfg)


def test_validate_config_rejects_missing_rng_seed():
    cfg = _config([{"name": "default", "count": 1}])
    del cfg["world"]["rng_seed"]

    with pytest.raises(ValueError, match=r"world\.rng_seed"):
        validate_config(cfg)


def test_validate_config_rejects_duplicate_profile_names():
    """Two profiles sharing a name would silently overwrite each other's
    entry in profile_counts/scenario features (e.g. profile_count.<name>),
    making the reported population inconsistent with num_agents."""
    cfg = _config(
        [
            {"name": "dup", "count": 2},
            {"name": "dup", "count": 3},
        ]
    )

    with pytest.raises(ValueError, match="duplicate agent profile name: 'dup'"):
        validate_config(cfg)


def test_validate_config_rejects_mixed_type_truth_keys():
    """A claim id given as a string (e.g. from a quoted YAML key) alongside
    genuine int claim ids would otherwise pass validation and crash
    execute_run later inside json.dumps(sort_keys=True), which can't order
    mixed int/str dict keys."""
    cfg = _config([{"name": "default", "count": 1}])
    cfg["world"]["truths"] = {0: True, "1": False}

    with pytest.raises(ValueError, match=r"world\.truths"):
        validate_config(cfg)


def test_validate_config_rejects_non_string_profile_name():
    """A non-string profile name would let e.g. profile 1 (int) and profile
    "1" (str) both pass duplicate-name detection (1 != "1" in Python) while
    colliding once flattened into scenario feature keys like
    profile_count.1, and can independently crash
    json.dumps(profile_counts, sort_keys=True) on mixed key types."""
    cfg = _config([{"name": 1, "count": 1}])

    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.name"):
        validate_config(cfg)


def test_validate_config_rejects_bool_for_world_rate():
    """bool is an int subclass, so global_event_rate: true would otherwise
    pass the 0 <= x <= 1 range check -- but fingerprint normalization
    deliberately keeps bools distinct from numbers (needed so world.truths
    values stay true/false rather than becoming 1/0), so a boolean rate and
    its numeric equivalent (1.0) would fingerprint differently despite
    building an identical simulation."""
    cfg = _config([{"name": "default", "count": 1}])
    cfg["world"]["observation"]["global_event_rate"] = True

    with pytest.raises(ValueError, match=r"world\.observation\.global_event_rate"):
        validate_config(cfg)


def test_validate_config_rejects_bool_for_agent_attention():
    """Same bool-as-int gap, on an agent settings field."""
    cfg = _config([{"name": "default", "count": 1}])
    cfg["agent"]["defaults"]["observation"] = {"attention": True}

    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.observation\.attention"):
        validate_config(cfg)


def test_single_default_profile_builds():
    """A single 'default' profile builds the requested number of agents."""
    cfg = _config([{"name": "default", "count": 4}])
    validate_config(cfg)
    world = world_from_config(cfg)

    assert len(world.agents) == 4
    assert all(agent.profile_name == "default" for agent in world.agents)
    assert world.profile_counts == {"default": 4}

    # Cognition params should fall back to Agent defaults.
    agent = world.agents[0]
    assert agent.observation_attention == 1.0
    assert agent.learning_rate == 0.1


def test_profiles_expand_counts_and_params():
    """Structured config expands profiles into the right counts and overrides."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.1, "HEAR": 0.15, "VERIFY": 0.05},
        },
        "agent": {
            "defaults": {
                "observation": {"attention": 1.0, "bias": 0.0},
                "trust": {"default": 0.5},
                "learning": {"rate": 0.1, "hear_weight": 0.3},
            },
            "profiles": [
                {"name": "attentive", "count": 20, "observation": {"attention": 0.95}},
                {"name": "distracted", "count": 20, "observation": {"attention": 0.35}},
                {
                    "name": "skeptical",
                    "count": 10,
                    "trust": {"default": 0.25},
                    "learning": {"hear_weight": 0.15},
                },
            ],
        },
    }

    world = _build_valid_world(config_dict)

    assert len(world.agents) == 50
    assert world.profile_counts == {
        "attentive": 20,
        "distracted": 20,
        "skeptical": 10,
    }

    by_profile = {a.profile_name: a for a in world.agents}
    assert by_profile["attentive"].observation_attention == 0.95
    assert by_profile["distracted"].observation_attention == 0.35
    # Skeptical inherits default attention but overrides trust + hear_weight.
    assert by_profile["skeptical"].observation_attention == 1.0
    assert by_profile["skeptical"].default_trust == 0.25
    assert by_profile["skeptical"].hear_weight == 0.15
    # Non-overridden default propagates.
    assert by_profile["attentive"].default_trust == 0.5


def test_profile_counts_determine_total_agents():
    """Total agents is the sum of profile counts; no separate world total."""
    cfg = _config([{"name": "a", "count": 20}, {"name": "b", "count": 29}])
    validate_config(cfg)
    world = world_from_config(cfg)
    assert len(world.agents) == 49
    assert world.profile_counts == {"a": 20, "b": 29}


def test_empty_profiles_raise():
    """An empty profiles list is rejected."""
    config_dict = _config([])
    with pytest.raises(ValueError, match=r"agent\.profiles"):
        validate_config(config_dict)


def test_missing_defaults_raises():
    """agent.defaults is required."""
    config_dict = _config([{"name": "default", "count": 3}])
    del config_dict["agent"]["defaults"]

    with pytest.raises(ValueError, match=r"agent\.defaults"):
        validate_config(config_dict)


def test_missing_profiles_raises():
    """agent.profiles is required."""
    config_dict = _config([{"name": "default", "count": 3}])
    del config_dict["agent"]["profiles"]

    with pytest.raises(ValueError, match=r"agent\.profiles"):
        validate_config(config_dict)


def test_profile_missing_count_raises():
    """Each profile must define a count."""
    config_dict = _config([{"name": "default"}])
    with pytest.raises(
        ValueError, match="each agent profile must define name and count"
    ):
        validate_config(config_dict)


def test_expand_agent_specs_single_profile():
    """expand_agent_specs returns one spec per agent for a single default profile."""
    cfg = _config([{"name": "default", "count": 3}])
    specs = expand_agent_specs(cfg)

    assert len(specs) == 3
    assert all(spec["profile_name"] == "default" for spec in specs)
    assert all(ActionType.VERIFY in spec["action_preference"] for spec in specs)


def test_settings_to_agent_kwargs_completeness():
    """_settings_to_agent_kwargs walks a materialized settings dict
    generically (deriving each field's Agent kwarg name via a naming
    convention + a small exceptions table), so it can't silently drop a
    field the way a fixed table could. This checks the other direction: its
    output must be exactly the settings-derived kwargs Agent.__init__
    accepts -- no missing kwarg (a field it doesn't know how to name) and
    no extra one (a wrong guess, which would also fail loudly at
    Agent(**kwargs) time)."""
    settings = AgentSettings().model_dump()
    kwargs = _settings_to_agent_kwargs(settings, "test-profile")

    expected = set(inspect.signature(Agent.__init__).parameters) - {
        "self",
        "id",
        "rng_seed",
    }
    assert set(kwargs) == expected


def test_settings_to_world_kwargs_completeness():
    """_settings_to_world_kwargs must produce exactly the settings-derived
    kwargs World.__init__ accepts -- mirrors
    test_settings_to_agent_kwargs_completeness for World's construction
    path."""
    world_settings = _materialize_world_settings(
        {
            "world": {
                "rng_seed": 0,
                "truths": {0: True},
                "noise": {},
                "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            }
        }
    )
    kwargs = _settings_to_world_kwargs(world_settings)

    expected = set(inspect.signature(World.__init__).parameters) - {"self", "agents"}
    assert set(kwargs) == expected


def test_validate_config_rejects_non_integral_count():
    """A fractional profile count is rejected instead of being floored.

    ``load_config`` performs the validation at the input boundary, so the
    build path cannot silently change the requested population size.
    """
    cfg = _config([{"name": "default", "count": 2.9}])
    with pytest.raises(ValueError, match=r"agent\.profiles\.0\.count"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Social param validation
# ---------------------------------------------------------------------------


def _social_config(social: dict) -> dict:
    """Build a valid config dict with the given social overrides in defaults."""
    base = _config([{"name": "default", "count": 2}])
    base["agent"]["defaults"]["social"] = social
    return base


def test_validate_social_confidence_bound_valid():
    """Valid confidence_bound values in [0, 1] pass validation."""
    for val in [0.0, 0.5, 1.0]:
        cfg = _social_config({"confidence_bound": val})
        validate_config(cfg)  # should not raise


def test_validate_social_confidence_bound_invalid():
    """confidence_bound outside [0, 1] is rejected."""
    for val in [-0.1, 1.1]:
        cfg = _social_config({"confidence_bound": val})
        with pytest.raises(
            ValueError, match=r"agent\.profiles\.0\.social\.confidence_bound"
        ):
            validate_config(cfg)


def test_validate_social_trust_update_rate_valid():
    """Valid trust_update_rate values in [0, 1] pass validation."""
    for val in [0.0, 0.3, 1.0]:
        cfg = _social_config({"trust_update_rate": val})
        validate_config(cfg)


def test_validate_social_trust_update_rate_invalid():
    """trust_update_rate outside [0, 1] is rejected."""
    for val in [-0.01, 1.5]:
        cfg = _social_config({"trust_update_rate": val})
        with pytest.raises(
            ValueError, match=r"agent\.profiles\.0\.social\.trust_update_rate"
        ):
            validate_config(cfg)


def test_validate_social_update_trust_on_rejection_valid():
    """Boolean update_trust_on_rejection passes validation."""
    for val in [True, False]:
        cfg = _social_config({"update_trust_on_rejection": val})
        validate_config(cfg)


def test_validate_social_update_trust_on_rejection_invalid():
    """Non-boolean update_trust_on_rejection is rejected."""
    cfg = _social_config({"update_trust_on_rejection": "yes"})
    with pytest.raises(
        ValueError, match=r"agent\.profiles\.0\.social\.update_trust_on_rejection"
    ):
        validate_config(cfg)


def test_social_params_propagate_to_agents():
    """Social params set in defaults propagate through build_world to Agent attrs."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.0, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        },
        "agent": {
            "defaults": {
                "social": {
                    "confidence_bound": 0.4,
                    "trust_update_rate": 0.2,
                    "update_trust_on_rejection": False,
                },
            },
            "profiles": [{"name": "default", "count": 3}],
        },
    }
    world = _build_valid_world(config_dict)

    for agent in world.agents:
        assert agent.social_confidence_bound == pytest.approx(0.4)
        assert agent.social_trust_update_rate == pytest.approx(0.2)
        assert agent.social_update_trust_on_rejection is False


def test_social_params_profile_overrides_defaults():
    """Profile-level social overrides are merged on top of defaults."""
    config_dict = {
        "world": {
            "rng_seed": 0,
            "observation": {"private_event_rate": 0.0, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        },
        "agent": {
            "defaults": {
                "social": {
                    "confidence_bound": 1.0,
                    "trust_update_rate": 0.0,
                    "update_trust_on_rejection": True,
                },
            },
            "profiles": [
                {"name": "open", "count": 2},
                {
                    "name": "closed",
                    "count": 2,
                    "social": {
                        "confidence_bound": 0.3,
                        "trust_update_rate": 0.5,
                        "update_trust_on_rejection": False,
                    },
                },
            ],
        },
    }
    world = _build_valid_world(config_dict)

    open_agents = [a for a in world.agents if a.profile_name == "open"]
    closed_agents = [a for a in world.agents if a.profile_name == "closed"]

    for agent in open_agents:
        assert agent.social_confidence_bound == pytest.approx(1.0)
        assert agent.social_trust_update_rate == pytest.approx(0.0)
        assert agent.social_update_trust_on_rejection is True

    for agent in closed_agents:
        assert agent.social_confidence_bound == pytest.approx(0.3)
        assert agent.social_trust_update_rate == pytest.approx(0.5)
        assert agent.social_update_trust_on_rejection is False


def test_social_params_absent_uses_agent_defaults():
    """When social section is omitted, Agent defaults (1.0 / 0.0 / True) apply."""
    cfg = _config([{"name": "default", "count": 2}])
    validate_config(cfg)
    world = world_from_config(cfg)

    for agent in world.agents:
        assert agent.social_confidence_bound == pytest.approx(1.0)
        assert agent.social_trust_update_rate == pytest.approx(0.0)
        assert agent.social_update_trust_on_rejection is True
