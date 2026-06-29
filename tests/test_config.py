import pytest
import tempfile
import os
from omegaconf import OmegaConf, DictConfig

from simlab.config import (
    load_config,
    validate_config,
    convert_noise_strings,
    expand_agent_specs,
    build_world,
)
from simlab.sim import ActionType, MemoryType, Snapshot


def create_test_config_file(config_dict: dict) -> str:
    """Create a temporary YAML config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(config_dict, f.name)
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
        # load_config returns a DictConfig
        assert isinstance(cfg, DictConfig)

        # Test accessing the config as OmegaConf/DictConfig
        assert cfg.agent.profiles[0].count == 5
        assert cfg.world.rng_seed == 42
        assert cfg.world.observation.private_event_rate == 0.1
        assert cfg.world.truths == {0: True, 1: False}
        assert cfg.agent.defaults.action_preference.IDLE == 0.0
        assert cfg.agent.defaults.action_preference.VERIFY == 0.9
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

    cfg = OmegaConf.create(config_dict)

    # Should not raise any exceptions
    validate_config(cfg)


def test_validate_config_invalid_profile_count():
    """Test config validation with a non-positive profile count."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(
        ValueError, match="agent profile default count must be a positive integer"
    ):
        validate_config(cfg)


def test_validate_config_non_integral_profile_count():
    """A non-integral count is rejected rather than silently floored."""
    config_dict = {
        "world": {
            "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05},
        },
        "agent": {
            "defaults": {},
            "profiles": [{"name": "default", "count": 2.9}],  # Invalid: not an int
        },
    }

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(
        ValueError, match="agent profile default count must be a positive integer"
    ):
        validate_config(cfg)


def test_validate_config_invalid_observation_rate():
    """Test config validation with invalid observation event rate."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(
        ValueError,
        match="world.observation.private_event_rate must be in \\[0, 1\\]",
    ):
        validate_config(cfg)


def test_validate_config_invalid_global_event_rate():
    """Test config validation with out-of-range global event rate."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(
        ValueError,
        match="world.observation.global_event_rate must be in \\[0, 1\\]",
    ):
        validate_config(cfg)


def test_validate_config_invalid_observation_attention():
    """Test config validation with out-of-range observation attention."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(
        ValueError,
        match="agent.defaults.observation.attention must be in \\[0, 1\\]",
    ):
        validate_config(cfg)


def test_validate_config_invalid_observation_bias():
    """Test config validation with out-of-range observation bias on a profile."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(
        ValueError,
        match="agent.profiles.extreme.observation.bias must be in \\[-1, 1\\]",
    ):
        validate_config(cfg)


def test_validate_config_negative_noise():
    """Test config validation with negative noise values."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(ValueError, match="world.noise.OBSERVE must be non-negative"):
        validate_config(cfg)


def test_validate_config_invalid_action_preference():
    """Test config validation with invalid action preference."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(
        ValueError,
        match="agent.defaults.action_preference.VERIFY must be in \\[0, 1\\]",
    ):
        validate_config(cfg)


def test_validate_config_invalid_action_name():
    """Test config validation with invalid action name."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(ValueError, match="Invalid action: INVALID_ACTION"):
        validate_config(cfg)


def test_validate_config_negative_action_cost():
    """Test config validation with negative action cost."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(
        ValueError, match="agent.defaults.action_cost.VERIFY must be non-negative"
    ):
        validate_config(cfg)


def test_validate_config_invalid_truths():
    """Test config validation with invalid truth values."""
    config_dict = {
        "world": {
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

    cfg = OmegaConf.create(config_dict)

    with pytest.raises(ValueError, match="world.truths.0 must be boolean"):
        validate_config(cfg)


def test_convert_noise_strings():
    """Test conversion of string noise keys to MemoryType enums."""
    config_dict = {"world": {"noise": {"OBSERVE": 0.0, "HEAR": 0.1, "VERIFY": 0.05}}}

    cfg = OmegaConf.create(config_dict)

    converted_cfg = convert_noise_strings(cfg)

    # Noise keys should be converted to MemoryType enums
    assert MemoryType.OBSERVE in converted_cfg.world.noise
    assert MemoryType.HEAR in converted_cfg.world.noise
    assert MemoryType.VERIFY in converted_cfg.world.noise

    # Check values are preserved
    assert converted_cfg.world.noise[MemoryType.OBSERVE] == 0.0
    assert converted_cfg.world.noise[MemoryType.HEAR] == 0.1
    assert converted_cfg.world.noise[MemoryType.VERIFY] == 0.05


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

    cfg = OmegaConf.create(config_dict)
    world = build_world(cfg)

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

    cfg = OmegaConf.create(config_dict)
    world = build_world(cfg)

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

    cfg = OmegaConf.create(config_dict)
    world = build_world(cfg)

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
        world = build_world(cfg)

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


def test_single_default_profile_builds():
    """A single 'default' profile builds the requested number of agents."""
    cfg = OmegaConf.create(_config([{"name": "default", "count": 4}]))
    world = build_world(cfg)

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

    cfg = OmegaConf.create(config_dict)
    world = build_world(cfg)

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
    cfg = OmegaConf.create(
        _config([{"name": "a", "count": 20}, {"name": "b", "count": 29}])
    )
    world = build_world(cfg)
    assert len(world.agents) == 49
    assert world.profile_counts == {"a": 20, "b": 29}


def test_empty_profiles_raise():
    """An empty profiles list is rejected."""
    config_dict = _config([])
    cfg = OmegaConf.create(config_dict)
    with pytest.raises(ValueError, match="at least one profile"):
        validate_config(cfg)


def test_missing_defaults_raises():
    """agent.defaults is required."""
    config_dict = _config([{"name": "default", "count": 3}])
    del config_dict["agent"]["defaults"]

    cfg = OmegaConf.create(config_dict)
    with pytest.raises(ValueError, match="agent.defaults is required"):
        validate_config(cfg)


def test_missing_profiles_raises():
    """agent.profiles is required."""
    config_dict = _config([{"name": "default", "count": 3}])
    del config_dict["agent"]["profiles"]

    cfg = OmegaConf.create(config_dict)
    with pytest.raises(ValueError, match="agent.profiles is required"):
        validate_config(cfg)


def test_profile_missing_count_raises():
    """Each profile must define a count."""
    config_dict = _config([{"name": "default"}])
    cfg = OmegaConf.create(config_dict)
    with pytest.raises(
        ValueError, match="agent profile default count must be a positive integer"
    ):
        validate_config(cfg)


def test_expand_agent_specs_single_profile():
    """expand_agent_specs returns one spec per agent for a single default profile."""
    cfg = OmegaConf.create(_config([{"name": "default", "count": 3}]))
    specs = expand_agent_specs(cfg)

    assert len(specs) == 3
    assert all(spec["profile_name"] == "default" for spec in specs)
    assert all(ActionType.VERIFY in spec["action_preference"] for spec in specs)
