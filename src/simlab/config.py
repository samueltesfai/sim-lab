import os

import yaml
from pydantic import ValidationError

from simlab.agent import DEFAULT_SETTINGS, Agent
from simlab.config_schema import SimConfig
from simlab.world import DEFAULT_NOISE, World
from simlab.kernel_types import ActionType, MemoryType


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Validate configuration structure, types, and ranges.

    Delegates to ``config_schema.SimConfig``. The validated model is
    discarded -- everything downstream keeps consuming the original plain
    ``cfg`` dict unchanged.

    :param cfg: The loaded configuration
    :type cfg: dict
    :raises ValueError: if ``cfg`` doesn't match the expected schema
    """
    try:
        SimConfig.model_validate(cfg)
    except ValidationError as e:
        raise ValueError(str(e)) from e


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (one level deep dicts)."""
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Maps each flat Agent constructor kwarg to the nested (section, field) path
# it reads from a materialized settings dict. Declarative on purpose: a test
# (test_settings_to_agent_kwargs_completeness in tests/test_config.py) walks
# config_schema.AgentSettings and asserts every scalar leaf field appears
# here as a value -- otherwise a newly added settings field would validate,
# materialize, and hash into fingerprints fine, but silently never reach
# Agent, since nothing else would catch a missing translation.
_SCALAR_KWARG_PATHS: dict[str, tuple[str, str]] = {
    "observation_attention": ("observation", "attention"),
    "observation_bias": ("observation", "bias"),
    "default_trust": ("trust", "default"),
    "learning_rate": ("learning", "rate"),
    "observe_weight": ("learning", "observe_weight"),
    "hear_weight": ("learning", "hear_weight"),
    "verify_weight": ("learning", "verify_weight"),
    "social_confidence_bound": ("social", "confidence_bound"),
    "social_trust_update_rate": ("social", "trust_update_rate"),
    "social_update_trust_on_rejection": ("social", "update_trust_on_rejection"),
}


def _settings_to_agent_kwargs(settings: dict, profile_name: str) -> dict:
    """Translate a fully-materialized agent settings node into Agent
    constructor kwargs.

    ``settings`` must already have every field present (see
    ``_materialize_agent_profiles``); nothing here is optional.

    :param settings: Fully-merged agent settings for one profile
    :type settings: dict
    :param profile_name: The profile's name, passed through as a kwarg
    :type profile_name: str
    :return: Keyword arguments ready to pass to ``Agent()``
    :rtype: dict
    """
    kwargs: dict = {
        "profile_name": profile_name,
        "action_preference": {
            ActionType[k]: v for k, v in settings["action_preference"].items()
        },
        "action_cost": {ActionType[k]: v for k, v in settings["action_cost"].items()},
    }
    for kwarg_name, (section, field) in _SCALAR_KWARG_PATHS.items():
        kwargs[kwarg_name] = settings[section][field]
    return kwargs


def _materialize_agent_profiles(cfg: dict) -> list[dict]:
    """Expand ``agent.defaults`` + ``agent.profiles`` into one fully-specified,
    JSON-safe settings dict per profile.

    Every Agent-recognized field is present regardless of what the YAML
    omitted -- omitted fields are filled from ``agent.DEFAULT_SETTINGS``. This
    is a pure transformation; callers must ensure ``cfg`` has already passed
    ``validate_config``.

    :param cfg: The loaded, validated configuration
    :type cfg: dict
    :return: One ``{"name":, "count":, **settings}`` dict per profile
    :rtype: list[dict]
    """
    agent_cfg = cfg["agent"]
    base = _deep_merge(DEFAULT_SETTINGS, agent_cfg["defaults"])

    profiles: list[dict] = []
    for profile in agent_cfg["profiles"]:
        name = profile["name"]
        count = profile["count"]
        overrides = {k: v for k, v in profile.items() if k not in {"name", "count"}}
        merged = _deep_merge(base, overrides)
        profiles.append({"name": name, "count": count, **merged})

    return profiles


def expand_agent_specs(cfg: dict) -> list[dict]:
    """Expand ``agent.defaults`` + ``agent.profiles`` into one Agent spec per agent.

    Each profile inherits ``agent.defaults`` and may override any subset of
    settings. The total number of agents is the sum of the profile counts.
    This is a pure transformation; callers must ensure ``cfg`` has already
    passed ``validate_config``.

    :param cfg: The loaded, validated configuration
    :type cfg: dict
    :return: One Agent constructor kwargs dict per agent
    :rtype: list[dict]
    """
    specs: list[dict] = []
    for profile in _materialize_agent_profiles(cfg):
        name = profile["name"]
        count = profile["count"]
        settings = {k: v for k, v in profile.items() if k not in {"name", "count"}}
        kwargs = _settings_to_agent_kwargs(settings, name)
        specs.extend(dict(kwargs) for _ in range(count))

    return specs


def _materialize_world_settings(cfg: dict) -> dict:
    """Return the ``world`` section with every field explicit, including noise
    keys the YAML omitted (each defaults to 0.0 -- see ``world.DEFAULT_NOISE``).

    This is a pure transformation; callers must ensure ``cfg`` has already
    passed ``validate_config``.

    :param cfg: The loaded, validated configuration
    :type cfg: dict
    :return: The fully-materialized ``world`` section
    :rtype: dict
    """
    world_cfg = cfg["world"]
    return {
        "rng_seed": world_cfg["rng_seed"],
        "truths": dict(world_cfg["truths"]),
        "noise": {**DEFAULT_NOISE, **world_cfg["noise"]},
        "observation": {
            "private_event_rate": world_cfg["observation"]["private_event_rate"],
            "global_event_rate": world_cfg["observation"]["global_event_rate"],
        },
    }


def materialize_config(cfg: dict) -> dict:
    """Return the fully effective configuration -- every field explicit, no
    field silently defaulted downstream by Agent/World construction.

    Suitable for hashing or storing as a reproducibility record: two configs
    that build identical simulations always materialize to the same result,
    regardless of which defaulted fields either one happened to spell out.

    :param cfg: The loaded, validated configuration
    :type cfg: dict
    :return: The fully effective configuration, keyed like the source YAML
    :rtype: dict
    """
    return {
        "world": _materialize_world_settings(cfg),
        "agent": {"profiles": _materialize_agent_profiles(cfg)},
    }


def materialize_scenario(cfg: dict) -> dict:
    """Return the behaviorally meaningful configuration -- the fully
    effective config minus ``world.rng_seed``.

    Two runs with different seeds are stochastic replicates of the same
    scenario, not different scenarios, so the seed is deliberately excluded
    here: this is what a scenario fingerprint should be hashed from, as
    opposed to ``materialize_config`` (which keeps the seed, for humans
    inspecting a single run's resolved config).

    :param cfg: The loaded, validated configuration
    :type cfg: dict
    :return: The effective configuration with ``world.rng_seed`` removed
    :rtype: dict
    """
    full = materialize_config(cfg)
    world = {k: v for k, v in full["world"].items() if k != "rng_seed"}
    return {"world": world, "agent": full["agent"]}


def build_world(cfg: dict) -> World:
    """Build a World instance from a validated configuration."""
    world_settings = _materialize_world_settings(cfg)

    # World noise -> enum-keyed dict.
    noise = {MemoryType[k]: v for k, v in world_settings["noise"].items()}

    # Expand agent.defaults + agent.profiles into concrete agents.
    specs = expand_agent_specs(cfg)

    agents = []
    for i, spec in enumerate(specs):
        agent = Agent(
            id=i,
            rng_seed=world_settings["rng_seed"]
            + i
            + 1,  # add i to differ seed, and 1 to offset from world rng
            **spec,
        )
        agents.append(agent)

    # Create world
    world = World(
        agents=agents,
        truths=world_settings["truths"],
        rng_seed=world_settings["rng_seed"],
        noise=noise,
        private_event_rate=world_settings["observation"]["private_event_rate"],
        global_event_rate=world_settings["observation"]["global_event_rate"],
    )

    return world
