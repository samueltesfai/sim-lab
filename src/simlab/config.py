import copy
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


def _check_structure(cfg: dict) -> None:
    """Check ``cfg`` has the container shapes ``_materialize_agent_profiles``/
    ``_materialize_world_settings`` need to safely index into it -- nothing
    more. Full type/range/unknown-key checking happens afterward, against
    the *merged* result (see ``validate_config``), so this only needs to
    prevent a raw ``KeyError``/``TypeError`` during that merge.

    :param cfg: The loaded configuration
    :type cfg: dict
    :raises ValueError: if a required key is missing or has the wrong
        container type
    """
    if not isinstance(cfg, dict) or "world" not in cfg or "agent" not in cfg:
        raise ValueError("config must have 'world' and 'agent' top-level keys")

    world = cfg["world"]
    if not isinstance(world, dict):
        raise ValueError("world must be a mapping")
    for key in ("rng_seed", "truths", "observation"):
        if key not in world:
            raise ValueError(f"world.{key} is required")
    for key in ("truths", "observation"):
        if not isinstance(world[key], dict):
            raise ValueError(f"world.{key} must be a mapping")
    if not isinstance(world.get("noise", {}), dict):
        raise ValueError("world.noise must be a mapping")

    agent = cfg["agent"]
    if (
        not isinstance(agent, dict)
        or "defaults" not in agent
        or "profiles" not in agent
    ):
        raise ValueError("agent.defaults and agent.profiles are required")
    if not isinstance(agent["defaults"], dict):
        raise ValueError("agent.defaults must be a mapping")
    profiles = agent["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("agent.profiles must be a non-empty list")
    for profile in profiles:
        if (
            not isinstance(profile, dict)
            or "name" not in profile
            or "count" not in profile
        ):
            raise ValueError("each agent profile must define name and count")


def validate_config(cfg: dict) -> None:
    """Validate configuration structure, types, and ranges.

    First checks ``cfg`` has the container shapes needed to merge safely
    (``_check_structure``), then merges defaults in (``materialize_config``)
    and validates the *merged* result against ``config_schema.SimConfig``.
    Validating after merging, rather than before, means every value --
    whether it came from user YAML or from a built-in default -- is
    something explicitly present in the dict being validated, so a bad
    built-in default can't slip through unnoticed the way it could if
    validation ran on the raw, possibly-partial config.

    The validated model is discarded; everything downstream keeps consuming
    the original plain ``cfg`` dict (and re-merges it) unchanged. Re-running
    the merge is cheap (a handful of small dict operations); it's the
    redundant work worth accepting here, unlike re-running full validation
    for no correctness benefit.

    :param cfg: The loaded configuration
    :type cfg: dict
    :raises ValueError: if ``cfg`` doesn't match the expected schema
    """
    _check_structure(cfg)
    try:
        SimConfig.model_validate(materialize_config(cfg))
    except ValidationError as e:
        raise ValueError(str(e)) from e


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Deep-copies ``base`` first so that multiple merges sharing the same
    ``base`` (e.g. every profile merging against the same resolved
    ``agent.defaults``) never share a nested dict object -- mutating one
    profile's merged settings must never affect another's.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Settings sections translated as a whole dict (string action name ->
# ActionType), not field-by-field like the others.
_ACTION_MAP_FIELDS = {"action_preference", "action_cost"}

# The Agent constructor kwarg for a materialized settings path (section,
# field) defaults to "{section}_{field}" (e.g. social.confidence_bound ->
# social_confidence_bound). These are the fields where Agent's actual kwarg
# name doesn't follow that convention.
_KWARG_NAME_OVERRIDES: dict[tuple[str, str], str] = {
    ("trust", "default"): "default_trust",
    ("learning", "observe_weight"): "observe_weight",
    ("learning", "hear_weight"): "hear_weight",
    ("learning", "verify_weight"): "verify_weight",
}


def _kwarg_name(section: str, field: str) -> str:
    """The Agent constructor kwarg a materialized settings path
    (section, field) translates to.

    :param section: Top-level settings key (e.g. "social")
    :type section: str
    :param field: Field name within that section (e.g. "confidence_bound")
    :type field: str
    :return: The corresponding ``Agent.__init__`` keyword argument name
    :rtype: str
    """
    return _KWARG_NAME_OVERRIDES.get((section, field), f"{section}_{field}")


def _settings_to_agent_kwargs(settings: dict, profile_name: str) -> dict:
    """Translate a fully-materialized agent settings node into Agent
    constructor kwargs.

    Walks every settings section generically (deriving each field's kwarg
    name via ``_kwarg_name``) instead of reading from a fixed field list, so
    a newly added settings field is threaded through automatically -- only
    a field whose Agent kwarg name doesn't follow the "{section}_{field}"
    convention needs an entry in ``_KWARG_NAME_OVERRIDES``.
    ``action_preference``/``action_cost`` are handled separately since they
    translate as a whole dict, not field-by-field.

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
    for section, fields in settings.items():
        if section in _ACTION_MAP_FIELDS:
            continue
        for field, value in fields.items():
            kwargs[_kwarg_name(section, field)] = value
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


# World settings sections translated as a whole dict, not field-by-field:
# truths passes straight through, noise additionally needs its string keys
# converted to MemoryType.
_WORLD_WHOLE_DICT_FIELDS = {"truths", "noise"}


def _settings_to_world_kwargs(world_settings: dict) -> dict:
    """Translate a fully-materialized world settings dict into World
    constructor kwargs.

    Walks ``world_settings`` generically: a scalar leaf field (including a
    nested one, e.g. ``observation.private_event_rate``) uses its own field
    name as the World kwarg directly -- unlike agent settings, World's field
    names don't collide across sections, so no section prefix is needed.
    ``truths``/``noise`` pass through as whole dicts (``noise`` additionally
    converted to ``MemoryType``-keyed) rather than being decomposed, mirroring
    how ``action_preference``/``action_cost`` are handled for Agent.

    :param world_settings: The materialized world section (see
        ``_materialize_world_settings``); every field must already be present
    :type world_settings: dict
    :return: Keyword arguments ready to pass to ``World()`` (besides ``agents``)
    :rtype: dict
    """
    kwargs: dict = {
        "truths": world_settings["truths"],
        "noise": {MemoryType[k]: v for k, v in world_settings["noise"].items()},
    }
    for key, value in world_settings.items():
        if key in _WORLD_WHOLE_DICT_FIELDS:
            continue
        if isinstance(value, dict):
            kwargs.update(value)
        else:
            kwargs[key] = value
    return kwargs


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


def world_from_config(cfg: dict) -> World:
    """Build a World instance from a validated configuration."""
    world_settings = _materialize_world_settings(cfg)
    world_kwargs = _settings_to_world_kwargs(world_settings)

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

    return World(agents=agents, **world_kwargs)
