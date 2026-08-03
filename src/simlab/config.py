import os

import yaml
from pydantic import ValidationError

from simlab._merge import deep_merge
from simlab.agent import Agent
from simlab.config_schema import (
    AgentProfile,
    AgentSettings,
    SimConfig,
    WorldObservation,
    WorldSection,
)
from simlab.world import World


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that rejects a mapping with a repeated key --
    plain ``yaml.safe_load`` silently keeps only the last value."""

    def construct_mapping(self, node, deep=False):
        seen: set[object] = set()
        for key_node, _ in node.value:
            # merge key isn't constructible yet; overriding a merged-in
            # default is expected, not a duplicate
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key: {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


def load_config(path: str) -> SimConfig:
    """Load, merge, and validate configuration from a YAML file.

    :param path: Path to the YAML config file
    :type path: str
    :return: The resolved, validated config
    :rtype: SimConfig
    :raises FileNotFoundError: if ``path`` doesn't exist
    :raises ValueError: if the config doesn't match the expected schema
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_UniqueKeyLoader)
    return validate_config(cfg)


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
    for key in ("rng_seed", "truths"):
        if key not in world:
            raise ValueError(f"world.{key} is required")
    if not isinstance(world["truths"], dict):
        raise ValueError("world.truths must be a mapping")
    if not isinstance(world.get("noise", {}), dict):
        raise ValueError("world.noise must be a mapping")
    if not isinstance(world.get("observation", {}), dict):
        raise ValueError("world.observation must be a mapping")

    agent = cfg["agent"]
    if (
        not isinstance(agent, dict)
        or "defaults" not in agent
        or "profiles" not in agent
    ):
        raise ValueError("agent.defaults and agent.profiles are required")
    if not isinstance(agent["defaults"], dict):
        raise ValueError("agent.defaults must be a mapping")
    if "name" in agent["defaults"] or "count" in agent["defaults"]:
        # Per-profile fields; letting these into defaults would silently
        # overwrite every profile's own explicit name/count on merge.
        raise ValueError(
            "agent.defaults must not contain 'name' or 'count' -- those are "
            "per-profile fields, not shared settings"
        )
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


def validate_config(cfg: dict) -> SimConfig:
    """Merge defaults into ``cfg`` and validate the result.

    First checks ``cfg`` has the container shapes needed to merge safely
    (``_check_structure``), then merges defaults in (``_materialize_config``)
    and validates the *merged* result against ``config_schema.SimConfig``.
    Validating after merging, rather than before, means every value --
    whether it came from user YAML or from a built-in default -- is
    something explicitly present in the dict being validated, so a bad
    built-in default can't slip through unnoticed the way it could if
    validation ran on the raw, possibly-partial config.

    The returned, fully-resolved ``SimConfig`` is what the rest of the
    pipeline (``world_from_config``, ``expand_agent_specs``, callers'
    fingerprinting/reporting) should use from here on -- none of it needs to
    re-derive resolved settings from the raw ``cfg`` dict again. Callers that
    only want the pass/fail check can call this and ignore the return value.

    :param cfg: The loaded configuration
    :type cfg: dict
    :return: The resolved, validated config
    :rtype: SimConfig
    :raises ValueError: if ``cfg`` doesn't match the expected schema
    """
    _check_structure(cfg)
    try:
        return SimConfig.model_validate(_materialize_config(cfg))
    except ValidationError as e:
        raise ValueError(str(e)) from e


_DEFAULT_AGENT_SETTINGS: dict = AgentSettings().model_dump()
_DEFAULT_WORLD_NOISE: dict = WorldSection.model_fields["noise"].get_default(
    call_default_factory=True
)
_DEFAULT_WORLD_OBSERVATION: dict = WorldObservation().model_dump()


def _materialize_agent_profiles(cfg: dict) -> list[dict]:
    """Expand ``agent.defaults`` + ``agent.profiles`` into one fully-specified,
    JSON-safe settings dict per profile.

    Every Agent-recognized field is present regardless of what the YAML
    omitted -- omitted fields are filled from ``config_schema.AgentSettings``'
    own defaults. This is a pure transformation; callers must ensure ``cfg``
    has already passed ``_check_structure``.

    :param cfg: The loaded, validated configuration
    :type cfg: dict
    :return: One ``{"name":, "count":, **settings}`` dict per profile
    :rtype: list[dict]
    """
    agent_cfg = cfg["agent"]
    base = deep_merge(_DEFAULT_AGENT_SETTINGS, agent_cfg["defaults"])

    profiles: list[dict] = []
    for profile in agent_cfg["profiles"]:
        name = profile["name"]
        count = profile["count"]
        overrides = {k: v for k, v in profile.items() if k not in {"name", "count"}}
        merged = deep_merge(base, overrides)
        profiles.append({"name": name, "count": count, **merged})

    return profiles


def expand_agent_specs(cfg: SimConfig) -> list[AgentProfile]:
    """Expand ``cfg.agent.profiles`` into one resolved profile per agent.

    ``cfg.agent.profiles`` is already fully resolved (``validate_config``
    merged defaults in and validated the result), so this only needs to
    replicate each profile ``count`` times -- no re-merging, and no
    translation, since ``Agent`` accepts an ``AgentSettings``/``AgentProfile``
    directly.

    :param cfg: The resolved, validated config
    :type cfg: SimConfig
    :return: One resolved ``AgentProfile`` per agent
    :rtype: list[AgentProfile]
    """
    specs: list[AgentProfile] = []
    for profile in cfg.agent.profiles:
        specs.extend(profile for _ in range(profile.count))
    return specs


def _materialize_world_settings(cfg: dict) -> dict:
    """Return the ``world`` section with every field explicit, including noise
    and observation keys the YAML omitted (defaulted from
    ``config_schema.WorldSection``/``WorldObservation``).

    This is a pure transformation; callers must ensure ``cfg`` has already
    passed ``_check_structure``.

    :param cfg: The loaded, validated configuration
    :type cfg: dict
    :return: The fully-materialized ``world`` section
    :rtype: dict
    """
    world_cfg = cfg["world"]
    return {
        # Spread first so an unrecognized key still reaches extra="forbid";
        # explicit keys below still win.
        **world_cfg,
        "rng_seed": world_cfg["rng_seed"],
        "truths": dict(world_cfg["truths"]),
        "noise": {**_DEFAULT_WORLD_NOISE, **world_cfg.get("noise", {})},
        "observation": {
            **_DEFAULT_WORLD_OBSERVATION,
            **world_cfg.get("observation", {}),
        },
    }


def _materialize_config(cfg: dict) -> dict:
    """Return the fully effective configuration -- every field explicit, no
    field silently defaulted downstream by Agent/World construction.

    Internal: the dict shape ``SimConfig.model_validate`` consumes inside
    ``validate_config``. Nothing outside this module needs it -- once a caller
    holds a ``SimConfig``, ``.model_dump()``/``.model_dump(exclude=...)``
    covers the same need (a reproducibility record to hash or store) without
    re-deriving it from the raw ``cfg`` dict.

    :param cfg: The loaded configuration
    :type cfg: dict
    :return: The fully effective configuration, keyed like the source YAML
    :rtype: dict
    """
    # Spread cfg/agent first so an unrecognized key at either level still
    # reaches extra="forbid"; explicit keys below still win. "defaults" is
    # deliberately excluded -- it's fully consumed by _materialize_agent_
    # profiles, not part of the validated shape.
    agent_extra = {k: v for k, v in cfg["agent"].items() if k != "defaults"}
    return {
        **cfg,
        "world": _materialize_world_settings(cfg),
        "agent": {**agent_extra, "profiles": _materialize_agent_profiles(cfg)},
    }


def world_from_config(cfg: SimConfig) -> World:
    """Build a World instance from a resolved, validated configuration.

    :param cfg: The resolved, validated config (see ``load_config``/
        ``validate_config``)
    :type cfg: SimConfig
    :return: The constructed world, with its agents
    :rtype: World
    """
    profiles = expand_agent_specs(cfg)
    agents = [
        Agent(
            id=i,
            settings=profile,
            rng_seed=cfg.world.rng_seed
            + i
            + 1,  # add i to differ seed, and 1 to offset from world rng
            profile_name=profile.name,
        )
        for i, profile in enumerate(profiles)
    ]
    return World(agents=agents, settings=cfg.world)
