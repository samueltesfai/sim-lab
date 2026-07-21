from omegaconf import OmegaConf
import os
from simlab.sim import World, Agent, ActionType, MemoryType


VALID_ACTIONS = {"IDLE", "VERIFY", "COMMUNICATE", "BROADCAST"}


def load_config(path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = OmegaConf.load(path)
    validate_config(cfg)
    return cfg


def _validate_action_map(
    action_map, *, kind: str, context: str, max_value: float | None
) -> None:
    """Validate an action_preference or action_cost mapping.

    :param action_map: Mapping of action name -> value
    :param kind: "preference" or "cost" (used in error messages)
    :param context: Human-readable location for error messages
    :param max_value: Upper bound (1.0 for preferences); None for costs (>= 0)
    """
    for action in action_map:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}")
        value = action_map[action]
        if max_value is not None:
            if not 0 <= value <= max_value:
                raise ValueError(f"{context}.action_{kind}.{action} must be in [0, 1]")
        elif value < 0:
            raise ValueError(f"{context}.action_{kind}.{action} must be non-negative")


def _validate_agent_settings(settings, *, context: str) -> None:
    """Validate the maps inside an agent settings node (defaults/profile)."""
    if "action_preference" in settings:
        _validate_action_map(
            settings.action_preference,
            kind="preference",
            context=context,
            max_value=1.0,
        )
    if "action_cost" in settings:
        _validate_action_map(
            settings.action_cost,
            kind="cost",
            context=context,
            max_value=None,
        )

    observation = settings.get("observation", {})
    if "attention" in observation and not 0 <= observation.attention <= 1:
        raise ValueError(f"{context}.observation.attention must be in [0, 1]")
    if "bias" in observation and not -1 <= observation.bias <= 1:
        raise ValueError(f"{context}.observation.bias must be in [-1, 1]")

    social = settings.get("social", {})
    if "confidence_bound" in social and not 0 <= social.confidence_bound <= 1:
        raise ValueError(f"{context}.social.confidence_bound must be in [0, 1]")
    if "trust_update_rate" in social and not 0 <= social.trust_update_rate <= 1:
        raise ValueError(f"{context}.social.trust_update_rate must be in [0, 1]")
    if "update_trust_on_rejection" in social and not isinstance(
        social.update_trust_on_rejection, bool
    ):
        raise ValueError(f"{context}.social.update_trust_on_rejection must be boolean")


def _validate_profile_count(count, name: str) -> None:
    """Reject missing, non-integral, or non-positive profile counts."""
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"agent profile {name} count must be a positive integer")


def validate_config(cfg: OmegaConf) -> None:
    """Perform light validation on configuration."""
    private_rate = cfg.world.observation.private_event_rate
    if not 0 <= private_rate <= 1:
        raise ValueError("world.observation.private_event_rate must be in [0, 1]")

    global_rate = cfg.world.observation.global_event_rate
    if not 0 <= global_rate <= 1:
        raise ValueError("world.observation.global_event_rate must be in [0, 1]")

    # Noise validation: only provided values are checked; missing keys are
    # filled in with defaults by ``build_world``.
    for noise_type in ["OBSERVE", "HEAR", "VERIFY"]:
        if noise_type in cfg.world.noise and cfg.world.noise[noise_type] < 0:
            raise ValueError(f"world.noise.{noise_type} must be non-negative")

    # Agent validation. There is exactly one canonical schema:
    #   agent.defaults  -> baseline cognitive/action parameters
    #   agent.profiles  -> concrete subpopulations (each with a count)
    # The total number of agents is the sum of the profile counts.
    if "defaults" not in cfg.agent:
        raise ValueError("agent.defaults is required")
    if "profiles" not in cfg.agent:
        raise ValueError("agent.profiles is required")

    _validate_agent_settings(cfg.agent.defaults, context="agent.defaults")

    if not cfg.agent.profiles:
        raise ValueError("agent.profiles must contain at least one profile")

    for profile in cfg.agent.profiles:
        if "name" not in profile:
            raise ValueError("each agent profile must define name")
        name = profile.name
        _validate_profile_count(profile.count if "count" in profile else None, name)
        _validate_agent_settings(profile, context=f"agent.profiles.{name}")

    # Truths validation
    for claim_id, truth in cfg.world.truths.items():
        if not isinstance(truth, bool):
            raise ValueError(f"world.truths.{claim_id} must be boolean")


def convert_noise_strings(cfg: OmegaConf) -> OmegaConf:
    """Convert string noise keys to MemoryType enums."""

    noise_dict = {}
    for noise_str, value in cfg.world.noise.items():
        noise_dict[MemoryType[noise_str]] = value
    cfg.world.noise = noise_dict

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (one level deep dicts)."""
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _settings_to_agent_kwargs(settings: dict, profile_name: str) -> dict:
    """Translate a merged agent settings node into Agent constructor kwargs.

    Only keys present in ``settings`` are passed through; the Agent applies its
    own defaults for anything omitted.
    """
    kwargs: dict = {"profile_name": profile_name}

    if "action_preference" in settings:
        kwargs["action_preference"] = {
            ActionType[k]: v for k, v in settings["action_preference"].items()
        }
    if "action_cost" in settings:
        kwargs["action_cost"] = {
            ActionType[k]: v for k, v in settings["action_cost"].items()
        }

    observation = settings.get("observation", {})
    if "attention" in observation:
        kwargs["observation_attention"] = observation["attention"]
    if "bias" in observation:
        kwargs["observation_bias"] = observation["bias"]

    trust = settings.get("trust", {})
    if "default" in trust:
        kwargs["default_trust"] = trust["default"]

    learning = settings.get("learning", {})
    if "rate" in learning:
        kwargs["learning_rate"] = learning["rate"]
    if "observe_weight" in learning:
        kwargs["observe_weight"] = learning["observe_weight"]
    if "hear_weight" in learning:
        kwargs["hear_weight"] = learning["hear_weight"]
    if "verify_weight" in learning:
        kwargs["verify_weight"] = learning["verify_weight"]

    social = settings.get("social", {})
    if "confidence_bound" in social:
        kwargs["social_confidence_bound"] = social["confidence_bound"]
    if "trust_update_rate" in social:
        kwargs["social_trust_update_rate"] = social["trust_update_rate"]
    if "update_trust_on_rejection" in social:
        kwargs["social_update_trust_on_rejection"] = social["update_trust_on_rejection"]

    return kwargs


def expand_agent_specs(cfg: OmegaConf) -> list[dict]:
    """Expand ``agent.defaults`` + ``agent.profiles`` into one Agent spec per agent.

    Each profile inherits ``agent.defaults`` and may override any subset of
    settings. The total number of agents is the sum of the profile counts.

    This is a pure transformation; callers must ensure ``cfg`` has already
    passed ``validate_config``.
    """
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)

    specs: list[dict] = []
    for profile in agent_cfg["profiles"]:
        name = profile["name"]
        count = profile["count"]
        overrides = {k: v for k, v in profile.items() if k not in {"name", "count"}}
        merged = _deep_merge(agent_cfg["defaults"], overrides)
        kwargs = _settings_to_agent_kwargs(merged, name)
        specs.extend(dict(kwargs) for _ in range(count))

    return specs


def build_world(cfg: OmegaConf):
    """Build a World instance from a validated configuration."""

    # World noise -> enum-keyed dict.
    noise = {
        MemoryType[noise_str]: value for noise_str, value in cfg.world.noise.items()
    }

    # Expand agent.defaults + agent.profiles into concrete agents.
    specs = expand_agent_specs(cfg)

    agents = []
    for i, spec in enumerate(specs):
        agent = Agent(
            id=i,
            rng_seed=cfg.world.rng_seed
            + i
            + 1,  # add i to differ seed, and 1 to offset from world rng
            **spec,
        )
        agents.append(agent)

    # Create world
    world = World(
        agents=agents,
        truths=cfg.world.truths,
        rng_seed=cfg.world.rng_seed,
        noise=noise,
        private_event_rate=cfg.world.observation.private_event_rate,
        global_event_rate=cfg.world.observation.global_event_rate,
    )

    return world
