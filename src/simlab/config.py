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
    """Validate the action maps inside an agent settings node (defaults/profile)."""
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


def validate_config(cfg: OmegaConf) -> None:
    """Perform light validation on configuration."""
    # World validation
    if cfg.world.num_agents <= 0:
        raise ValueError("world.num_agents must be > 0")

    rate = cfg.world.observation.individual_event_rate
    if not 0 <= rate <= 1:
        raise ValueError("world.observation.individual_event_rate must be in [0, 1]")

    # Noise validation
    for noise_type in ["OBSERVE", "HEAR", "VERIFY"]:
        if cfg.world.noise[noise_type] < 0:
            raise ValueError(f"world.noise.{noise_type} must be non-negative")

    # Agent validation: support both the flat form (action_preference/action_cost
    # directly under `agent`) and the structured form (defaults + profiles).
    if "defaults" in cfg.agent or "profiles" in cfg.agent:
        if "defaults" in cfg.agent:
            _validate_agent_settings(cfg.agent.defaults, context="agent.defaults")
        if "profiles" in cfg.agent:
            total = 0
            for profile in cfg.agent.profiles:
                if "name" not in profile:
                    raise ValueError("agent.profiles entries require a name")
                if "count" not in profile or profile.count < 0:
                    raise ValueError(
                        f"agent.profiles.{profile.get('name')} requires a non-negative count"
                    )
                total += profile.count
                _validate_agent_settings(
                    profile, context=f"agent.profiles.{profile.name}"
                )
            if total != cfg.world.num_agents:
                raise ValueError(
                    "agent.profiles counts must sum to world.num_agents "
                    f"(got {total}, expected {cfg.world.num_agents})"
                )
    else:
        _validate_agent_settings(cfg.agent, context="agent")

    # Truths validation
    for claim_id, truth in cfg.world.truths.items():
        if not isinstance(truth, bool):
            raise ValueError(f"world.truths.{claim_id} must be boolean")


def convert_action_strings(cfg: OmegaConf) -> OmegaConf:
    """Convert string action keys to ActionType enums (flat agent form)."""

    # Convert action preferences
    pref_dict = {}
    for action_str, value in cfg.agent.action_preference.items():
        pref_dict[ActionType[action_str]] = value
    cfg.agent.action_preference = pref_dict

    # Convert action costs
    cost_dict = {}
    for action_str, value in cfg.agent.action_cost.items():
        cost_dict[ActionType[action_str]] = value
    cfg.agent.action_cost = cost_dict

    return cfg


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

    return kwargs


def expand_agent_specs(cfg: OmegaConf) -> list[dict]:
    """Expand the agent config into one Agent kwargs spec per agent.

    Supports both the flat form (action maps directly under ``agent``) and the
    structured form (``agent.defaults`` + ``agent.profiles``). When no profiles
    are given, a single implicit ``default`` profile covering all agents is used.
    """
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)

    if "defaults" in agent_cfg or "profiles" in agent_cfg:
        defaults = agent_cfg.get("defaults", {})
        profiles = agent_cfg.get("profiles")
    else:
        # Flat form: the whole agent node acts as the defaults for one profile.
        defaults = agent_cfg
        profiles = None

    if not profiles:
        profiles = [{"name": "default", "count": cfg.world.num_agents}]

    specs: list[dict] = []
    for profile in profiles:
        name = profile.get("name", "default")
        count = profile.get("count", 0)
        overrides = {k: v for k, v in profile.items() if k not in {"name", "count"}}
        merged = _deep_merge(defaults, overrides)
        kwargs = _settings_to_agent_kwargs(merged, name)
        specs.extend(kwargs for _ in range(count))

    return specs


def build_world(cfg: OmegaConf):
    """Build a World instance from configuration."""

    # World noise -> enum-keyed dict (does not depend on agent form).
    noise = {
        MemoryType[noise_str]: value for noise_str, value in cfg.world.noise.items()
    }

    # Expand agents (handles both flat and profile-based configs).
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
        individual_observation_event_rate=cfg.world.observation.individual_event_rate,
    )

    return world
