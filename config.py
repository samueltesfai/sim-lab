from omegaconf import OmegaConf
import os


def load_config(path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = OmegaConf.load(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: OmegaConf) -> None:
    """Perform light validation on configuration."""
    # World validation
    if cfg.world.num_agents <= 0:
        raise ValueError("world.num_agents must be > 0")

    if not 0 <= cfg.world.observation_probability <= 1:
        raise ValueError("world.observation_probability must be in [0, 1]")

    # Noise validation
    for noise_type in ["OBSERVE", "HEAR", "VERIFY"]:
        if cfg.world.noise[noise_type] < 0:
            raise ValueError(f"world.noise.{noise_type} must be non-negative")

    # Action preference validation
    valid_actions = {"IDLE", "VERIFY", "COMMUNICATE", "BROADCAST"}
    for action in cfg.agent.action_preference:
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}")
        if not 0 <= cfg.agent.action_preference[action] <= 1:
            raise ValueError(f"agent.action_preference.{action} must be in [0, 1]")

    # Action cost validation
    for action in cfg.agent.action_cost:
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}")
        if cfg.agent.action_cost[action] < 0:
            raise ValueError(f"agent.action_cost.{action} must be non-negative")

    # Truths validation
    for claim_id, truth in cfg.world.truths.items():
        if not isinstance(truth, bool):
            raise ValueError(f"world.truths.{claim_id} must be boolean")


def convert_action_strings(cfg: OmegaConf) -> OmegaConf:
    """Convert string action keys to ActionType enums."""
    # Import here to avoid circular import
    from sim import ActionType

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
    # Import here to avoid circular import
    from sim import MemoryType

    noise_dict = {}
    for noise_str, value in cfg.world.noise.items():
        noise_dict[MemoryType[noise_str]] = value
    cfg.world.noise = noise_dict

    return cfg


def build_world(cfg: OmegaConf):
    """Build a World instance from configuration."""
    # Import here to avoid circular import
    from sim import World, Agent

    # Convert string keys to enums
    cfg = convert_action_strings(cfg)
    cfg = convert_noise_strings(cfg)

    # Convert OmegaConf DictConfig to regular dicts for constructors
    action_preference = dict(cfg.agent.action_preference)
    action_cost = dict(cfg.agent.action_cost)
    noise = dict(cfg.world.noise)

    # Create agents
    agents = []
    for i in range(cfg.world.num_agents):
        agent = Agent(
            id=i,
            rng_seed=cfg.world.rng_seed
            + i
            + 1,  # add i to differ seed, and 1 to offset from world rng
            action_preference=action_preference,
            action_cost=action_cost,
        )
        agents.append(agent)

    # Create world
    world = World(
        agents=agents,
        truths=cfg.world.truths,
        rng_seed=cfg.world.rng_seed,
        noise=noise,
        observation_probability=cfg.world.observation_probability,
    )

    return world
