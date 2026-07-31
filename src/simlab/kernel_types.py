from dataclasses import dataclass, field
from enum import Enum


def clamp(value, min_value=0.0, max_value=1.0):
    """The only utility function we need."""
    return max(min_value, min(value, max_value))


class ActionType(Enum):
    IDLE = "idle"
    VERIFY = "verify"
    COMMUNICATE = "communicate"
    BROADCAST = "broadcast"


class MemoryType(Enum):
    OBSERVE = "observed"
    VERIFY = "verified"
    HEAR = "heard"


@dataclass
class Action:
    type: ActionType
    claim_id: int | None = None
    target_agent_id: int | None = None

    def __post_init__(self):
        """Validate action parameters based on type."""
        if isinstance(self.type, str):
            self.type = ActionType(self.type)
        if self.type == ActionType.VERIFY and self.claim_id is None:
            raise ValueError("VERIFY action requires claim_id")
        if self.type == ActionType.COMMUNICATE:
            if self.claim_id is None or self.target_agent_id is None:
                raise ValueError(
                    "COMMUNICATE action requires claim_id and target_agent_id"
                )
        if self.type == ActionType.BROADCAST and (
            self.claim_id is None or self.target_agent_id is not None
        ):
            raise ValueError(
                "BROADCAST action requires claim_id and no target_agent_id"
            )


@dataclass
class Memory:
    id: int
    type: MemoryType
    timestamp: int
    source: int | None
    claim_id: int | None
    evidence: float | None


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """A world-generated observation opportunity.

    The world produces these; agents may notice and encode them into memories.
    An event carries a truth-grounded evidence signal and the agents who could
    potentially perceive it.
    """

    id: int
    tick: int
    claim_id: int
    evidence: float
    visible_agent_ids: tuple[int, ...]


@dataclass(slots=True)
class Snapshot:
    tick: int  # Current simulation tick
    observation_event_count: int  # Number of world observation events emitted this tick
    observed_ids: list[int]  # List of agent IDs that observed a claim this tick
    verified_ids: list[int]  # List of agent IDs that verified a claim this tick
    communicate_edges: list[
        tuple[int, int]
    ]  # List of (source, target) agent pairs that communicated this tick
    broadcast_edges: list[
        tuple[int, int]
    ]  # List of (source, target) agent pairs that broadcasted this tick
    num_memory_processing_agents: int  # Agents that processed >=1 new memory this tick
    num_belief_updating_agents: int  # Agents whose belief values changed this tick
    num_trust_updating_agents: int  # Agents whose trust values changed this tick
    agent_beliefs: dict[int, dict[int, float]]  # {agent_id: {claim_id: belief}}
    agent_memory_sizes: dict[int, int]  # {agent_id: memory_size}


@dataclass(slots=True)
class ActionTrace:
    """Records what a single action execution caused, for accumulation in step()."""

    verified_ids: list[int] = field(default_factory=list)
    communicate_edges: list[tuple[int, int]] = field(default_factory=list)
    broadcast_edges: list[tuple[int, int]] = field(default_factory=list)


@dataclass(slots=True)
class MemoryProcessTrace:
    """Records the effect of processing a single memory, for accumulation in
    Agent.update_beliefs()."""

    belief_changed: bool = False
    trust_changed: bool = False


@dataclass(slots=True)
class AgentUpdateTrace:
    """Records what Agent.update_beliefs() did this tick, for accumulation in
    World.step()."""

    processed_memory: bool = False
    belief_changed: bool = False
    trust_changed: bool = False
