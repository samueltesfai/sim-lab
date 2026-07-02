from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import random


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
    n_agent_updates: int  # Number of agents that updated this tick
    agent_beliefs: dict[int, dict[int, float]]  # {agent_id: {claim_id: belief}}
    agent_memory_sizes: dict[int, int]  # {agent_id: memory_size}


class Agent:
    def __init__(
        self,
        id: int,
        rng_seed: int = 0,
        action_preference: dict[ActionType, float] | None = None,
        action_cost: dict[ActionType, float] | None = None,
        profile_name: str = "default",
        observation_attention: float = 1.0,
        observation_bias: float = 0.0,
        default_trust: float = 0.5,
        learning_rate: float = 0.1,
        observe_weight: float = 0.6,
        hear_weight: float = 0.3,
        verify_weight: float = 1.0,
        social_confidence_bound: float = 1.0,
        social_trust_update_rate: float = 0.0,
        social_update_trust_on_rejection: bool = True,
    ):
        self.id = id
        self.rng = random.Random(rng_seed)
        self.profile_name = profile_name

        # Cognition parameters: how this kind of mind perceives and learns.
        self.observation_attention = observation_attention  # P(notice an event)
        self.observation_bias = observation_bias  # systematic perceptual bias
        self.default_trust = default_trust  # trust for unseen agents
        self.learning_rate = learning_rate  # global plasticity
        self.observe_weight = observe_weight  # channel weight for OBSERVE
        self.hear_weight = hear_weight  # channel weight for HEAR
        self.verify_weight = verify_weight  # channel weight for VERIFY
        self.social_confidence_bound = (
            social_confidence_bound  # max distance for HEAR to update belief
        )
        self.social_trust_update_rate = (
            social_trust_update_rate  # rate of dynamic trust adjustment
        )
        self.social_update_trust_on_rejection = (
            social_update_trust_on_rejection  # update trust even when HEAR rejected
        )

        self.beliefs: defaultdict[int, float] = defaultdict(lambda: self.rng.random())
        self.trust: defaultdict[int, float] = defaultdict(lambda: self.default_trust)
        self.memory: list[Memory] = []
        self._mem_cursor = 0  # Cursor to track memories for belief updates
        default_action_preference = {
            ActionType.IDLE: 0.0,
            ActionType.VERIFY: 0.9,
            ActionType.COMMUNICATE: 0.7,
            ActionType.BROADCAST: 0.5,
        }

        default_action_cost = {
            ActionType.IDLE: 0.0,
            ActionType.VERIFY: 0.35,
            ActionType.COMMUNICATE: 0.15,
            ActionType.BROADCAST: 0.30,
        }
        self.action_preference: dict[ActionType, float] = default_action_preference | (
            action_preference or {}
        )
        self.action_cost: dict[ActionType, float] = default_action_cost | (
            action_cost or {}
        )

    def __repr__(self):
        return f"Agent(id={self.id}, profile={self.profile_name!r}, beliefs={dict(self.beliefs)}, trust={dict(self.trust)}, memory={len(self.memory)})"

    def _add_memory(
        self,
        *,
        tick: int,
        memory_type: MemoryType,
        claim_id: int,
        evidence: float,
        source: int | None = None,
    ) -> None:
        """
        Add a memory to the agent's memory list.

        Memories only store already-formed evidence. Evidence is produced by
        the world (observation/verification) or by social interaction (hearing)
        and encoded by the agent before being stored here, so every memory
        type requires an explicit ``claim_id`` and ``evidence``.

        This is internal model mechanics: memories are created on the agent's
        behalf by the world during action execution and delivery, not by
        external callers.

        :param self:
        :param tick: The simulation tick at which the memory is formed
        :type tick: int
        :param memory_type: The type of memory being added
        :type memory_type: MemoryType
        :param claim_id: The ID of the claim for the memory
        :type claim_id: int
        :param evidence: The evidence value for the memory
        :type evidence: float
        :param source: The ID of the source agent for the memory (if applicable)
        :type source: int | None
        """
        memory = Memory(
            id=len(self.memory),
            type=memory_type,
            timestamp=tick,
            source=source,
            claim_id=claim_id,
            evidence=evidence,
        )
        self.memory.append(memory)

    def notices_observation(self, event: ObservationEvent) -> bool:
        """
        Decide whether this agent notices an available observation event.

        Attention is the probability that this kind of mind attends to an
        observation opportunity presented by the world.

        :param self:
        :param event: The observation event presented to the agent
        :type event: ObservationEvent
        :return: True if the agent notices the event
        :rtype: bool
        """
        # Deterministic endpoints: avoid consuming any RNG so that the volume of
        # observation events an agent is offered cannot shift the main RNG stream
        # used for lazy belief initialization and action choices.
        if self.observation_attention <= 0.0:
            return False
        if self.observation_attention >= 1.0:
            return True
        return self.rng.random() < self.observation_attention

    def encode_observation(self, event: ObservationEvent) -> float:
        """
        Encode a noticed observation event into subjective evidence.

        The world already produced a truth-grounded, noisy signal. Encoding
        applies the agent's systematic perceptual bias. (Perceptual noise is
        intentionally not re-applied here to avoid double-counting the world's
        observation noise.)

        :param self:
        :param event: The observation event being encoded
        :type event: ObservationEvent
        :return: The subjectively encoded evidence value in [0, 1]
        :rtype: float
        """
        return clamp(event.evidence + self.observation_bias)

    @property
    def memory_size(self) -> int:
        """
        Get the size of the agent's memory.

        :param self:
        :return: The size of the agent's memory
        :rtype: int
        """
        return len(self.memory)

    def confidence(self, claim_id: int) -> float:
        """
        Calculate the confidence in a claim based on the agent's current beliefs
        [0.0, 1.0].

        :param self:
        :param claim_id: The ID of the claim
        :type claim_id: int
        :return: The confidence in the claim
        :rtype: float
        """
        return abs(self.beliefs[claim_id] - 0.5) * 2

    def uncertainty(self, claim_id: int) -> float:
        """
        Calculate the uncertainty in a claim based on the agent's current beliefs
        [0.0, 1.0].

        :param self:
        :param claim_id: The ID of the claim
        :type claim_id: int
        :return: The uncertainty in the claim
        :rtype: float
        """
        return 1.0 - self.confidence(claim_id)

    def disagreement(self, claim_id: int, agent_id: int, world: "World") -> float:
        """
        Calculate the disagreement in a claim based on the agent's current beliefs
        [0.0, 1.0].

        :param self:
        :param claim_id: The ID of the claim
        :type claim_id: int
        :param agent_id: The ID of the agent
        :type agent_id: int
        :param world: The world in which the agent is generating candidate actions
        :type world: 'World'
        :return: The disagreement in the claim
        :rtype: float
        """
        neighbor = world.get_agent(agent_id)
        return abs(self.beliefs[claim_id] - neighbor.beliefs[claim_id])

    def local_disagreement(self, claim_id: int, world: "World") -> float:
        """
        Calculate the local disagreement in a claim based on the agent's current
        beliefs and their neighbors' beliefs.

        :param self:
        :param claim_id: The ID of the claim
        :type claim_id: int
        :param world: The world in which the agent is generating candidate actions
        :type world: 'World'
        :return: The local disagreement in the claim
        :rtype: float
        """
        n_neighbors = len(world.network[self.id])
        if n_neighbors == 0:
            return 0.0
        return (
            sum(
                self.disagreement(claim_id, neighbor_id, world)
                for neighbor_id in world.network[self.id]
            )
            / n_neighbors
        )

    def generate_candidate_actions(self, world: "World") -> list[Action]:
        """
        Generate a list of candidate actions for the agent to choose from.

        :param self:
        :param world: The world in which the agent is generating candidate actions
        :type world: 'World'
        :return: A list of candidate actions
        :rtype: list[Action]
        """
        candidates = [Action(ActionType.IDLE)]

        for claim_id in world.truths:
            candidates.append(Action(ActionType.VERIFY, claim_id=claim_id))
            candidates.append(Action(ActionType.BROADCAST, claim_id=claim_id))

            for neighbor_id in world.network[self.id]:
                candidates.append(
                    Action(
                        ActionType.COMMUNICATE,
                        claim_id=claim_id,
                        target_agent_id=neighbor_id,
                    )
                )

        return candidates

    def score_action(self, world: "World", action: Action) -> float:
        """
        Score an action based on the agent's current state and the world.

        :param self:
        :param world: The world in which the agent is scoring the action
        :type world: 'World'
        :param action: The action to score
        :type action: Action
        :return: The score of the action
        :rtype: float
        """

        match action.type:
            case ActionType.VERIFY:
                return (
                    self.action_preference[action.type]
                    * self.uncertainty(action.claim_id)
                    - self.action_cost[action.type]
                )
            case ActionType.COMMUNICATE:
                return (
                    self.action_preference[action.type]
                    * self.confidence(action.claim_id)
                    * self.disagreement(action.claim_id, action.target_agent_id, world)
                    - self.action_cost[action.type]
                )
            case ActionType.BROADCAST:
                return (
                    self.action_preference[action.type]
                    * self.confidence(action.claim_id)
                    * self.local_disagreement(action.claim_id, world)
                    - self.action_cost[action.type]
                )
            case ActionType.IDLE:
                return (
                    self.action_preference[action.type] - self.action_cost[action.type]
                )
            case _:
                raise ValueError(
                    f"Unknown action type: {action.type} for action {action}"
                )

    def choose_action(self, world: "World") -> Action:
        """
        Choose an action based on the agent's current state and the world.

        :param self:
        :param world: The world in which the agent is choosing an action
        :type world: 'World'
        :return: The chosen action
        :rtype: Action
        """
        return max(
            self.generate_candidate_actions(world),
            key=lambda action: self.score_action(world, action),
        )

    def update_beliefs(self) -> bool:
        """
        Update the agent's beliefs based on accumulated memories.

        The effective learning rate for each memory is the agent's global
        plasticity (``learning_rate``) scaled by a channel-specific weight, and
        further modulated by trust for socially heard memories. HEAR memories
        are additionally subject to bounded confidence and dynamic trust updates.

        All newly accumulated memories are consumed regardless of the return
        value; trust may still change for HEAR memories even when no belief
        moves.

        :param self:
        :return: True if at least one belief value changed, False otherwise
        :rtype: bool
        """
        belief_changed = False
        while self._mem_cursor < len(self.memory):
            mem = self.memory[self._mem_cursor]
            if self._process_memory(mem):
                belief_changed = True
            self._mem_cursor += 1
        return belief_changed

    def _process_memory(self, mem: Memory) -> bool:
        """
        Apply a single memory to beliefs, with trust side-effects for HEAR.

        :param mem: The memory to process
        :type mem: Memory
        :return: True if the belief value for the memory's claim changed
        :rtype: bool
        """
        if mem.claim_id is None or mem.evidence is None:
            return False

        belief_before = self.beliefs[mem.claim_id]
        lr = clamp(self._effective_learning_rate(mem, belief_before))
        belief_after = clamp(belief_before + lr * (mem.evidence - belief_before))
        self.beliefs[mem.claim_id] = belief_after

        if mem.type == MemoryType.HEAR:
            accepted = self._should_accept_heard_memory(mem, belief_before)
            self._update_trust_from_heard_memory(mem, belief_before, accepted)

        return belief_after != belief_before

    def _effective_learning_rate(self, mem: Memory, belief_before: float) -> float:
        """
        Compute the effective learning rate for a memory before clamping.

        For HEAR memories, returns 0.0 when the heard evidence falls outside
        the agent's social confidence bound.

        :param mem: The memory being processed
        :type mem: Memory
        :param belief_before: The agent's belief for the claim before this update
        :type belief_before: float
        :return: Raw (unclamped) effective learning rate
        :rtype: float
        """
        match mem.type:
            case MemoryType.OBSERVE:
                return self.learning_rate * self.observe_weight
            case MemoryType.VERIFY:
                return self.learning_rate * self.verify_weight
            case MemoryType.HEAR:
                if mem.source is None:
                    return 0.0
                if not self._should_accept_heard_memory(mem, belief_before):
                    return 0.0
                return self.learning_rate * self.hear_weight * self.trust[mem.source]
            case _:
                return 0.0

    def _should_accept_heard_memory(self, mem: Memory, belief_before: float) -> bool:
        """
        Return True when heard evidence is within the agent's confidence bound.

        :param mem: The HEAR memory being evaluated
        :type mem: Memory
        :param belief_before: The agent's belief for the claim before this update
        :type belief_before: float
        :return: True if the memory is within the confidence bound
        :rtype: bool
        """
        distance = abs(mem.evidence - belief_before)
        return distance <= self.social_confidence_bound

    def _update_trust_from_heard_memory(
        self,
        mem: Memory,
        belief_before: float,
        accepted: bool,
    ) -> None:
        """
        Update trust in the source agent based on agreement with heard evidence.

        No-op when ``social_trust_update_rate`` is 0 or when the memory was
        rejected and ``social_update_trust_on_rejection`` is False.

        :param mem: The HEAR memory being evaluated
        :type mem: Memory
        :param belief_before: The agent's belief for the claim before this update
        :type belief_before: float
        :param accepted: Whether the memory was accepted by bounded confidence
        :type accepted: bool
        """
        if self.social_trust_update_rate == 0.0:
            return
        if not accepted and not self.social_update_trust_on_rejection:
            return
        if mem.source is None:
            return
        agreement = 1.0 - abs(mem.evidence - belief_before)
        trust_delta = self.social_trust_update_rate * (
            agreement - self.trust[mem.source]
        )
        self.trust[mem.source] = clamp(self.trust[mem.source] + trust_delta)


class World:
    def __init__(
        self,
        agents: list[Agent],
        truths: dict[int, bool],
        rng_seed: int = 0,
        noise: dict[MemoryType, float] | None = None,
        private_event_rate: float = 0.1,
        global_event_rate: float = 0.0,
    ):
        self._agents = {a.id: a for a in agents}
        self.tick = 0
        self.rng = random.Random(rng_seed)
        self.noise = {
            MemoryType.OBSERVE: 0.0,
            MemoryType.HEAR: 0.0,
            MemoryType.VERIFY: 0.0,
        } | (noise or {})
        self.truths = truths
        # private_event_rate: per-agent per-tick chance of a private observation
        #   event visible only to that agent.
        # global_event_rate: per-tick chance of one shared observation event
        #   visible to all agents.
        self.private_event_rate = private_event_rate
        self.global_event_rate = global_event_rate
        self._next_event_id = 0
        self.network = self._generate_dummy_network(
            # TODO: We can implement a more complex network generation mechanism here,
            # potentially based on real-world social network structures or using a
            # configurable graph model.
            agents
        )

    def _generate_dummy_network(self, agents: list[Agent]) -> dict[int, list[int]]:
        network = defaultdict(list)
        max_degree = min(4, len(agents) - 1)
        for agent in agents:
            connections = self.rng.sample(
                [a.id for a in agents if a.id != agent.id],
                k=self.rng.randint(0, max_degree),  # Allows for isolated agents
            )
            network[agent.id] = connections
        return network

    def __repr__(self):
        return f"World(tick={self.tick}, agents={len(self._agents)}, truths={self.truths}, network={dict(self.network)})"

    @property
    def agents(self) -> list[Agent]:
        return list(self._agents.values())

    @property
    def claims(self) -> list[int]:
        return list(self.truths.keys())

    @property
    def edges(self) -> list[tuple[int, int]]:
        return [(src, dest) for src, nei in self.network.items() for dest in nei]

    @property
    def profile_counts(self) -> dict[str, int]:
        """Return the number of agents per profile name."""
        counts: defaultdict[str, int] = defaultdict(int)
        for agent in self.agents:
            counts[agent.profile_name] += 1
        return dict(counts)

    def get_agent_beliefs_snapshot(self) -> dict[int, dict[int, float]]:
        """
        Return a complete belief snapshot for all agents and all known claims.

        This intentionally indexes into each agent's belief defaultdict using the
        world's known claim IDs, rather than calling dict(agent.beliefs), because
        lazy defaultdict entries may not exist until accessed.
        """
        claim_ids = list(self.truths.keys())

        return {
            agent.id: {claim_id: agent.beliefs[claim_id] for claim_id in claim_ids}
            for agent in self.agents
        }

    def get_agent(self, agent_id: int) -> Agent:
        return self._agents[agent_id]

    def _generate_observation_evidence(self, claim_id: int) -> float:
        """
        Generate truth-grounded observation evidence for a claim.

        Observation evidence reflects objective truth plus environmental
        observation noise. Subjective perceptual distortion is applied later by
        the observing agent during encoding.

        :param claim_id: The ID of the claim being observed
        :type claim_id: int
        :return: Evidence value in [0, 1]
        :rtype: float
        """
        base = float(self.truths[claim_id])
        noise = self.rng.gauss(0, self.noise[MemoryType.OBSERVE])
        return clamp(base + noise)

    def _generate_verification_evidence(self, claim_id: int) -> float:
        """
        Generate truth-grounded verification evidence for a claim.

        :param claim_id: The ID of the claim being verified
        :type claim_id: int
        :return: Evidence value in [0, 1]
        :rtype: float
        """
        base = float(self.truths[claim_id])
        noise = self.rng.gauss(0, self.noise[MemoryType.VERIFY])
        return clamp(base + noise)

    def _generate_heard_evidence(self, sender_id: int, claim_id: int) -> float:
        """
        Generate social (heard) evidence from a sender's current belief.

        Heard evidence is grounded in the sender's belief plus channel noise,
        not in objective truth.

        :param sender_id: The ID of the agent the evidence originates from
        :type sender_id: int
        :param claim_id: The ID of the claim being communicated
        :type claim_id: int
        :return: Evidence value in [0, 1]
        :rtype: float
        """
        base = self.get_agent(sender_id).beliefs[claim_id]
        noise = self.rng.gauss(0, self.noise[MemoryType.HEAR])
        return clamp(base + noise)

    def _generate_observation_events(self) -> list[ObservationEvent]:
        """
        Generate this tick's passive observation events.

        Two kinds of events are emitted:

        - Private events preserve the old per-agent observation behavior: each
          agent independently has a ``private_event_rate`` chance of an
          observation opportunity visible only to them.
        - Global events represent shared world incidents: with a per-tick
          ``global_event_rate`` chance, the world emits one observation event
          visible to every agent.

        :param self: The world instance
        :type self: World
        :return: The observation events emitted this tick
        :rtype: list[ObservationEvent]
        """
        events: list[ObservationEvent] = []

        # Private observation events: one-agent visibility.
        for agent in self.agents:
            if self.rng.random() >= self.private_event_rate:
                continue

            claim_id = self.rng.choice(self.claims)
            evidence = self._generate_observation_evidence(claim_id)

            events.append(
                ObservationEvent(
                    id=self._next_event_id,
                    tick=self.tick,
                    claim_id=claim_id,
                    evidence=evidence,
                    visible_agent_ids=(agent.id,),
                )
            )
            self._next_event_id += 1

        # Global observation event: all-agent visibility.
        if self.rng.random() < self.global_event_rate:
            claim_id = self.rng.choice(self.claims)
            evidence = self._generate_observation_evidence(claim_id)

            events.append(
                ObservationEvent(
                    id=self._next_event_id,
                    tick=self.tick,
                    claim_id=claim_id,
                    evidence=evidence,
                    visible_agent_ids=tuple(agent.id for agent in self.agents),
                )
            )
            self._next_event_id += 1

        return events

    def _deliver_observation_events(self, events: list[ObservationEvent]) -> list[int]:
        """
        Deliver observation events to their visible agents.

        Each visible agent may notice the event (per its attention) and, if so,
        encodes it into a subjective OBSERVE memory.

        :param self: The world instance
        :type self: World
        :param events: The observation events to deliver
        :type events: list[ObservationEvent]
        :return: List of agent IDs that formed an observation memory
        :rtype: list[int]
        """
        observed_ids: list[int] = []

        for event in events:
            for agent_id in event.visible_agent_ids:
                agent = self.get_agent(agent_id)
                if not agent.notices_observation(event):
                    continue
                evidence = agent.encode_observation(event)
                agent._add_memory(
                    tick=self.tick,
                    memory_type=MemoryType.OBSERVE,
                    claim_id=event.claim_id,
                    evidence=evidence,
                )
                observed_ids.append(agent.id)

        return observed_ids

    def _deliver_communicate(self, sender_id: int, receiver_id: int, claim_id: int):
        """
        Handle the reception of a communicated claim from one agent to another,
        allowing the receiving agent to receive social evidence based on the sending
        agent's beliefs and some noise.

        :param self: The world instance
        :type self: World
        :param sender_id: The ID of the agent sending the communication
        :type sender_id: int
        :param receiver_id: The ID of the agent receiving the communication
        :type receiver_id: int
        :param claim_id: The ID of the claim being communicated
        :type sender_id: int
        :type receiver_id: int
        :type claim_id: int
        """
        evidence = self._generate_heard_evidence(sender_id, claim_id)
        self.get_agent(receiver_id)._add_memory(
            tick=self.tick,
            memory_type=MemoryType.HEAR,
            source=sender_id,
            claim_id=claim_id,
            evidence=evidence,
        )

    def _deliver_broadcast(self, sender_id: int, claim_id: int):
        """
        Handle the reception of a broadcasted claim from an agent, allowing connected
        agents to receive social evidence based on the broadcasting agent's beliefs and
        some noise.

        :param self: The world instance
        :type self: World
        :param sender_id: The ID of the agent sending the broadcast
        :type sender_id: int
        :param claim_id: The ID of the claim being broadcast
        :type claim_id: int
        """
        for receiver_id in self.network[sender_id]:
            evidence = self._generate_heard_evidence(sender_id, claim_id)
            self.get_agent(receiver_id)._add_memory(
                tick=self.tick,
                memory_type=MemoryType.HEAR,
                source=sender_id,
                claim_id=claim_id,
                evidence=evidence,
            )

    def _execute_action(self, agent: Agent, action: Action) -> None:
        """
        Execute an agent's chosen action in the environment.

        The world owns action execution: it generates environmental evidence,
        delivers social messages, and records the resulting memories on the
        acting or receiving agents. Agents only decide what they want to do.

        :param self: The world instance
        :type self: World
        :param agent: The agent performing the action
        :type agent: Agent
        :param action: The action to execute
        :type action: Action
        """
        match action.type:
            case ActionType.VERIFY:
                evidence = self._generate_verification_evidence(action.claim_id)
                agent._add_memory(
                    tick=self.tick,
                    memory_type=MemoryType.VERIFY,
                    claim_id=action.claim_id,
                    evidence=evidence,
                )
            case ActionType.COMMUNICATE:
                self._deliver_communicate(
                    agent.id, action.target_agent_id, action.claim_id
                )
            case ActionType.BROADCAST:
                self._deliver_broadcast(agent.id, action.claim_id)
            case ActionType.IDLE:
                pass  # Do nothing
            case _:
                raise ValueError(
                    f"Unknown action type: {action.type} for action {action}"
                )

    def step(self) -> Snapshot:
        """
        Advance the simulation by one tick, allowing each agent to perform their
        actions and update their beliefs based on their interactions with the world
        and other agents.

        :param self: The world instance
        :type self: World
        :return: The snapshot of the world after the step
        :rtype: Snapshot
        """
        observation_events = self._generate_observation_events()
        observed_ids = self._deliver_observation_events(observation_events)
        verified_ids: list[int] = []
        communicate_edges: list[tuple[int, int]] = []
        broadcast_edges: list[tuple[int, int]] = []
        agent_updates = 0

        for agent in self.agents:
            action = agent.choose_action(self)
            self._execute_action(agent, action)

            # Track what happened
            match action.type:
                case ActionType.VERIFY:
                    verified_ids.append(agent.id)
                case ActionType.COMMUNICATE:
                    communicate_edges.append((agent.id, action.target_agent_id))
                case ActionType.BROADCAST:
                    broadcast_edges.extend(
                        [(agent.id, rid) for rid in self.network[agent.id]]
                    )

        # Update beliefs for all agents with new memories
        for agent in self.agents:
            agent_updates += agent.update_beliefs()

        # Create full belief snapshot for all agents and all claims
        beliefs = self.get_agent_beliefs_snapshot()

        snapshot = Snapshot(
            tick=self.tick,
            observation_event_count=len(observation_events),
            observed_ids=observed_ids,
            verified_ids=verified_ids,
            communicate_edges=communicate_edges,
            broadcast_edges=broadcast_edges,
            n_agent_updates=agent_updates,
            agent_beliefs=beliefs,
            agent_memory_sizes={agent.id: agent.memory_size for agent in self.agents},
        )

        self.tick += 1
        return snapshot


if __name__ == "__main__":
    from simlab.config import load_config, build_world

    config_path = "configs/default.yaml"
    cfg = load_config(config_path)
    world = build_world(cfg)

    for _ in range(100):
        world.step()
