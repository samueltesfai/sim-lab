from __future__ import annotations

from collections import defaultdict
import random
from typing import TYPE_CHECKING

from simlab.types import (
    Action,
    ActionType,
    AgentUpdateTrace,
    Memory,
    MemoryProcessTrace,
    MemoryType,
    ObservationEvent,
    clamp,
)

# Avoid circular import: World is defined in world.py and referenced via string
# annotations above.
if TYPE_CHECKING:
    from simlab.world import World


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

    def disagreement(self, claim_id: int, agent_id: int, world: World) -> float:
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

    def local_disagreement(self, claim_id: int, world: World) -> float:
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
        neighbors = world.neighbors(self.id)
        n_neighbors = len(neighbors)
        if n_neighbors == 0:
            return 0.0
        return (
            sum(
                self.disagreement(claim_id, neighbor_id, world)
                for neighbor_id in neighbors
            )
            / n_neighbors
        )

    def generate_candidate_actions(self, world: World) -> list[Action]:
        """
        Generate a list of candidate actions for the agent to choose from.

        :param self:
        :param world: The world in which the agent is generating candidate actions
        :type world: 'World'
        :return: A list of candidate actions
        :rtype: list[Action]
        """
        candidates = [Action(ActionType.IDLE)]

        for claim_id in world.claims:
            candidates.append(Action(ActionType.VERIFY, claim_id=claim_id))
            candidates.append(Action(ActionType.BROADCAST, claim_id=claim_id))

            for neighbor_id in world.neighbors(self.id):
                candidates.append(
                    Action(
                        ActionType.COMMUNICATE,
                        claim_id=claim_id,
                        target_agent_id=neighbor_id,
                    )
                )

        return candidates

    def score_action(self, world: World, action: Action) -> float:
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

    def choose_action(self, world: World) -> Action:
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

    def update_beliefs(self) -> AgentUpdateTrace:
        """
        Update the agent's beliefs based on accumulated memories.

        The effective learning rate for each memory is the agent's global
        plasticity (``learning_rate``) scaled by a channel-specific weight, and
        further modulated by trust for socially heard memories. HEAR memories
        are additionally subject to bounded confidence and dynamic trust updates.

        All newly accumulated memories are consumed regardless of the returned
        trace; trust may still change for HEAR memories even when no belief
        moves, and rejected HEAR memories may still update trust without ever
        moving belief. These are tracked as distinct outcomes because they
        measure different simulation phenomena.

        :param self:
        :return: A trace of which agent-level effects occurred this tick
        :rtype: AgentUpdateTrace
        """
        trace = AgentUpdateTrace()
        while self._mem_cursor < len(self.memory):
            trace.processed_memory = True
            mem_trace = self._process_memory(self.memory[self._mem_cursor])
            trace.belief_changed |= mem_trace.belief_changed
            trace.trust_changed |= mem_trace.trust_changed
            self._mem_cursor += 1
        return trace

    def _process_memory(self, mem: Memory) -> MemoryProcessTrace:
        """
        Apply a single memory to beliefs, with trust side-effects for HEAR.

        :param mem: The memory to process
        :type mem: Memory
        :return: A trace of which effects this memory caused
        :rtype: MemoryProcessTrace
        """
        if mem.claim_id is None or mem.evidence is None:
            return MemoryProcessTrace()

        belief_before = self.beliefs[mem.claim_id]

        accepted = (
            self._should_accept_heard_memory(mem, belief_before)
            if mem.type == MemoryType.HEAR
            else None
        )

        lr = clamp(self._effective_learning_rate(mem, belief_before, accepted))
        belief_after = clamp(belief_before + lr * (mem.evidence - belief_before))
        self.beliefs[mem.claim_id] = belief_after

        trust_changed = False
        if mem.type == MemoryType.HEAR:
            trust_changed = self._update_trust_from_heard_memory(
                mem, belief_before, accepted
            )

        return MemoryProcessTrace(
            belief_changed=belief_after != belief_before,
            trust_changed=trust_changed,
        )

    def _effective_learning_rate(
        self, mem: Memory, belief_before: float, accepted: bool | None
    ) -> float:
        """
        Compute the effective learning rate for a memory before clamping.

        For HEAR memories, returns 0.0 when ``accepted`` is False, i.e. the
        heard evidence fell outside the agent's social confidence bound.

        :param mem: The memory being processed
        :type mem: Memory
        :param belief_before: The agent's belief for the claim before this update
        :type belief_before: float
        :param accepted: For HEAR memories, whether bounded confidence accepted
            the evidence (see ``_should_accept_heard_memory``); unused otherwise.
        :type accepted: bool | None
        :return: Raw (unclamped) effective learning rate
        :rtype: float
        """
        match mem.type:
            case MemoryType.OBSERVE:
                return self.learning_rate * self.observe_weight
            case MemoryType.VERIFY:
                return self.learning_rate * self.verify_weight
            case MemoryType.HEAR:
                if mem.source is None or not accepted:
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
    ) -> bool:
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
        :return: True if the trust value for the memory's source changed
        :rtype: bool
        """
        if self.social_trust_update_rate == 0.0:
            return False
        if not accepted and not self.social_update_trust_on_rejection:
            return False
        if mem.source is None:
            return False
        trust_before = self.trust[mem.source]
        agreement = 1.0 - abs(mem.evidence - belief_before)
        trust_delta = self.social_trust_update_rate * (agreement - trust_before)
        trust_after = clamp(trust_before + trust_delta)
        self.trust[mem.source] = trust_after
        return trust_after != trust_before
