from __future__ import annotations

import random
from collections import defaultdict

from simlab._merge import deep_merge
from simlab.agent import Agent
from simlab.config_schema import WorldSection
from simlab.kernel_types import (
    Action,
    ActionTrace,
    ActionType,
    MemoryType,
    ObservationEvent,
    Snapshot,
    clamp,
)

_DEFAULT_WORLD_NOISE: dict = WorldSection.model_fields["noise"].get_default(
    call_default_factory=True
)


class World:
    def __init__(self, agents: list[Agent], settings: WorldSection):
        self._agents = {a.id: a for a in agents}
        self.tick = 0
        self.rng = random.Random(settings.rng_seed)
        self.noise = {MemoryType[k]: v for k, v in settings.noise.items()}
        self.truths = dict(settings.truths)
        # private_event_rate: per-agent per-tick chance of a private observation
        #   event visible only to that agent.
        # global_event_rate: per-tick chance of one shared observation event
        #   visible to all agents.
        self.private_event_rate = settings.observation.private_event_rate
        self.global_event_rate = settings.observation.global_event_rate
        self._next_event_id = 0
        self.network = self._generate_dummy_network(
            # TODO: We can implement a more complex network generation mechanism here,
            # potentially based on real-world social network structures or using a
            # configurable graph model.
            agents
        )

    @classmethod
    def from_dict(cls, agents: list[Agent], raw: dict) -> World:
        """Ad-hoc/direct construction entry point: ``raw`` is a possibly
        partial world settings dict, e.g. ``{"truths": {0: True}, "rng_seed": 1}``.

        ``truths``/``rng_seed`` have no schema default, so must be present
        in ``raw``. ``noise`` is deep-merged onto its default first (same
        dict-typed-field caveat as ``Agent.from_dict``); ``observation`` is
        a nested section, which pydantic already fills in per-field.

        :param agents: The world's agents
        :type agents: list[Agent]
        :param raw: A possibly partial world settings dict
        :type raw: dict
        :return: The constructed world
        :rtype: World
        """
        merged = deep_merge({"noise": _DEFAULT_WORLD_NOISE}, raw)
        return cls(agents=agents, settings=WorldSection.model_validate(merged))

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

    def neighbors(self, agent_id: int) -> list[int]:
        """Return outgoing neighbor IDs for an agent."""
        return self.network[agent_id]

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
        for receiver_id in self.neighbors(sender_id):
            evidence = self._generate_heard_evidence(sender_id, claim_id)
            self.get_agent(receiver_id)._add_memory(
                tick=self.tick,
                memory_type=MemoryType.HEAR,
                source=sender_id,
                claim_id=claim_id,
                evidence=evidence,
            )

    def _execute_action(self, agent: Agent, action: Action) -> ActionTrace:
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
        :return: A trace of what the action caused
        :rtype: ActionTrace
        """
        trace = ActionTrace()

        match action.type:
            case ActionType.VERIFY:
                evidence = self._generate_verification_evidence(action.claim_id)
                agent._add_memory(
                    tick=self.tick,
                    memory_type=MemoryType.VERIFY,
                    claim_id=action.claim_id,
                    evidence=evidence,
                )
                trace.verified_ids.append(agent.id)
            case ActionType.COMMUNICATE:
                self._deliver_communicate(
                    agent.id, action.target_agent_id, action.claim_id
                )
                trace.communicate_edges.append((agent.id, action.target_agent_id))
            case ActionType.BROADCAST:
                self._deliver_broadcast(agent.id, action.claim_id)
                trace.broadcast_edges.extend(
                    (agent.id, receiver_id) for receiver_id in self.neighbors(agent.id)
                )
            case ActionType.IDLE:
                pass
            case _:
                raise ValueError(
                    f"Unknown action type: {action.type} for action {action}"
                )

        return trace

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
        num_memory_processing_agents = 0
        num_belief_updating_agents = 0
        num_trust_updating_agents = 0

        for agent in self.agents:
            action = agent.choose_action(self)
            trace = self._execute_action(agent, action)

            verified_ids.extend(trace.verified_ids)
            communicate_edges.extend(trace.communicate_edges)
            broadcast_edges.extend(trace.broadcast_edges)

        # Update beliefs for all agents with new memories
        for agent in self.agents:
            update_trace = agent.update_beliefs()
            num_memory_processing_agents += int(update_trace.processed_memory)
            num_belief_updating_agents += int(update_trace.belief_changed)
            num_trust_updating_agents += int(update_trace.trust_changed)

        # Create full belief snapshot for all agents and all claims
        beliefs = self.get_agent_beliefs_snapshot()

        snapshot = Snapshot(
            tick=self.tick,
            observation_event_count=len(observation_events),
            observed_ids=observed_ids,
            verified_ids=verified_ids,
            communicate_edges=communicate_edges,
            broadcast_edges=broadcast_edges,
            num_memory_processing_agents=num_memory_processing_agents,
            num_belief_updating_agents=num_belief_updating_agents,
            num_trust_updating_agents=num_trust_updating_agents,
            agent_beliefs=beliefs,
            agent_memory_sizes={agent.id: agent.memory_size for agent in self.agents},
        )

        self.tick += 1
        return snapshot
