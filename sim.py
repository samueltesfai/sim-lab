from dataclasses import dataclass
from typing import Any
from collections import defaultdict
from enum import Enum
import random
import math


def clamp(value, min_value, max_value):
    """Because I'm too lazy to import numpy"""
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


class Agent:
    def __init__(
        self,
        id: int,
        rng_seed: int = 0,
        action_preference: dict[ActionType, float] | None = None,
        action_cost: dict[ActionType, float] | None = None,
    ):
        self.id = id
        self.rng = random.Random(rng_seed)
        self.beliefs: defaultdict[int, float] = defaultdict(lambda: self.rng.random())
        self.trust: defaultdict[int, float] = defaultdict(lambda: 0.5)
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
        return f"Agent(id={self.id}, beliefs={dict(self.beliefs)}, trust={dict(self.trust)}, memory={len(self.memory)})"

    def add_memory(
        self,
        world: "World",
        memory_type: MemoryType,
        source: int | None = None,
        claim_id: int | None = None,
        evidence: float | None = None,
    ):
        """
        Add a memory to the agent's memory list.

        :param self:
        :param world: The world in which the agent is adding the memory
        :type world: 'World'
        :param memory_type: The type of memory being added
        :type memory_type: MemoryType
        :param source: The ID of the source agent for the memory (if applicable)
        :type source: int | None
        :param claim_id: The ID of the claim for the memory (if applicable)
        :type claim_id: int | None
        :param evidence: The evidence value for the memory (if applicable)
        :type evidence: float | None
        """

        if evidence is None:
            match memory_type:
                case MemoryType.OBSERVE:
                    base = float(world.truths[claim_id])
                    noise = world.rng.gauss(0, world.noise[MemoryType.OBSERVE])
                case MemoryType.HEAR:
                    base = world.get_agent(source).beliefs[claim_id]
                    noise = world.rng.gauss(0, world.noise[MemoryType.HEAR])
                case MemoryType.VERIFY:
                    base = float(world.truths[claim_id])
                    noise = world.rng.gauss(0, world.noise[MemoryType.VERIFY])
                case _:
                    base = 0.5
                    noise = 0.0
            evidence = clamp(base + noise, 0.0, 1.0)

        memory = Memory(
            id=len(self.memory),
            type=memory_type,
            timestamp=world.tick,
            source=source,
            claim_id=claim_id,
            evidence=evidence,
        )
        self.memory.append(memory)

    def _communicate(self, world: "World", target_agent_id: int, claim_id: int):
        """
        Communicate a claim to another agent, allowing them to receive social evidence
        based on the broadcasting agent's beliefs and some noise.

        :param self:
        :param world: The world in which the agent is communicating the claim
        :type world: 'World'
        :param target_agent_id: The ID of the target agent for communication
        :type target_agent_id: int
        :param claim_id: The ID of the claim being communicated
        :type claim_id: int
        """
        world.deliver_communicate(self.id, target_agent_id, claim_id)

    def _broadcast(self, world: "World", claim_id: int):
        """
        Broadcast a claim to all connected agents in the social network, allowing them
        to receive social evidence based on the broadcasting agent's beliefs and some
        noise.

        :param self:
        :param world: The world in which the agent is broadcasting the claim
        :type world: 'World'
        :param claim_id: The ID of the claim being broadcast
        :type claim_id: int
        """
        world.deliver_broadcast(self.id, claim_id)

    def _verify(self, world: "World", claim_id: int):
        """
        Verify a claim by directly checking its truth in the world, generating evidence
        based on the truth of the claim and some noise.

        :param self:
        :param world: The world in which the agent is verifying the claim
        :type world: 'World'
        :param claim_id: The ID of the claim being verified
        :type claim_id: int
        """
        self.add_memory(world, MemoryType.VERIFY, claim_id=claim_id)

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

    def get_action_preference(self, world: "World", action: Action) -> float:
        """
        Get the preference for a given action.

        :param world: The world in which the agent is generating candidate actions
        :type world: 'World'
        :param action: The action
        :type action: Action
        :return: The preference for the action
        :rtype: float
        """
        return self.action_preference[action.type]

    def get_action_cost(self, world: "World", action: Action) -> float:
        """
        Get the cost for a given action.

        :param world: The world in which the agent is generating candidate actions
        :type world: 'World'
        :param action: The action
        :type action: Action
        :return: The cost for the action
        :rtype: float
        """
        return self.action_cost[action.type]

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
                return self.get_action_preference(world, action) * self.uncertainty(
                    action.claim_id
                ) - self.get_action_cost(world, action)
            case ActionType.COMMUNICATE:
                return self.get_action_preference(world, action) * self.confidence(
                    action.claim_id
                ) * self.disagreement(
                    action.claim_id, action.target_agent_id, world
                ) - self.get_action_cost(world, action)
            case ActionType.BROADCAST:
                return self.get_action_preference(world, action) * self.confidence(
                    action.claim_id
                ) * self.local_disagreement(
                    action.claim_id, world
                ) - self.get_action_cost(world, action)
            case ActionType.IDLE:
                return self.get_action_preference(world, action) - self.get_action_cost(
                    world, action
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

    def act(self, world: "World", action: Action):
        """
        Execute the chosen action using a dispatcher pattern.

        :param self:
        :param world: The world in which the agent is performing the action
        :type world: 'World'
        :param action: The action to perform
        :type action: Action
        """
        match action.type:
            case ActionType.VERIFY:
                self._verify(world, action.claim_id)
            case ActionType.COMMUNICATE:
                self._communicate(world, action.target_agent_id, action.claim_id)
            case ActionType.BROADCAST:
                self._broadcast(world, action.claim_id)
            case ActionType.IDLE:
                pass  # Do nothing
            case _:
                raise ValueError(
                    f"Unknown action type: {action.type} for action {action}"
                )

    def update_beliefs(
        self,
        eta: float = 0.1,
        w_observe: float = 0.6,
        w_hear: float = 0.3,
        w_verify: float = 1.0,
    ) -> bool:
        """
        Update the agent's beliefs based on accumulated memories.

        :param self:
        :param eta: Learning rate
        :type eta: float
        :param w_observe: Weight for observation memories
        :type w_observe: float
        :param w_hear: Weight for hear memories
        :type w_hear: float
        :param w_verify: Weight for verify memories
        :type w_verify: float
        :return: True if any beliefs were updated, False otherwise
        :rtype: bool
        """
        updated = False
        while self._mem_cursor < len(self.memory):
            updated = True
            mem = self.memory[self._mem_cursor]

            if mem.claim_id is not None and mem.evidence is not None:
                match mem.type:
                    case MemoryType.OBSERVE:
                        lr = eta * w_observe
                    case MemoryType.VERIFY:
                        lr = eta * w_verify
                    case MemoryType.HEAR:
                        if mem.source is None:
                            lr = 0.0
                        else:
                            lr = eta * w_hear * self.trust[mem.source]
                    case _:
                        lr = 0.0

                lr = clamp(lr, 0.0, 1.0)

                b = self.beliefs[mem.claim_id]
                b_new = b + lr * (mem.evidence - b)
                self.beliefs[mem.claim_id] = clamp(b_new, 0.0, 1.0)

            self._mem_cursor += 1

        return updated


class World:
    def __init__(
        self,
        agents: list[Agent],
        truths: dict[int, bool],
        rng_seed: int = 0,
        noise: dict[MemoryType, float] | None = None,
    ):
        self._agents = {a.id: a for a in agents}
        self.last_step: dict[str, Any] | None = {}
        self.tick = 0
        self.rng = random.Random(rng_seed)
        self.noise = {
            MemoryType.OBSERVE: 0.0,
            MemoryType.HEAR: 0.0,
            MemoryType.VERIFY: 0.0,
        } | (noise or {})
        self.truths = truths
        self.network = self._generate_dummy_network(
            # TODO: We can implement a more complex network generation mechanism here,
            # potentially based on real-world social network structures or using a
            # configurable graph model.
            agents
        )
        # For now we just track one claim, but we can easily extend this to multiple
        # claims and track them separately in the logs and metrics.
        self.subject_claim_id = 0

    def _generate_dummy_network(self, agents: list[Agent]) -> dict[int, list[int]]:
        network = defaultdict(list)
        max_degree = min(4, len(agents) - 1)
        for agent in agents:
            connections = self.rng.sample(
                [a.id for a in agents if a.id != agent.id],
                k=self.rng.randint(1, max_degree),
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

    def get_agent(self, agent_id: int) -> Agent:
        return self._agents[agent_id]

    def deliver_observation(self) -> list[int]:
        """
        Deliver observations to agents in the world.

        :param self: The world instance
        :type self: World
        :return: List of agent IDs that received observations
        :rtype: list[int]
        """
        observed_agents = []

        for agent in self.agents:
            if self.rng.random() < 0.1:  # 10% observation probability per agent
                claim_id = self.rng.choice(self.claims)
                agent.add_memory(self, MemoryType.OBSERVE, claim_id=claim_id)
                observed_agents.append(agent.id)

        return observed_agents

    def deliver_communicate(self, sender_id: int, receiver_id: int, claim_id: int):
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
        self.get_agent(receiver_id).add_memory(
            self,
            MemoryType.HEAR,
            source=sender_id,
            claim_id=claim_id,
        )

    def deliver_broadcast(self, sender_id: int, claim_id: int):
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
            self.get_agent(receiver_id).add_memory(
                self,
                MemoryType.HEAR,
                source=sender_id,
                claim_id=claim_id,
            )

    def step(self):
        """
        Advance the simulation by one tick, allowing each agent to perform their
        actions and update their beliefs based on their interactions with the world
        and other agents.

        :param self: The world instance
        :type self: World
        """
        belief_before = {
            aid: a.beliefs[self.subject_claim_id] for aid, a in self._agents.items()
        }

        self.last_step = {
            "tick": self.tick,
            "claim_id": self.subject_claim_id,
            "agent_updates": 0,
            "observed_ids": [],
            "verified_ids": [],
            "communicate_edges": [],
            "broadcast_edges": [],
            "belief_before": belief_before,
        }

        self.last_step["observed_ids"] = self.deliver_observation()

        for agent in self.agents:
            action = agent.choose_action(self)
            agent.act(self, action)

            # Track what happened
            match action.type:
                case ActionType.VERIFY:
                    self.last_step["verified_ids"].append(agent.id)
                case ActionType.COMMUNICATE:
                    self.last_step["communicate_edges"].append(
                        (agent.id, action.target_agent_id)
                    )
                case ActionType.BROADCAST:
                    self.last_step["broadcast_edges"].extend(
                        [(agent.id, rid) for rid in self.network[agent.id]]
                    )

        # Update beliefs for all agents with new memories
        for agent in self.agents:
            self.last_step["agent_updates"] += agent.update_beliefs()

        belief_after = {
            aid: a.beliefs[self.subject_claim_id] for aid, a in self._agents.items()
        }
        self.last_step["belief_after"] = belief_after

        self.tick += 1
        if self.tick % 10 == 0:
            self.log_step()

    def log_step(self):
        if not self.last_step:
            print("No step data yet.")
            return

        claim_id = self.last_step["claim_id"]
        before = self.last_step["belief_before"]
        after = self.last_step.get(
            "belief_after",
            {aid: a.beliefs[claim_id] for aid, a in self._agents.items()},
        )

        vals = list(after.values())
        n = len(vals)
        if n == 0:
            print(f"Tick {self.tick}: no agents")
            return

        vals_sorted = sorted(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(var)
        mn, mx = vals_sorted[0], vals_sorted[-1]

        def q(p: float) -> float:
            # simple nearest-rank quantile
            idx = int(round(p * (n - 1)))
            return vals_sorted[idx]

        q10, q50, q90 = q(0.10), q(0.50), q(0.90)

        low = sum(x < 0.2 for x in vals) / n
        high = sum(x > 0.8 for x in vals) / n

        # movement metrics
        deltas = [abs(after[aid] - before[aid]) for aid in after.keys()]
        mean_abs_delta = sum(deltas) / n
        max_abs_delta = max(deltas)

        # who moved most?
        top_movers = sorted(
            ((aid, after[aid] - before[aid]) for aid in after.keys()),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:3]

        # simple “influence suspects”: highest out-degree in network
        degrees = [(aid, len(self.network.get(aid, []))) for aid in self._agents.keys()]
        top_degree = sorted(degrees, key=lambda t: t[1], reverse=True)[:3]

        print(
            f"Tick {self.last_step['tick']:4d} | claim {claim_id} | "
            f"belief mean={mean:.3f} std={std:.3f} min={mn:.3f} max={mx:.3f} | "
            f"q10={q10:.3f} q50={q50:.3f} q90={q90:.3f} | "
            f"<0.2={low:.2f} >0.8={high:.2f} | "
            f"Δabs_mean={mean_abs_delta:.4f} Δmax={max_abs_delta:.4f} | "
            f"events: com={len(self.last_step['communicate_edges'])} | "
            f"bcast={len(self.last_step['broadcast_edges'])} | "
            f"obs={len(self.last_step['observed_ids'])} | "
            f"ver={len(self.last_step['verified_ids'])} | "
            f"updates={self.last_step['agent_updates']}"
        )

        # optional second line with “who changed / who’s connected”
        print(
            f"  top movers (aid, Δbelief): {[(aid, round(db, 4)) for aid, db in top_movers]}"
        )
        print(f"  top degree (aid, out_degree): {top_degree}")


def init_world(num_agents: int = 50, rng_seed: int = 0) -> World:
    agents = [
        Agent(id=i, rng_seed=rng_seed + i + 1) for i in range(num_agents)
    ]  # add i to differ seed, and 1 to offset from world rng

    truths = {0: True}  # single claim for POC

    noise = {
        MemoryType.OBSERVE: 0.1,
        MemoryType.HEAR: 0.15,
        MemoryType.VERIFY: 0.05,
    }

    return World(
        agents=agents,
        truths=truths,
        rng_seed=rng_seed,
        noise=noise,
    )


if __name__ == "__main__":
    world = init_world(num_agents=10, rng_seed=42)
    for _ in range(100):
        world.step()
