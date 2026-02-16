from dataclasses import dataclass
from typing import Any, Literal
from collections import defaultdict
import random
import math

def clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))

@dataclass
class Memory:
    id: int
    type: Literal['observe', 'hear', 'verify', 'action']
    timestamp: int
    source: int | None
    claim_id: int | None
    evidence: float | None


class Agent:
    def __init__(self, id: int, rng_seed: int = 0):
        self.id = id
        self.rng = random.Random(rng_seed)
        self.beliefs: defaultdict[int, float] = defaultdict(lambda: self.rng.random()) # Can map claim IDs to Belief object or just single float representing confidence level
        self.trust: defaultdict[int, float] = defaultdict(lambda: 0.5)  # default trust level for other agents
        self.memory: list[Memory] = []
        self._mem_cursor = 0 # Cursor to track which memories have been processed for belief updates
    
    def __repr__(self):        return f"Agent(id={self.id}, beliefs={dict(self.beliefs)}, trust={dict(self.trust)}, memory={len(self.memory)})"


    def _add_memory(self, world: 'World', memory_type: Literal['observe', 'hear', 'verify', 'action'], source: int | None = None, claim_id: int | None = None, evidence: float | None = None):
        memory = Memory(
            id=len(self.memory),
            type=memory_type,
            timestamp=world.tick,
            source=source,
            claim_id=claim_id,
            evidence=evidence
        )
        self.memory.append(memory)

    def observe(self, world: 'World', claim_id: int):
        """
        Observe a claim in the world, generating evidence based on the truth of the claim and some noise.
        
        :param self:
        :param world: The world in which the agent is observing the claim
        :type world: 'World'
        :param claim_id: The ID of the claim being observed
        :type claim_id: int
        """
        base = 1.0 if world.truths[claim_id] else 0.0
        evidence = clamp(base + world.rng.gauss(0.0, world.noise['observe']), 0.0, 1.0)
        self._add_memory(world, 'observe', claim_id=claim_id, evidence=evidence)

    def communicate(self, world: 'World', target_agent_id: int, claim_id: int):
        """
        Communicate a claim to another agent, allowing them to receive social evidence based on the broadcasting agent's beliefs and some noise.

        :param self:
        :param world: The world in which the agent is communicating the claim
        :type world: 'World'
        :param target_agent_id: The ID of the target agent for communication
        :type target_agent_id: int
        :param claim_id: The ID of the claim being communicated
        :type claim_id: int
        """
        world.deliver_communicate(self.id, target_agent_id, claim_id)
        

    def broadcast(self, world: 'World', claim_id: int):
        """
        Broadcast a claim to all connected agents in the social network, allowing them to receive social evidence based on the broadcasting agent's beliefs and some noise.
        
        :param self:
        :param world: The world in which the agent is broadcasting the claim
        :type world: 'World'
        :param claim_id: The ID of the claim being broadcast
        :type claim_id: int
        """
        world.deliver_broadcast(self.id, claim_id)
        
    def verify(self, world: 'World', claim_id: int):
        """
        Verify a claim by directly checking its truth in the world, generating evidence based on the truth of the claim and some noise.

        :param self:
        :param world: The world in which the agent is verifying the claim
        :type world: 'World'
        :param claim_id: The ID of the claim being verified
        :type claim_id: int
        """
        base = 1.0 if world.truths[claim_id] else 0.0
        evidence = clamp(base + world.rng.gauss(0.0, world.noise['verify']), 0.0, 1.0)
        self._add_memory(world, 'verify', claim_id=claim_id, evidence=evidence)

    def action(self, world: 'World', target_agent_id: int | None = None, action: Any = None):
        """
        Perform a generic action in the world, potentially targeting another agent or based on a specific action type.
        
        :param self: 
        :param world: The world in which the agent is performing the action
        :type world: 'World'
        :param target_agent_id: The ID of the target agent for the action, if any
        :type target_agent_id: int | None
        :param action: The specific action being performed
        :type action: Any
        """
        pass

    def update_beliefs(self, eta: float = 0.1, w_observe: float = 0.6, w_hear: float = 0.3, w_verify: float = 1.0):
        """
        Periodically update the agent's beliefs based on accumulated memories.
        
        :param self: 
        """

        while self._mem_cursor < len(self.memory):
            mem = self.memory[self._mem_cursor]

            if mem.claim_id is not None and mem.evidence is not None:

                match mem.type:
                    case 'observe':
                        lr = eta * w_observe
                    case 'verify':
                        lr = eta * w_verify
                    case 'hear':
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
            

class World:
    def __init__(self, agents: list[Agent], truths: dict[int, bool], p_observe: float, p_verify: float, k_interactions: int, rng_seed: int = 0, noise: dict[str, float] | None = None):
        self._agents = {a.id: a for a in agents}
        self.last_step: dict[str, Any] | None = {}
        self.tick = 0
        self.p_observe = p_observe
        self.p_verify = p_verify
        self.k_interactions = k_interactions
        self.rng = random.Random(rng_seed)
        self.noise = {'observe': 0.0, 'hear': 0.0, 'verify': 0.0, 'action': 0.0} | (noise or {})
        self.truths = truths
        self.network = self._generate_dummy_network(agents) # TODO: We can implement a more complex network generation mechanism here, potentially based on real-world social network structures or using a configurable graph model.
        self.subject_claim_id = 0 # for now we just track one claim, but we can easily extend this to multiple claims and track them separately in the logs and metrics.
    
    def __repr__(self):        return f"World(tick={self.tick}, agents={len(self._agents)}, truths={self.truths}, network={dict(self.network)})"

    @property
    def agents(self) -> list[Agent]:
        return list(self._agents.values())
    
    @property
    def edges(self) -> list[tuple[int, int]]:
        return [(src, dest) for src, nei in self.network.items() for dest in nei]
    
    def get_agent(self, agent_id: int) -> Agent:
        return self._agents[agent_id]

    def deliver_communicate(self, sender_id: int, receiver_id: int, claim_id: int):
        """
        Handle the reception of a communicated claim from one agent to another, allowing the receiving agent to receive social evidence based on the sending agent's beliefs and some noise.
        
        :param self: 
        :param sender_id: The ID of the agent sending the communication
        :param receiver_id: The ID of the agent receiving the communication
        :param claim_id: The ID of the claim being communicated
        :type sender_id: int
        :type receiver_id: int
        :type claim_id: int
        """
        evidence = self.heard_evidence(sender_id, claim_id)
        self._agents[receiver_id]._add_memory(self, 'hear', source=sender_id, claim_id=claim_id, evidence=evidence)
    
    def deliver_broadcast(self, sender_id: int, claim_id: int):
        """
        Handle the reception of a broadcasted claim from an agent, allowing connected agents to receive social evidence based on the broadcasting agent's beliefs and some noise.
        
        :param self: 
        :param sender_id: The ID of the agent sending the broadcast
        :param claim_id: The ID of the claim being broadcast
        :type sender_id: int
        :type claim_id: int
        """
        for receiver_id in self.network[sender_id]:
            evidence = self.heard_evidence(sender_id, claim_id)
            self._agents[receiver_id]._add_memory(self, 'hear', source=sender_id, claim_id=claim_id, evidence=evidence)

    def heard_evidence(self, sender_id: int, claim_id: int) -> float:
        """
        Sample social evidence for a claim from a specific sender.
        
        :param self: 
        :param sender_id: The ID of the agent sending the evidence
        :param claim_id: The ID of the claim being evaluated
        :type sender_id: int
        :type claim_id: int
        """
        base = self._agents[sender_id].beliefs[claim_id]
        return clamp(base + self.rng.gauss(0, self.noise['hear']), 0.0, 1.0)
    
    def event_generator(self):
        """
        Generate events in the world that agents can observe, verify, or communicate about. This can be extended to include more complex event types, dependencies between events, and temporal dynamics.
        
        :param self:
        """
        pass
    
    def step(self):
        """
        Advance the simulation by one tick, allowing each agent to perform their actions and update their beliefs based on their interactions with the world and other agents.
        
        :param self:
        """
        # TODO: We can implement a more complex event generation mechanism here, potentially based on real-world news cycles or using a configurable event model. For now, we will just randomly select some agents to observe and verify the claim, and randomly select some interactions for communication.

        belief_before = {aid: a.beliefs[self.subject_claim_id] for aid, a in self._agents.items()}

        self.last_step = {
            "tick": self.tick,
            "claim_id": self.subject_claim_id,
            "n_updates": 0,
            "observed_ids": [],
            "verified_ids": [],
            "heard_edges": [],
            "belief_before": belief_before,
        }

        relationships = [(k, entry) for k, v in self.network.items() for entry in v]
        interactions = self.rng.sample(relationships, k=min(self.k_interactions, len(relationships)))
        observers = [a for a in self._agents.values() if self.rng.random() < self.p_observe]
        verifiers = [a for a in self._agents.values() if self.rng.random() < self.p_verify]
        updated_agents = set()

        for observer in observers:
            claim_id = self.rng.choice(list(self.truths.keys()))
            observer.observe(self, claim_id)
            self.last_step["observed_ids"].append(observer.id)
            updated_agents.add(observer.id)
        
        for verifier in verifiers:
            claim_id = self.rng.choice(list(self.truths.keys()))
            verifier.verify(self, claim_id)
            self.last_step["verified_ids"].append(verifier.id)
            updated_agents.add(verifier.id)

        for sender_id, receiver_id in interactions:
            claim_id = self.rng.choice(list(self.truths.keys()))
            self.deliver_communicate(sender_id, receiver_id, claim_id)
            self.last_step["heard_edges"].append((sender_id, receiver_id, claim_id))
            updated_agents.add(receiver_id)

        for agent_id in updated_agents:
            self._agents[agent_id].update_beliefs()
            self.last_step["n_updates"] += 1

        belief_after = {aid: a.beliefs[self.subject_claim_id] for aid, a in self._agents.items()}
        self.last_step["belief_after"] = belief_after

        self.tick += 1
        if self.tick % 10 == 0:
            self.log_step()

    
    def _generate_dummy_network(self, agents: list[Agent]) -> dict[int, list[int]]:
        network = defaultdict(list)
        max_degree = min(4, len(agents) - 1)
        for agent in agents:
            connections = self.rng.sample(
                [a.id for a in agents if a.id != agent.id],
                k=self.rng.randint(1, max_degree)
            )
            network[agent.id] = connections
        return network
    

    def log_step(self):
        if not self.last_step:
            print("No step data yet.")
            return

        claim_id = self.last_step["claim_id"]
        before = self.last_step["belief_before"]
        after = self.last_step.get("belief_after", {aid: a.beliefs[claim_id] for aid, a in self._agents.items()})

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

        q10, q25, q50, q75, q90 = q(0.10), q(0.25), q(0.50), q(0.75), q(0.90)

        low = sum(x < 0.2 for x in vals) / n
        high = sum(x > 0.8 for x in vals) / n

        # movement metrics
        deltas = [abs(after[aid] - before[aid]) for aid in after.keys()]
        mean_abs_delta = sum(deltas) / n
        max_abs_delta = max(deltas)

        # who moved most?
        top_movers = sorted(((aid, after[aid] - before[aid]) for aid in after.keys()),
                            key=lambda t: abs(t[1]), reverse=True)[:3]

        # simple “influence suspects”: highest out-degree in network
        degrees = [(aid, len(self.network.get(aid, []))) for aid in self._agents.keys()]
        top_degree = sorted(degrees, key=lambda t: t[1], reverse=True)[:3]

        print(
            f"Tick {self.last_step['tick']:4d} | claim {claim_id} | "
            f"belief mean={mean:.3f} std={std:.3f} min={mn:.3f} max={mx:.3f} | "
            f"q10={q10:.3f} q50={q50:.3f} q90={q90:.3f} | "
            f"<0.2={low:.2f} >0.8={high:.2f} | "
            f"Δabs_mean={mean_abs_delta:.4f} Δmax={max_abs_delta:.4f} | "
            f"events: hear={len(self.last_step['heard_edges'])} obs={len(self.last_step['observed_ids'])} ver={len(self.last_step['verified_ids'])} updates={self.last_step['n_updates']}"
        )

        # optional second line with “who changed / who’s connected”
        print(f"  top movers (aid, Δbelief): {[(aid, round(db,4)) for aid, db in top_movers]}")
        print(f"  top degree (aid, out_degree): {top_degree}")



def init_world(num_agents: int = 50, rng_seed: int = 0) -> World:
    agents = [Agent(id=i, rng_seed=rng_seed + i + 1) for i in range(num_agents)] # add i to differ seed, and 1 to offset from world rng

    truths = {0: True}  # single claim for POC

    noise = {
        "observe": 0.25,
        "hear": 0.15,
        "verify": 0.05,
    }

    return World(
        agents=agents,
        truths=truths,
        p_observe=0.05,
        p_verify=0.01,
        k_interactions=num_agents * 2,
        rng_seed=rng_seed,
        noise=noise,
    )


if __name__ == "__main__":
    world = init_world(num_agents=10, rng_seed=42)
    for _ in range(100):
        world.step()