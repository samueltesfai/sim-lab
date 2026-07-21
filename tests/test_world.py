import pytest
import sys
import io

from simlab.agent import Agent
from simlab.world import World
from simlab.types import Action, ActionType, MemoryType, Snapshot


def _build_world(n: int = 5) -> World:
    agents = [Agent(i, rng_seed=i) for i in range(n)]
    return World(agents=agents, truths={0: True}, rng_seed=1)


# ---------------------------------------------------------------------------
# Initialization and properties
# ---------------------------------------------------------------------------


def test_world_initialization():
    """Test World initialization."""
    agents = [Agent(i, rng_seed=i) for i in range(3)]
    world = World(
        agents=agents,
        truths={0: True, 1: False},
        rng_seed=42,
        noise={MemoryType.OBSERVE: 0.1, MemoryType.HEAR: 0.2},
        private_event_rate=0.3,
        global_event_rate=0.2,
    )

    assert len(world.agents) == 3
    assert world.truths == {0: True, 1: False}
    assert world.tick == 0
    assert world.private_event_rate == 0.3
    assert world.global_event_rate == 0.2
    assert world.noise[MemoryType.OBSERVE] == 0.1
    assert world.noise[MemoryType.HEAR] == 0.2
    assert world.noise[MemoryType.VERIFY] == 0.0

    assert isinstance(world.network, dict)
    assert len(world.network) == 3
    for agent_id, connections in world.network.items():
        assert isinstance(connections, list)
        assert agent_id not in connections


def test_world_properties():
    """Test World properties."""
    world = _build_world(3)

    agents = world.agents
    assert len(agents) == 3
    assert all(isinstance(a, Agent) for a in agents)

    assert world.claims == [0]

    edges = world.edges
    assert isinstance(edges, list)
    for edge in edges:
        assert isinstance(edge, tuple) and len(edge) == 2


def test_world_get_agent():
    """Test World.get_agent method."""
    world = _build_world(3)

    agent = world.get_agent(0)
    assert isinstance(agent, Agent)
    assert agent.id == 0

    with pytest.raises(KeyError):
        world.get_agent(999)


def test_world_get_agent_beliefs_snapshot():
    """Test World.get_agent_beliefs_snapshot method."""
    world = _build_world(2)

    snapshot = world.get_agent_beliefs_snapshot()

    assert isinstance(snapshot, dict)
    assert len(snapshot) == 2

    for agent_id, claim_beliefs in snapshot.items():
        assert isinstance(agent_id, int)
        assert isinstance(claim_beliefs, dict)
        assert len(claim_beliefs) == len(world.truths)
        for claim_id, belief_value in claim_beliefs.items():
            assert isinstance(claim_id, int)
            assert isinstance(belief_value, float)
            assert 0.0 <= belief_value <= 1.0


def test_get_agent_beliefs_snapshot_materializes_known_claims_from_lazy_beliefs():
    world = _build_world(3)

    for agent in world.agents:
        agent.beliefs.clear()

    snapshot = world.get_agent_beliefs_snapshot()

    assert set(snapshot.keys()) == {agent.id for agent in world.agents}
    for agent in world.agents:
        claim_beliefs = snapshot[agent.id]
        assert set(claim_beliefs.keys()) == set(world.truths.keys())
        for claim_id in world.truths:
            assert isinstance(claim_beliefs[claim_id], float)


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------


def test_execute_action_idle_is_noop():
    """World._execute_action IDLE leaves agent memory unchanged."""
    world = _build_world(2)
    agent = world.get_agent(0)

    world._execute_action(agent, Action(ActionType.IDLE))
    assert agent.memory == []


def test_execute_action_verify_records_memory():
    """World._execute_action VERIFY adds a VERIFY memory to the acting agent."""
    world = _build_world(2)
    agent = world.get_agent(0)

    initial = len(agent.memory)
    world._execute_action(agent, Action(ActionType.VERIFY, claim_id=0))

    assert len(agent.memory) == initial + 1
    assert agent.memory[-1].type == MemoryType.VERIFY
    assert agent.memory[-1].claim_id == 0


def test_execute_action_invalid_type_rejected_at_construction():
    """Invalid action types are rejected when building the Action."""
    with pytest.raises(ValueError, match="'INVALID_TYPE' is not a valid ActionType"):
        Action("INVALID_TYPE")


def test_world_deliver_communicate():
    """Executing COMMUNICATE delivers a HEAR memory to the receiver."""
    world = _build_world(2)
    world.network[0] = [1]

    sender = world.get_agent(0)
    receiver = world.get_agent(1)
    initial = len(receiver.memory)

    world._execute_action(
        sender, Action(ActionType.COMMUNICATE, claim_id=0, target_agent_id=1)
    )

    assert len(receiver.memory) == initial + 1
    mem = receiver.memory[-1]
    assert mem.type == MemoryType.HEAR
    assert mem.source == 0
    assert mem.claim_id == 0


def test_world_deliver_broadcast():
    """Executing BROADCAST delivers a HEAR memory to every neighbor."""
    world = _build_world(3)
    world.network[0] = [1, 2]

    sender = world.get_agent(0)
    r1, r2 = world.get_agent(1), world.get_agent(2)
    init1, init2 = len(r1.memory), len(r2.memory)

    world._execute_action(sender, Action(ActionType.BROADCAST, claim_id=0))

    assert len(r1.memory) == init1 + 1
    assert len(r2.memory) == init2 + 1

    for receiver in (r1, r2):
        mem = receiver.memory[-1]
        assert mem.type == MemoryType.HEAR
        assert mem.source == 0
        assert mem.claim_id == 0


# ---------------------------------------------------------------------------
# Observation events
# ---------------------------------------------------------------------------


def test_world_observation_events():
    """World.step delivers one private observation event per agent."""
    world = _build_world(3)
    world.private_event_rate = 1.0

    snapshot = world.step()

    assert snapshot.observation_event_count == 3
    assert sorted(snapshot.observed_ids) == [0, 1, 2]

    for agent_id in snapshot.observed_ids:
        agent = world.get_agent(agent_id)
        obs = [m for m in agent.memory if m.type == MemoryType.OBSERVE]
        assert len(obs) >= 1
        assert obs[-1].claim_id in world.claims
        assert 0.0 <= obs[-1].evidence <= 1.0


def test_attention_zero_forms_no_observation_memories():
    """With attention 0, events are still emitted but no memories are formed."""
    agents = [Agent(i, rng_seed=i, observation_attention=0.0) for i in range(5)]
    world = World(agents=agents, truths={0: True}, rng_seed=1, private_event_rate=1.0)

    snapshot = world.step()

    assert snapshot.observation_event_count >= 1
    assert snapshot.observed_ids == []


def test_attention_one_all_visible_agents_observe():
    """With rate and attention at 1.0, every agent forms an observation memory."""
    agents = [Agent(i, rng_seed=i, observation_attention=1.0) for i in range(5)]
    world = World(agents=agents, truths={0: True}, rng_seed=1, private_event_rate=1.0)

    snapshot = world.step()

    assert snapshot.observation_event_count == 5
    assert sorted(snapshot.observed_ids) == [0, 1, 2, 3, 4]


def test_private_events_one_per_agent():
    """With private_event_rate 1.0 and no global events, one event per agent."""
    agents = [Agent(i, rng_seed=i) for i in range(5)]
    world = World(
        agents=agents,
        truths={0: True},
        rng_seed=1,
        private_event_rate=1.0,
        global_event_rate=0.0,
    )

    snapshot = world.step()

    assert snapshot.observation_event_count == 5
    assert sorted(snapshot.observed_ids) == [0, 1, 2, 3, 4]


def test_global_event_visible_to_all_agents():
    """With global_event_rate 1.0 and no private events, one all-visible event."""
    agents = [Agent(i, rng_seed=i) for i in range(5)]
    world = World(
        agents=agents,
        truths={0: True},
        rng_seed=1,
        private_event_rate=0.0,
        global_event_rate=1.0,
    )

    snapshot = world.step()

    assert snapshot.observation_event_count == 1
    assert sorted(snapshot.observed_ids) == [0, 1, 2, 3, 4]


def test_global_event_attention_zero_forms_no_memories():
    """A global event is emitted but no memories form when attention is 0."""
    agents = [Agent(i, rng_seed=i, observation_attention=0.0) for i in range(5)]
    world = World(
        agents=agents,
        truths={0: True},
        rng_seed=1,
        private_event_rate=0.0,
        global_event_rate=1.0,
    )

    snapshot = world.step()

    assert snapshot.observation_event_count == 1
    assert len(snapshot.observed_ids) == 0


def test_global_event_attention_one_all_observe():
    """A global event noticed by every agent forms a memory for each."""
    agents = [Agent(i, rng_seed=i, observation_attention=1.0) for i in range(5)]
    world = World(
        agents=agents,
        truths={0: True},
        rng_seed=1,
        private_event_rate=0.0,
        global_event_rate=1.0,
    )

    snapshot = world.step()

    assert snapshot.observation_event_count == 1
    assert len(snapshot.observed_ids) == len(agents)


def test_private_and_global_events_both_observed_in_one_tick():
    """An agent hit by both a private and a global event observes each separately.

    Events are distinct world signals, so they are not deduplicated: with both
    rates and attention at 1.0, every agent forms two OBSERVE memories and
    appears twice in observed_ids in the same tick.
    """
    agents = [Agent(i, rng_seed=i, observation_attention=1.0) for i in range(5)]
    world = World(
        agents=agents,
        truths={0: True},
        rng_seed=1,
        private_event_rate=1.0,
        global_event_rate=1.0,
    )

    snapshot = world.step()

    assert snapshot.observation_event_count == len(agents) + 1
    assert len(snapshot.observed_ids) == 2 * len(agents)
    for agent in agents:
        assert snapshot.observed_ids.count(agent.id) == 2


# ---------------------------------------------------------------------------
# step() and Snapshot
# ---------------------------------------------------------------------------


def test_world_step():
    """Test World.step method."""
    world = _build_world(3)

    initial_tick = world.tick
    snapshot = world.step()

    assert world.tick == initial_tick + 1
    assert isinstance(snapshot, Snapshot)
    assert snapshot.tick == initial_tick
    assert isinstance(snapshot.observed_ids, list)
    assert isinstance(snapshot.verified_ids, list)
    assert isinstance(snapshot.communicate_edges, list)
    assert isinstance(snapshot.broadcast_edges, list)
    assert isinstance(snapshot.num_memory_processing_agents, int)
    assert isinstance(snapshot.num_belief_updating_agents, int)
    assert isinstance(snapshot.num_trust_updating_agents, int)
    assert isinstance(snapshot.agent_beliefs, dict)
    assert isinstance(snapshot.agent_memory_sizes, dict)

    assert len(snapshot.agent_beliefs) == len(world.agents)
    for agent_id, claim_beliefs in snapshot.agent_beliefs.items():
        assert isinstance(claim_beliefs, dict)
        assert len(claim_beliefs) == len(world.truths)


def test_snapshot_stores_full_beliefs():
    """Snapshot stores full beliefs as agent_id -> claim_id -> belief_value."""
    world = _build_world(3)
    snapshot = world.step()

    assert isinstance(snapshot.agent_beliefs, dict)
    assert len(snapshot.agent_beliefs) == 3

    for agent_id, claim_beliefs in snapshot.agent_beliefs.items():
        assert isinstance(agent_id, int)
        assert isinstance(claim_beliefs, dict)
        assert 0 in claim_beliefs
        for claim_id, belief_value in claim_beliefs.items():
            assert isinstance(claim_id, int)
            assert isinstance(belief_value, float)
            assert 0.0 <= belief_value <= 1.0


def test_world_step_does_not_print_logs():
    """Test that World.step() does not print logs."""
    world = _build_world(3)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        for _ in range(5):
            world.step()
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    assert output == ""


def test_execute_action_verify_returns_trace():
    world = _build_world(2)
    agent = world.get_agent(0)

    trace = world._execute_action(agent, Action(ActionType.VERIFY, claim_id=0))

    assert trace.verified_ids == [0]
    assert trace.communicate_edges == []
    assert trace.broadcast_edges == []


def test_execute_action_communicate_returns_trace():
    world = _build_world(2)
    world.network[0] = [1]
    sender = world.get_agent(0)

    trace = world._execute_action(
        sender,
        Action(ActionType.COMMUNICATE, claim_id=0, target_agent_id=1),
    )

    assert trace.verified_ids == []
    assert trace.communicate_edges == [(0, 1)]
    assert trace.broadcast_edges == []


def test_execute_action_broadcast_returns_trace():
    world = _build_world(3)
    world.network[0] = [1, 2]
    sender = world.get_agent(0)

    trace = world._execute_action(
        sender,
        Action(ActionType.BROADCAST, claim_id=0),
    )

    assert trace.verified_ids == []
    assert trace.communicate_edges == []
    assert trace.broadcast_edges == [(0, 1), (0, 2)]


def test_step_accumulates_verify_trace(mocker):
    """World.step() aggregates a per-agent VERIFY ActionTrace into the Snapshot."""
    world = _build_world(2)
    world.private_event_rate = 0.0
    world.global_event_rate = 0.0

    a0 = world.get_agent(0)
    a1 = world.get_agent(1)

    mocker.patch.object(
        a0,
        "choose_action",
        return_value=Action(ActionType.VERIFY, claim_id=0),
    )
    mocker.patch.object(
        a1,
        "choose_action",
        return_value=Action(ActionType.IDLE),
    )

    snapshot = world.step()

    assert snapshot.verified_ids == [0]
    assert snapshot.communicate_edges == []
    assert snapshot.broadcast_edges == []


def test_step_accumulates_communicate_trace(mocker):
    """World.step() aggregates a per-agent COMMUNICATE ActionTrace into the Snapshot."""
    world = _build_world(2)
    world.private_event_rate = 0.0
    world.global_event_rate = 0.0
    world.network[0] = [1]

    a0 = world.get_agent(0)
    a1 = world.get_agent(1)

    mocker.patch.object(
        a0,
        "choose_action",
        return_value=Action(ActionType.COMMUNICATE, claim_id=0, target_agent_id=1),
    )
    mocker.patch.object(
        a1,
        "choose_action",
        return_value=Action(ActionType.IDLE),
    )

    snapshot = world.step()

    assert snapshot.verified_ids == []
    assert snapshot.communicate_edges == [(0, 1)]
    assert snapshot.broadcast_edges == []


def test_step_distinguishes_belief_and_trust_updates(mocker):
    """A rejected HEAR memory can update trust without moving belief.

    This is core to bounded-confidence social dynamics, not an edge case:
    num_belief_updating_agents and num_trust_updating_agents must be able to
    diverge in the same tick.
    """
    sender = Agent(0, rng_seed=0)
    receiver = Agent(
        1,
        rng_seed=1,
        social_confidence_bound=0.01,
        social_trust_update_rate=0.5,
        social_update_trust_on_rejection=True,
    )
    world = World(agents=[sender, receiver], truths={0: True}, rng_seed=1)
    world.private_event_rate = 0.0
    world.global_event_rate = 0.0
    world.network[0] = [1]

    sender.beliefs[0] = 1.0
    receiver.beliefs[0] = 0.0

    mocker.patch.object(
        sender,
        "choose_action",
        return_value=Action(ActionType.COMMUNICATE, claim_id=0, target_agent_id=1),
    )
    mocker.patch.object(receiver, "choose_action", return_value=Action(ActionType.IDLE))

    snapshot = world.step()

    assert snapshot.num_memory_processing_agents == 1
    assert snapshot.num_belief_updating_agents == 0
    assert snapshot.num_trust_updating_agents == 1
