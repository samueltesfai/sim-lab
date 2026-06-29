import pytest
import random
import sys
import io
from collections import defaultdict

from simlab.sim import (
    Agent,
    World,
    Action,
    Memory,
    Snapshot,
    ActionType,
    MemoryType,
    ObservationEvent,
    clamp,
)


def _build_world(n: int = 5) -> World:
    """Helper to build a test world with n agents."""
    agents = [Agent(i, rng_seed=i) for i in range(n)]
    return World(agents=agents, truths={0: True}, rng_seed=1)


def test_clamp():
    """Test the clamp utility function."""
    # Test basic clamping
    assert clamp(0.5) == 0.5
    assert clamp(-0.1) == 0.0
    assert clamp(1.1) == 1.0

    # Test custom bounds
    assert clamp(5, min_value=0, max_value=10) == 5
    assert clamp(-5, min_value=0, max_value=10) == 0
    assert clamp(15, min_value=0, max_value=10) == 10


def test_action_validation():
    """Test Action validation logic."""
    # Test string enum conversion - this should fail as intended
    with pytest.raises(ValueError, match="'VERIFY' is not a valid ActionType"):
        Action("VERIFY", claim_id=0)

    # Test invalid actions - these test business logic validation
    with pytest.raises(ValueError, match="VERIFY action requires claim_id"):
        Action(ActionType.VERIFY)

    with pytest.raises(
        ValueError, match="COMMUNICATE action requires claim_id and target_agent_id"
    ):
        Action(ActionType.COMMUNICATE, claim_id=0)

    with pytest.raises(
        ValueError, match="COMMUNICATE action requires claim_id and target_agent_id"
    ):
        Action(ActionType.COMMUNICATE, target_agent_id=1)

    with pytest.raises(
        ValueError, match="BROADCAST action requires claim_id and no target_agent_id"
    ):
        Action(ActionType.BROADCAST)

    with pytest.raises(
        ValueError, match="BROADCAST action requires claim_id and no target_agent_id"
    ):
        Action(ActionType.BROADCAST, claim_id=0, target_agent_id=1)


def test_agent_initialization():
    """Test Agent initialization and default values."""
    agent = Agent(id=0, rng_seed=42)

    assert agent.id == 0
    assert isinstance(agent.rng, random.Random)
    assert isinstance(agent.beliefs, defaultdict)
    assert isinstance(agent.trust, defaultdict)
    assert agent.memory == []
    assert agent._mem_cursor == 0

    # Test default action preferences
    expected_preferences = {
        ActionType.IDLE: 0.0,
        ActionType.VERIFY: 0.9,
        ActionType.COMMUNICATE: 0.7,
        ActionType.BROADCAST: 0.5,
    }
    assert agent.action_preference == expected_preferences

    # Test default action costs
    expected_costs = {
        ActionType.IDLE: 0.0,
        ActionType.VERIFY: 0.35,
        ActionType.COMMUNICATE: 0.15,
        ActionType.BROADCAST: 0.30,
    }
    assert agent.action_cost == expected_costs


def test_agent_custom_initialization():
    """Test Agent initialization with custom parameters."""
    custom_preferences = {ActionType.VERIFY: 0.8, ActionType.BROADCAST: 0.6}
    custom_costs = {ActionType.VERIFY: 0.4, ActionType.BROADCAST: 0.25}

    agent = Agent(
        id=1,
        rng_seed=100,
        action_preference=custom_preferences,
        action_cost=custom_costs,
    )

    # Should merge with defaults
    assert agent.action_preference[ActionType.VERIFY] == 0.8  # Custom
    assert agent.action_preference[ActionType.BROADCAST] == 0.6  # Custom
    assert agent.action_preference[ActionType.IDLE] == 0.0  # Default
    assert agent.action_preference[ActionType.COMMUNICATE] == 0.7  # Default

    assert agent.action_cost[ActionType.VERIFY] == 0.4  # Custom
    assert agent.action_cost[ActionType.BROADCAST] == 0.25  # Custom
    assert agent.action_cost[ActionType.IDLE] == 0.0  # Default
    assert agent.action_cost[ActionType.COMMUNICATE] == 0.15  # Default


def test_agent_beliefs_and_trust():
    """Test agent belief and trust defaultdict behavior."""
    agent = Agent(id=0, rng_seed=42)

    # Test beliefs defaultdict
    assert agent.beliefs[0] >= 0.0 and agent.beliefs[0] <= 1.0
    assert agent.beliefs[1] >= 0.0 and agent.beliefs[1] <= 1.0

    # Test trust defaultdict
    assert agent.trust[0] == 0.5
    assert agent.trust[1] == 0.5

    # Test setting values
    agent.beliefs[0] = 0.8
    agent.trust[1] = 0.9
    assert agent.beliefs[0] == 0.8
    assert agent.trust[1] == 0.9


def test_agent_memory_size():
    """Test agent memory size property."""
    agent = Agent(id=0, rng_seed=42)

    assert agent.memory_size == 0

    # Add memories directly for testing
    agent.memory.append(Memory(0, MemoryType.OBSERVE, 0, None, 0, 0.5))
    assert agent.memory_size == 1

    agent.memory.append(Memory(1, MemoryType.VERIFY, 1, None, 0, 0.8))
    assert agent.memory_size == 2


def test_agent_confidence_and_uncertainty():
    """Test agent confidence and uncertainty calculations."""
    agent = Agent(id=0, rng_seed=42)

    # Test confidence calculation
    agent.beliefs[0] = 0.5  # Neutral -> confidence 0.0
    assert agent.confidence(0) == 0.0
    assert agent.uncertainty(0) == 1.0

    agent.beliefs[1] = 1.0  # Certain true -> confidence 1.0
    assert agent.confidence(1) == 1.0
    assert agent.uncertainty(1) == 0.0

    agent.beliefs[2] = 0.0  # Certain false -> confidence 1.0
    assert agent.confidence(2) == 1.0
    assert agent.uncertainty(2) == 0.0

    agent.beliefs[3] = 0.75  # Leaning true -> confidence 0.5
    assert agent.confidence(3) == 0.5
    assert agent.uncertainty(3) == 0.5


def test_agent_disagreement():
    """Test agent disagreement calculation."""
    world = _build_world(2)
    agent1 = world.get_agent(0)
    agent2 = world.get_agent(1)

    # Set different beliefs
    agent1.beliefs[0] = 0.8
    agent2.beliefs[0] = 0.2

    disagreement = agent1.disagreement(0, 1, world)
    assert disagreement == pytest.approx(0.6)  # |0.8 - 0.2|

    # Test with same beliefs
    agent1.beliefs[1] = 0.5
    agent2.beliefs[1] = 0.5
    disagreement = agent1.disagreement(1, 1, world)
    assert disagreement == 0.0


def test_agent_local_disagreement():
    """Test agent local disagreement calculation."""
    world = _build_world(3)
    agent = world.get_agent(0)

    # Set beliefs for all agents
    world.get_agent(0).beliefs[0] = 0.8
    world.get_agent(1).beliefs[0] = 0.2
    world.get_agent(2).beliefs[0] = 0.5

    # Force network connections for testing
    world.network[0] = [1, 2]

    local_disagreement = agent.local_disagreement(0, world)
    expected = (abs(0.8 - 0.2) + abs(0.8 - 0.5)) / 2  # Average disagreement
    assert local_disagreement == expected

    # Test with no neighbors
    world.network[0] = []
    assert agent.local_disagreement(0, world) == 0.0


def test_agent_action_preference_and_cost():
    """Test agent action preference and cost methods."""
    agent = Agent(id=0, rng_seed=42)
    world = _build_world(2)

    action = Action(ActionType.VERIFY, claim_id=0)

    preference = agent.get_action_preference(world, action)
    assert preference == agent.action_preference[ActionType.VERIFY]

    cost = agent.get_action_cost(world, action)
    assert cost == agent.action_cost[ActionType.VERIFY]


def test_agent_generate_candidate_actions():
    """Test agent candidate action generation."""
    world = _build_world(2)
    agent = world.get_agent(0)

    # Force network connection
    world.network[0] = [1]

    candidates = agent.generate_candidate_actions(world)

    # Should include IDLE
    idle_actions = [a for a in candidates if a.type == ActionType.IDLE]
    assert len(idle_actions) == 1

    # Should include VERIFY for each claim
    verify_actions = [a for a in candidates if a.type == ActionType.VERIFY]
    assert len(verify_actions) == len(world.truths)

    # Should include BROADCAST for each claim
    broadcast_actions = [a for a in candidates if a.type == ActionType.BROADCAST]
    assert len(broadcast_actions) == len(world.truths)

    # Should include COMMUNICATE for each neighbor-claim pair
    communicate_actions = [a for a in candidates if a.type == ActionType.COMMUNICATE]
    expected_communicates = len(world.network[0]) * len(world.truths)
    assert len(communicate_actions) == expected_communicates


def test_agent_score_action():
    """Test agent action scoring."""
    world = _build_world(2)
    agent = world.get_agent(0)

    # Set up beliefs for testing
    agent.beliefs[0] = 0.7  # High confidence
    world.get_agent(1).beliefs[0] = 0.2  # Different belief for disagreement

    # Force network connection
    world.network[0] = [1]

    # Test IDLE action
    idle_action = Action(ActionType.IDLE)
    idle_score = agent.score_action(world, idle_action)
    expected_idle = (
        agent.action_preference[ActionType.IDLE] - agent.action_cost[ActionType.IDLE]
    )
    assert idle_score == expected_idle

    # Test VERIFY action (should depend on uncertainty)
    verify_action = Action(ActionType.VERIFY, claim_id=0)
    verify_score = agent.score_action(world, verify_action)
    expected_verify = (
        agent.action_preference[ActionType.VERIFY] * agent.uncertainty(0)
        - agent.action_cost[ActionType.VERIFY]
    )
    assert verify_score == expected_verify

    # Test COMMUNICATE action
    communicate_action = Action(ActionType.COMMUNICATE, claim_id=0, target_agent_id=1)
    communicate_score = agent.score_action(world, communicate_action)
    expected_communicate = (
        agent.action_preference[ActionType.COMMUNICATE]
        * agent.confidence(0)
        * agent.disagreement(0, 1, world)
        - agent.action_cost[ActionType.COMMUNICATE]
    )
    assert communicate_score == expected_communicate

    # Test BROADCAST action
    broadcast_action = Action(ActionType.BROADCAST, claim_id=0)
    broadcast_score = agent.score_action(world, broadcast_action)
    expected_broadcast = (
        agent.action_preference[ActionType.BROADCAST]
        * agent.confidence(0)
        * agent.local_disagreement(0, world)
        - agent.action_cost[ActionType.BROADCAST]
    )
    assert broadcast_score == expected_broadcast


def test_agent_choose_action():
    """Test agent action selection."""
    world = _build_world(2)
    agent = world.get_agent(0)

    # Force network connection
    world.network[0] = [1]

    action = agent.choose_action(world)
    assert isinstance(action, Action)
    assert action.type in ActionType

    # Should be deterministic for given seed
    action2 = agent.choose_action(world)
    # This might be the same action due to deterministic scoring
    assert isinstance(action2, Action)


def test_agent_act():
    """Test agent action execution."""
    world = _build_world(2)
    agent = world.get_agent(0)

    # Test IDLE action
    idle_action = Action(ActionType.IDLE)
    agent.act(world, idle_action)  # Should not raise

    # Test VERIFY action
    verify_action = Action(ActionType.VERIFY, claim_id=0)
    initial_memory_count = len(agent.memory)
    agent.act(world, verify_action)
    assert len(agent.memory) == initial_memory_count + 1
    assert agent.memory[-1].type == MemoryType.VERIFY
    assert agent.memory[-1].claim_id == 0

    # Test invalid action type
    with pytest.raises(ValueError, match="'INVALID_TYPE' is not a valid ActionType"):
        Action("INVALID_TYPE")
    # This line should never be reached due to the above exception
    # with pytest.raises(ValueError, match="Unknown action type"):
    #     agent.act(world, invalid_action)


def test_agent_add_memory():
    """Test agent memory addition."""
    world = _build_world(2)
    agent = world.get_agent(0)

    # Test observation memory
    agent.add_memory(world, MemoryType.OBSERVE, claim_id=0, evidence=0.8)
    memory = agent.memory[-1]
    assert memory.type == MemoryType.OBSERVE
    assert memory.claim_id == 0
    assert memory.source is None
    assert memory.timestamp == world.tick
    assert memory.evidence == 0.8

    # Test verify memory
    agent.add_memory(world, MemoryType.VERIFY, claim_id=0, evidence=1.0)
    memory = agent.memory[-1]
    assert memory.type == MemoryType.VERIFY
    assert memory.claim_id == 0

    # Test hear memory
    agent.add_memory(world, MemoryType.HEAR, source=1, claim_id=0, evidence=0.4)
    memory = agent.memory[-1]
    assert memory.type == MemoryType.HEAR
    assert memory.source == 1
    assert memory.claim_id == 0

    # Memories require explicit claim_id and evidence
    with pytest.raises(TypeError):
        agent.add_memory(world, MemoryType.OBSERVE, claim_id=0)
    with pytest.raises(TypeError):
        agent.add_memory(world, MemoryType.VERIFY, evidence=0.5)


def test_agent_update_beliefs():
    """Test agent belief updates."""
    world = _build_world(1)
    agent = world.get_agent(0)

    # Set initial belief
    agent.beliefs[0] = 0.5

    # Add memory with evidence
    agent.add_memory(world, MemoryType.VERIFY, claim_id=0, evidence=1.0)

    # Update beliefs
    updated = agent.update_beliefs()
    assert updated
    assert agent._mem_cursor == 1

    # Belief should have moved toward evidence
    # (exact value depends on evidence and learning rate)
    assert 0.0 <= agent.beliefs[0] <= 1.0

    # Test no new memories
    updated_again = agent.update_beliefs()
    assert not updated_again


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
    assert world.noise[MemoryType.VERIFY] == 0.0  # Default

    # Test network generation
    assert isinstance(world.network, dict)
    assert len(world.network) == 3
    for agent_id, connections in world.network.items():
        assert isinstance(connections, list)
        assert agent_id not in connections  # No self-connections


def test_world_properties():
    """Test World properties."""
    world = _build_world(3)

    # Test agents property
    agents = world.agents
    assert len(agents) == 3
    assert all(isinstance(agent, Agent) for agent in agents)

    # Test claims property
    claims = world.claims
    assert claims == [0]  # From _build_world

    # Test edges property
    edges = world.edges
    assert isinstance(edges, list)
    for edge in edges:
        assert isinstance(edge, tuple)
        assert len(edge) == 2


def test_world_get_agent():
    """Test World.get_agent method."""
    world = _build_world(3)

    agent = world.get_agent(0)
    assert isinstance(agent, Agent)
    assert agent.id == 0

    # Test invalid agent ID
    with pytest.raises(KeyError):
        world.get_agent(999)


def test_world_get_agent_beliefs_snapshot():
    """Test World.get_agent_beliefs_snapshot method."""
    world = _build_world(2)

    snapshot = world.get_agent_beliefs_snapshot()

    assert isinstance(snapshot, dict)
    assert len(snapshot) == 2  # Number of agents

    for agent_id, claim_beliefs in snapshot.items():
        assert isinstance(agent_id, int)
        assert isinstance(claim_beliefs, dict)
        assert len(claim_beliefs) == len(world.truths)

        for claim_id, belief_value in claim_beliefs.items():
            assert isinstance(claim_id, int)
            assert isinstance(belief_value, float)
            assert 0.0 <= belief_value <= 1.0


def test_world_observation_events():
    """Test World observation event generation and delivery."""
    world = _build_world(3)

    # Set rate to 1.0 so the world emits one event per agent (deterministic).
    world.private_event_rate = 1.0

    events = world.generate_observation_events()

    assert isinstance(events, list)
    assert len(events) == 3  # One private event per agent
    for event in events:
        assert len(event.visible_agent_ids) == 1
        assert event.claim_id in world.claims
        assert 0.0 <= event.evidence <= 1.0

    # With default attention 1.0, all visible agents should observe.
    observed_agents = world.deliver_observation_events(events)

    assert isinstance(observed_agents, list)
    assert sorted(observed_agents) == [0, 1, 2]

    # Agents should have received memories
    for agent_id in observed_agents:
        agent = world.get_agent(agent_id)
        assert len(agent.memory) > 0
        memory = agent.memory[-1]
        assert memory.type == MemoryType.OBSERVE
        assert memory.claim_id in world.claims


def test_world_deliver_communicate():
    """Test World.deliver_communicate method."""
    world = _build_world(2)

    receiver = world.get_agent(1)

    initial_memory_count = len(receiver.memory)

    world.deliver_communicate(0, 1, 0)

    assert len(receiver.memory) == initial_memory_count + 1
    memory = receiver.memory[-1]
    assert memory.type == MemoryType.HEAR
    assert memory.source == 0
    assert memory.claim_id == 0


def test_world_deliver_broadcast():
    """Test World.deliver_broadcast method."""
    world = _build_world(3)

    # Force network connections
    world.network[0] = [1, 2]

    receiver1 = world.get_agent(1)
    receiver2 = world.get_agent(2)

    initial_memory_count_1 = len(receiver1.memory)
    initial_memory_count_2 = len(receiver2.memory)

    world.deliver_broadcast(0, 0)

    assert len(receiver1.memory) == initial_memory_count_1 + 1
    assert len(receiver2.memory) == initial_memory_count_2 + 1

    # Check memories
    for receiver in [receiver1, receiver2]:
        memory = receiver.memory[-1]
        assert memory.type == MemoryType.HEAR
        assert memory.source == 0
        assert memory.claim_id == 0


def test_world_step():
    """Test World.step method."""
    world = _build_world(3)

    initial_tick = world.tick
    snapshot = world.step()

    # Check tick advancement
    assert world.tick == initial_tick + 1

    # Check snapshot structure
    assert isinstance(snapshot, Snapshot)
    assert snapshot.tick == initial_tick
    assert isinstance(snapshot.observed_ids, list)
    assert isinstance(snapshot.verified_ids, list)
    assert isinstance(snapshot.communicate_edges, list)
    assert isinstance(snapshot.broadcast_edges, list)
    assert isinstance(snapshot.n_agent_updates, int)
    assert isinstance(snapshot.agent_beliefs, dict)
    assert isinstance(snapshot.agent_memory_sizes, dict)

    # Check belief snapshot
    assert len(snapshot.agent_beliefs) == len(world.agents)
    for agent_id, claim_beliefs in snapshot.agent_beliefs.items():
        assert isinstance(claim_beliefs, dict)
        assert len(claim_beliefs) == len(world.truths)


def test_snapshot_stores_full_beliefs():
    """Test that Snapshot stores full beliefs as agent_id -> claim_id -> belief_value."""
    world = _build_world(3)
    snapshot = world.step()

    # beliefs should be a dict with agent_id as keys
    assert isinstance(snapshot.agent_beliefs, dict)
    assert len(snapshot.agent_beliefs) == 3

    # Each agent's beliefs should be a dict with claim_id as keys
    for agent_id, claim_beliefs in snapshot.agent_beliefs.items():
        assert isinstance(agent_id, int)
        assert isinstance(claim_beliefs, dict)
        # Each agent should have beliefs for at least claim 0
        assert 0 in claim_beliefs
        # Belief values should be floats in [0, 1]
        for claim_id, belief_value in claim_beliefs.items():
            assert isinstance(claim_id, int)
            assert isinstance(belief_value, float)
            assert 0.0 <= belief_value <= 1.0


def test_world_step_does_not_print_logs():
    """Test that World.step() does not print logs (no World.log_step())."""
    world = _build_world(3)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        for _ in range(5):
            world.step()
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # World.step() should not print anything
    assert output == ""


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


def test_attention_zero_forms_no_observation_memories():
    """With attention 0, events are still emitted but no memories are formed."""
    agents = [Agent(i, rng_seed=i, observation_attention=0.0) for i in range(5)]
    world = World(
        agents=agents,
        truths={0: True},
        rng_seed=1,
        private_event_rate=1.0,
    )

    snapshot = world.step()

    assert snapshot.observation_event_count >= 1
    assert snapshot.observed_ids == []


def test_attention_one_all_visible_agents_observe():
    """With rate and attention at 1.0, every agent forms an observation memory."""
    agents = [Agent(i, rng_seed=i, observation_attention=1.0) for i in range(5)]
    world = World(
        agents=agents,
        truths={0: True},
        rng_seed=1,
        private_event_rate=1.0,
    )

    events = world.generate_observation_events()
    observed = world.deliver_observation_events(events)

    assert len(events) == 5
    assert sorted(observed) == [0, 1, 2, 3, 4]


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

    events = world.generate_observation_events()

    assert len(events) == 5
    for event in events:
        assert len(event.visible_agent_ids) == 1


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

    events = world.generate_observation_events()

    assert len(events) == 1
    assert sorted(events[0].visible_agent_ids) == [0, 1, 2, 3, 4]


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

    # One private event per agent plus a single global event.
    assert snapshot.observation_event_count == len(agents) + 1
    # Each agent observes twice (private + global).
    assert len(snapshot.observed_ids) == 2 * len(agents)
    for agent in agents:
        assert snapshot.observed_ids.count(agent.id) == 2


def test_encode_observation_applies_bias():
    """encode_observation shifts evidence by the agent's perceptual bias."""
    world = _build_world(1)
    agent = Agent(0, rng_seed=0, observation_bias=0.1)
    event = ObservationEvent(
        id=0,
        tick=0,
        claim_id=0,
        evidence=0.5,
        visible_agent_ids=(0,),
    )

    assert agent.encode_observation(world, event) == pytest.approx(0.6)


def test_learning_rate_heterogeneity_affects_update_magnitude():
    """Higher learning rate moves belief farther toward the same evidence."""
    world = _build_world(2)

    slow = Agent(0, rng_seed=0, learning_rate=0.01)
    fast = Agent(1, rng_seed=1, learning_rate=0.5)

    for agent in (slow, fast):
        agent.beliefs[0] = 0.5
        agent.add_memory(world, MemoryType.OBSERVE, claim_id=0, evidence=1.0)
        agent.update_beliefs()

    # Both move toward 1.0; the faster learner moves farther.
    assert fast.beliefs[0] > slow.beliefs[0]


def test_default_trust_heterogeneity_affects_heard_update():
    """Higher default trust gives heard evidence more weight."""
    world = _build_world(3)

    low = Agent(0, rng_seed=0, default_trust=0.1)
    high = Agent(1, rng_seed=1, default_trust=0.9)

    for agent in (low, high):
        agent.beliefs[0] = 0.5
        # source agent 2 is unseen, so trust falls back to default_trust
        agent.add_memory(world, MemoryType.HEAR, source=2, claim_id=0, evidence=1.0)
        agent.update_beliefs()

    assert high.beliefs[0] > low.beliefs[0]
