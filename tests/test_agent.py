import pytest
import random
from collections import defaultdict

from simlab.agent import Agent
from simlab.config_schema import AgentSettings
from simlab.world import World
from simlab.kernel_types import Action, ActionType, Memory, MemoryType, ObservationEvent


def _build_world(n: int = 5) -> World:
    agents = [Agent(i, rng_seed=i) for i in range(n)]
    return World.from_dict(agents, {"truths": {0: True}, "rng_seed": 1})


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_agent_init_consumes_every_settings_field(field_tracker):
    """Every AgentSettings field must be read somewhere in Agent.__init__ --
    otherwise a field could be added to the schema with no behavior wired up
    for it, and nothing would notice."""
    tracked = field_tracker(AgentSettings(), AgentSettings)
    Agent(id=0, settings=tracked)
    tracked.assert_fully_consumed()


def test_agent_initialization():
    """Test Agent initialization and default values."""
    agent = Agent(id=0, rng_seed=42)

    assert agent.id == 0
    assert isinstance(agent.rng, random.Random)
    assert isinstance(agent.beliefs, defaultdict)
    assert isinstance(agent.trust, defaultdict)
    assert agent.memory == []

    expected_preferences = {
        ActionType.IDLE: 0.0,
        ActionType.VERIFY: 0.9,
        ActionType.COMMUNICATE: 0.7,
        ActionType.BROADCAST: 0.5,
    }
    assert agent.action_preference == expected_preferences

    expected_costs = {
        ActionType.IDLE: 0.0,
        ActionType.VERIFY: 0.35,
        ActionType.COMMUNICATE: 0.15,
        ActionType.BROADCAST: 0.30,
    }
    assert agent.action_cost == expected_costs


def test_agent_custom_initialization():
    """Test Agent initialization with custom parameters."""
    custom_preferences = {"VERIFY": 0.8, "BROADCAST": 0.6}
    custom_costs = {"VERIFY": 0.4, "BROADCAST": 0.25}

    agent = Agent.from_dict(
        1,
        {"action_preference": custom_preferences, "action_cost": custom_costs},
        rng_seed=100,
    )

    assert agent.action_preference[ActionType.VERIFY] == 0.8
    assert agent.action_preference[ActionType.BROADCAST] == 0.6
    assert agent.action_preference[ActionType.IDLE] == 0.0
    assert agent.action_preference[ActionType.COMMUNICATE] == 0.7

    assert agent.action_cost[ActionType.VERIFY] == 0.4
    assert agent.action_cost[ActionType.BROADCAST] == 0.25
    assert agent.action_cost[ActionType.IDLE] == 0.0
    assert agent.action_cost[ActionType.COMMUNICATE] == 0.15


def test_agent_social_params_stored_on_init():
    """Social parameters are stored as attributes on Agent."""
    agent = Agent.from_dict(
        0,
        {
            "social": {
                "confidence_bound": 0.4,
                "trust_update_rate": 0.2,
                "update_trust_on_rejection": False,
            }
        },
        rng_seed=0,
    )
    assert agent.social_confidence_bound == pytest.approx(0.4)
    assert agent.social_trust_update_rate == pytest.approx(0.2)
    assert agent.social_update_trust_on_rejection is False


def test_agent_social_params_defaults():
    """Default social params preserve pre-existing behavior."""
    agent = Agent(0, rng_seed=0)
    assert agent.social_confidence_bound == pytest.approx(1.0)
    assert agent.social_trust_update_rate == pytest.approx(0.0)
    assert agent.social_update_trust_on_rejection is True


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_agent_memory_size():
    """Test agent memory size property."""
    agent = Agent(id=0, rng_seed=42)

    assert agent.memory_size == 0

    agent.memory.append(Memory(0, MemoryType.OBSERVE, 0, None, 0, 0.5))
    assert agent.memory_size == 1

    agent.memory.append(Memory(1, MemoryType.VERIFY, 1, None, 0, 0.8))
    assert agent.memory_size == 2


def test_add_memory_records_fields(memory_seeder):
    """Agent._add_memory stores evidence with the provided metadata."""
    world = _build_world(2)
    agent = world.get_agent(0)

    memory_seeder(
        agent, memory_type=MemoryType.OBSERVE, claim_id=0, evidence=0.8, tick=world.tick
    )
    memory = agent.memory[-1]
    assert memory.type == MemoryType.OBSERVE
    assert memory.claim_id == 0
    assert memory.source is None
    assert memory.timestamp == world.tick
    assert memory.evidence == 0.8

    memory_seeder(agent, memory_type=MemoryType.VERIFY, claim_id=0, evidence=1.0)
    memory = agent.memory[-1]
    assert memory.type == MemoryType.VERIFY
    assert memory.claim_id == 0

    memory_seeder(
        agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=0.4
    )
    memory = agent.memory[-1]
    assert memory.type == MemoryType.HEAR
    assert memory.source == 1
    assert memory.claim_id == 0

    with pytest.raises(TypeError):
        agent._add_memory(tick=0, memory_type=MemoryType.OBSERVE, claim_id=0)
    with pytest.raises(TypeError):
        agent._add_memory(tick=0, memory_type=MemoryType.VERIFY, evidence=0.5)


# ---------------------------------------------------------------------------
# Beliefs and trust
# ---------------------------------------------------------------------------


def test_agent_beliefs_and_trust():
    """Test agent belief and trust defaultdict behavior."""
    agent = Agent(id=0, rng_seed=42)

    assert 0.0 <= agent.beliefs[0] <= 1.0
    assert 0.0 <= agent.beliefs[1] <= 1.0
    assert agent.trust[0] == 0.5
    assert agent.trust[1] == 0.5

    agent.beliefs[0] = 0.8
    agent.trust[1] = 0.9
    assert agent.beliefs[0] == 0.8
    assert agent.trust[1] == 0.9


def test_agent_confidence_and_uncertainty():
    """Test agent confidence and uncertainty calculations."""
    agent = Agent(id=0, rng_seed=42)

    agent.beliefs[0] = 0.5
    assert agent.confidence(0) == 0.0
    assert agent.uncertainty(0) == 1.0

    agent.beliefs[1] = 1.0
    assert agent.confidence(1) == 1.0
    assert agent.uncertainty(1) == 0.0

    agent.beliefs[2] = 0.0
    assert agent.confidence(2) == 1.0
    assert agent.uncertainty(2) == 0.0

    agent.beliefs[3] = 0.75
    assert agent.confidence(3) == 0.5
    assert agent.uncertainty(3) == 0.5


# ---------------------------------------------------------------------------
# Action generation and scoring
# ---------------------------------------------------------------------------


def test_agent_disagreement():
    """Test agent disagreement calculation."""
    world = _build_world(2)
    agent1 = world.get_agent(0)
    agent2 = world.get_agent(1)

    agent1.beliefs[0] = 0.8
    agent2.beliefs[0] = 0.2

    assert agent1.disagreement(0, 1, world) == pytest.approx(0.6)

    agent1.beliefs[1] = 0.5
    agent2.beliefs[1] = 0.5
    assert agent1.disagreement(1, 1, world) == 0.0


def test_agent_local_disagreement():
    """Test agent local disagreement calculation."""
    world = _build_world(3)
    agent = world.get_agent(0)

    world.get_agent(0).beliefs[0] = 0.8
    world.get_agent(1).beliefs[0] = 0.2
    world.get_agent(2).beliefs[0] = 0.5

    world.network[0] = [1, 2]

    expected = (abs(0.8 - 0.2) + abs(0.8 - 0.5)) / 2
    assert agent.local_disagreement(0, world) == expected

    world.network[0] = []
    assert agent.local_disagreement(0, world) == 0.0


def test_agent_generate_candidate_actions():
    """Test agent candidate action generation."""
    world = _build_world(2)
    agent = world.get_agent(0)

    world.network[0] = [1]

    candidates = agent.generate_candidate_actions(world)

    idle_actions = [a for a in candidates if a.type == ActionType.IDLE]
    assert len(idle_actions) == 1

    verify_actions = [a for a in candidates if a.type == ActionType.VERIFY]
    assert len(verify_actions) == len(world.truths)

    broadcast_actions = [a for a in candidates if a.type == ActionType.BROADCAST]
    assert len(broadcast_actions) == len(world.truths)

    communicate_actions = [a for a in candidates if a.type == ActionType.COMMUNICATE]
    assert len(communicate_actions) == len(world.network[0]) * len(world.truths)


def test_agent_score_action():
    """Test agent action scoring."""
    world = _build_world(2)
    agent = world.get_agent(0)

    agent.beliefs[0] = 0.7
    world.get_agent(1).beliefs[0] = 0.2
    world.network[0] = [1]

    idle_score = agent.score_action(world, Action(ActionType.IDLE))
    assert idle_score == (
        agent.action_preference[ActionType.IDLE] - agent.action_cost[ActionType.IDLE]
    )

    verify_score = agent.score_action(world, Action(ActionType.VERIFY, claim_id=0))
    assert verify_score == (
        agent.action_preference[ActionType.VERIFY] * agent.uncertainty(0)
        - agent.action_cost[ActionType.VERIFY]
    )

    communicate_score = agent.score_action(
        world, Action(ActionType.COMMUNICATE, claim_id=0, target_agent_id=1)
    )
    assert communicate_score == (
        agent.action_preference[ActionType.COMMUNICATE]
        * agent.confidence(0)
        * agent.disagreement(0, 1, world)
        - agent.action_cost[ActionType.COMMUNICATE]
    )

    broadcast_score = agent.score_action(
        world, Action(ActionType.BROADCAST, claim_id=0)
    )
    assert broadcast_score == (
        agent.action_preference[ActionType.BROADCAST]
        * agent.confidence(0)
        * agent.local_disagreement(0, world)
        - agent.action_cost[ActionType.BROADCAST]
    )


def test_agent_choose_action():
    """Test agent action selection."""
    world = _build_world(2)
    agent = world.get_agent(0)

    world.network[0] = [1]

    action = agent.choose_action(world)
    assert isinstance(action, Action)
    assert action.type in ActionType

    action2 = agent.choose_action(world)
    assert isinstance(action2, Action)


# ---------------------------------------------------------------------------
# Belief updates
# ---------------------------------------------------------------------------


def test_agent_update_beliefs(memory_seeder):
    """Test agent belief updates."""
    world = _build_world(1)
    agent = world.get_agent(0)

    agent.beliefs[0] = 0.5
    memory_seeder(agent, memory_type=MemoryType.VERIFY, claim_id=0, evidence=1.0)

    trace = agent.update_beliefs()
    assert trace.processed_memory
    assert trace.belief_changed
    assert 0.0 <= agent.beliefs[0] <= 1.0

    trace = agent.update_beliefs()
    assert not trace.processed_memory
    assert not trace.belief_changed


def test_learning_rate_heterogeneity_affects_update_magnitude(memory_seeder):
    """Higher learning rate moves belief farther toward the same evidence."""
    slow = Agent.from_dict(0, {"learning": {"rate": 0.01}}, rng_seed=0)
    fast = Agent.from_dict(1, {"learning": {"rate": 0.5}}, rng_seed=1)

    for agent in (slow, fast):
        agent.beliefs[0] = 0.5
        memory_seeder(agent, memory_type=MemoryType.OBSERVE, claim_id=0, evidence=1.0)
        agent.update_beliefs()

    assert fast.beliefs[0] > slow.beliefs[0]


def test_default_trust_heterogeneity_affects_heard_update(memory_seeder):
    """Higher default trust gives heard evidence more weight."""
    low = Agent.from_dict(0, {"trust": {"default": 0.1}}, rng_seed=0)
    high = Agent.from_dict(1, {"trust": {"default": 0.9}}, rng_seed=1)

    for agent in (low, high):
        agent.beliefs[0] = 0.5
        memory_seeder(
            agent, memory_type=MemoryType.HEAR, source=2, claim_id=0, evidence=1.0
        )
        agent.update_beliefs()

    assert high.beliefs[0] > low.beliefs[0]


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


def test_encode_observation_applies_bias():
    """encode_observation shifts evidence by the agent's perceptual bias."""
    agent = Agent.from_dict(0, {"observation": {"bias": 0.1}}, rng_seed=0)
    event = ObservationEvent(
        id=0, tick=0, claim_id=0, evidence=0.5, visible_agent_ids=(0,)
    )
    assert agent.encode_observation(event) == pytest.approx(0.6)


def test_attention_edge_cases_do_not_perturb_belief_rng():
    """Deterministic attention (0.0 or 1.0) consumes no RNG."""
    event = ObservationEvent(
        id=0, tick=0, claim_id=0, evidence=0.5, visible_agent_ids=(0,)
    )

    always_attentive = Agent.from_dict(
        0, {"observation": {"attention": 1.0}}, rng_seed=42
    )
    never_attentive = Agent.from_dict(
        0, {"observation": {"attention": 0.0}}, rng_seed=42
    )
    quiet = Agent.from_dict(0, {"observation": {"attention": 1.0}}, rng_seed=42)

    for _ in range(25):
        always_attentive.notices_observation(event)
        never_attentive.notices_observation(event)

    assert always_attentive.beliefs[0] == quiet.beliefs[0]
    assert never_attentive.beliefs[0] == quiet.beliefs[0]


# ---------------------------------------------------------------------------
# Bounded confidence
# ---------------------------------------------------------------------------


def test_hear_inside_confidence_bound_updates_belief(memory_seeder):
    """HEAR within the confidence bound updates the belief normally."""
    agent = Agent.from_dict(0, {"social": {"confidence_bound": 1.0}}, rng_seed=0)
    agent.beliefs[0] = 0.5
    memory_seeder(
        agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=0.8
    )
    agent.update_beliefs()

    assert agent.beliefs[0] > 0.5


def test_hear_outside_confidence_bound_does_not_update_belief(memory_seeder):
    """HEAR farther than the confidence bound leaves the belief unchanged."""
    agent = Agent.from_dict(0, {"social": {"confidence_bound": 0.1}}, rng_seed=0)
    agent.beliefs[0] = 0.5
    memory_seeder(
        agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=1.0
    )
    agent.update_beliefs()

    assert agent.beliefs[0] == pytest.approx(0.5)


def test_observe_unaffected_by_confidence_bound(memory_seeder):
    """Bounded confidence applies only to HEAR; OBSERVE is always processed."""
    agent = Agent.from_dict(0, {"social": {"confidence_bound": 0.0}}, rng_seed=0)
    agent.beliefs[0] = 0.5
    memory_seeder(agent, memory_type=MemoryType.OBSERVE, claim_id=0, evidence=1.0)
    agent.update_beliefs()

    assert agent.beliefs[0] > 0.5


def test_verify_unaffected_by_confidence_bound(memory_seeder):
    """Bounded confidence applies only to HEAR; VERIFY is always processed."""
    agent = Agent.from_dict(0, {"social": {"confidence_bound": 0.0}}, rng_seed=0)
    agent.beliefs[0] = 0.5
    memory_seeder(agent, memory_type=MemoryType.VERIFY, claim_id=0, evidence=1.0)
    agent.update_beliefs()

    assert agent.beliefs[0] > 0.5


def test_confidence_bound_default_preserves_behavior(memory_seeder):
    """Default confidence_bound=1.0 never rejects HEAR evidence (max distance is 1)."""
    agent_default = Agent(0, rng_seed=0)
    agent_open = Agent.from_dict(0, {"social": {"confidence_bound": 1.0}}, rng_seed=0)

    for agent in (agent_default, agent_open):
        agent.beliefs[0] = 0.3
        memory_seeder(
            agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=0.9
        )
        agent.update_beliefs()

    assert agent_default.beliefs[0] == pytest.approx(agent_open.beliefs[0])


# ---------------------------------------------------------------------------
# Dynamic trust
# ---------------------------------------------------------------------------


def test_trust_increases_when_agreement_exceeds_current_trust(memory_seeder):
    """Trust increases when the source agrees more than the current trust level."""
    agent = Agent.from_dict(
        0,
        {"social": {"trust_update_rate": 0.5, "confidence_bound": 1.0}},
        rng_seed=0,
    )
    agent.beliefs[0] = 0.5
    agent.trust[1] = 0.2
    memory_seeder(
        agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=0.5
    )
    agent.update_beliefs()

    assert agent.trust[1] > 0.2


def test_trust_decreases_when_agreement_below_current_trust(memory_seeder):
    """Trust decreases when the source agrees less than the current trust level."""
    agent = Agent.from_dict(
        0,
        {"social": {"trust_update_rate": 0.5, "confidence_bound": 1.0}},
        rng_seed=0,
    )
    agent.beliefs[0] = 0.5
    agent.trust[1] = 0.9
    memory_seeder(
        agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=1.0
    )
    agent.update_beliefs()

    assert agent.trust[1] < 0.9


def test_rejected_hear_updates_trust_when_flag_true(memory_seeder):
    """Rejected HEAR updates trust when update_trust_on_rejection=True."""
    agent = Agent.from_dict(
        0,
        {
            "social": {
                "confidence_bound": 0.1,
                "trust_update_rate": 0.5,
                "update_trust_on_rejection": True,
            }
        },
        rng_seed=0,
    )
    agent.beliefs[0] = 0.5
    agent.trust[1] = 0.9
    initial_trust = agent.trust[1]
    memory_seeder(
        agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=1.0
    )
    agent.update_beliefs()

    assert agent.trust[1] != pytest.approx(initial_trust)


def test_rejected_hear_does_not_update_trust_when_flag_false(memory_seeder):
    """Rejected HEAR does not update trust when update_trust_on_rejection=False."""
    agent = Agent.from_dict(
        0,
        {
            "social": {
                "confidence_bound": 0.1,
                "trust_update_rate": 0.5,
                "update_trust_on_rejection": False,
            }
        },
        rng_seed=0,
    )
    agent.beliefs[0] = 0.5
    agent.trust[1] = 0.9
    initial_trust = agent.trust[1]
    memory_seeder(
        agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=1.0
    )
    agent.update_beliefs()

    assert agent.trust[1] == pytest.approx(initial_trust)


def test_trust_update_rate_zero_leaves_trust_unchanged(memory_seeder):
    """Default trust_update_rate=0.0 never mutates trust."""
    agent = Agent.from_dict(0, {"social": {"trust_update_rate": 0.0}}, rng_seed=0)
    agent.beliefs[0] = 0.5
    agent.trust[1] = 0.7
    memory_seeder(
        agent, memory_type=MemoryType.HEAR, source=1, claim_id=0, evidence=1.0
    )
    agent.update_beliefs()

    assert agent.trust[1] == pytest.approx(0.7)
