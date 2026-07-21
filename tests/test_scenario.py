import pytest

from simlab.agent import Agent
from simlab.scenario import extract_scenario_features
from simlab.telemetry import Telemetry
from simlab.world import World


def _build_world(profiles: list[tuple[str, dict]], *, truths=None) -> World:
    agents = []
    next_id = 0
    for profile_name, kwargs in profiles:
        agent = Agent(id=next_id, rng_seed=next_id, profile_name=profile_name, **kwargs)
        agents.append(agent)
        next_id += 1
    return World(agents=agents, truths=truths or {0: True, 1: False}, rng_seed=1)


def test_extract_scenario_features_population_and_world_settings():
    world = _build_world(
        [("default", {}), ("default", {}), ("skeptic", {"default_trust": 0.1})]
    )
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row)

    assert features["num_agents"] == 3
    assert features["num_claims"] == 2
    assert features["private_event_rate"] == world.private_event_rate
    assert features["global_event_rate"] == world.global_event_rate
    assert features["profile_count.default"] == 2
    assert features["profile_count.skeptic"] == 1
    assert features["profile_fraction.default"] == pytest.approx(2 / 3)
    assert features["profile_fraction.skeptic"] == pytest.approx(1 / 3)


def test_extract_scenario_features_graph_stats_match_network():
    world = _build_world([("default", {}) for _ in range(6)])
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row)

    out_degrees = [len(world.neighbors(agent.id)) for agent in world.agents]
    num_nodes = len(world.agents)
    num_edges = sum(out_degrees)

    assert features["graph.num_nodes"] == num_nodes
    assert features["graph.num_edges"] == num_edges
    assert features["graph.min_out_degree"] == min(out_degrees)
    assert features["graph.max_out_degree"] == max(out_degrees)
    assert features["graph.edge_density"] == pytest.approx(
        num_edges / (num_nodes * (num_nodes - 1))
    )
    assert features["graph.fraction_isolated"] == pytest.approx(
        sum(1 for d in out_degrees if d == 0) / num_nodes
    )


def test_extract_scenario_features_agent_parameter_stats():
    world = _build_world(
        [
            ("default", {"observation_attention": 0.2}),
            ("default", {"observation_attention": 0.8}),
        ]
    )
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row)

    assert features["agent_attention_mean"] == pytest.approx(0.5)
    assert features["agent_attention_std"] == pytest.approx(0.3)


def test_extract_scenario_features_initial_state_matches_telemetry_row():
    world = _build_world([("default", {}) for _ in range(4)])
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row)

    assert features["initial.belief_mean"] == initial_row.belief_mean
    assert features["initial.belief_std"] == initial_row.belief_std
    assert (
        features["initial.mean_abs_error_to_truth"]
        == initial_row.mean_abs_error_to_truth
    )
    assert (
        features["initial.fraction_truth_aligned"] == initial_row.fraction_truth_aligned
    )
    assert features["initial.mean_trust"] == initial_row.mean_trust


def test_extract_scenario_features_no_agents_does_not_crash():
    world = World(agents=[], truths={0: True}, rng_seed=1)
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row)

    assert features["num_agents"] == 0
    assert features["graph.num_nodes"] == 0
    assert features["graph.edge_density"] == 0.0
    assert features["graph.fraction_isolated"] == 0.0
