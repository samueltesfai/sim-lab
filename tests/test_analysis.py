import pytest

from simlab.agent import Agent
from simlab.analysis import (
    compute_run_summary,
    extract_scenario_features,
    find_convergence_tick,
)
from simlab.telemetry import Telemetry, TelemetryRow
from simlab.world import World


def _build_world(profiles: list[tuple[str, dict]], *, truths=None) -> World:
    agents = []
    next_id = 0
    for profile_name, kwargs in profiles:
        agent = Agent(id=next_id, rng_seed=next_id, profile_name=profile_name, **kwargs)
        agents.append(agent)
        next_id += 1
    return World(agents=agents, truths=truths or {0: True, 1: False}, rng_seed=1)


# ---------------------------------------------------------------------------
# extract_scenario_features
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# find_convergence_tick / compute_run_summary
# ---------------------------------------------------------------------------

DEFAULT_ROW_KWARGS = dict(
    belief_mean=0.5,
    belief_std=0.1,
    belief_min=0.0,
    belief_max=1.0,
    mean_abs_delta=0.0,
    max_abs_delta=0.0,
    mean_abs_error_to_truth=0.5,
    max_abs_error_to_truth=0.5,
    fraction_truth_aligned=0.0,
    num_observation_events=0,
    num_observations=0,
    num_verifications=0,
    num_communicate_edges=0,
    num_broadcast_edges=0,
    num_memory_processing_agents=0,
    num_belief_updating_agents=0,
    num_trust_updating_agents=0,
    mean_claim_belief_variance=0.1,
    fraction_confident_wrong=0.0,
    mean_trust=0.5,
    trust_std=0.1,
    step_runtime_ms=1.0,
)


def _row(tick: int, **overrides) -> TelemetryRow:
    kwargs = {**DEFAULT_ROW_KWARGS, **overrides}
    if tick == -1:
        kwargs["step_runtime_ms"] = None
    return TelemetryRow(tick=tick, **kwargs)


def test_find_convergence_tick_detects_stable_window():
    unstable = [
        _row(t, mean_abs_delta=0.5, mean_claim_belief_variance=0.5) for t in range(5)
    ]
    stable = [
        _row(t, mean_abs_delta=0.0005, mean_claim_belief_variance=0.001)
        for t in range(5, 30)
    ]
    rows = unstable + stable

    tick = find_convergence_tick(
        rows, delta_threshold=0.001, disagreement_threshold=0.0025, window=20
    )

    assert tick == 5


def test_find_convergence_tick_ignores_temporary_stability():
    rows = (
        [
            _row(t, mean_abs_delta=0.0005, mean_claim_belief_variance=0.001)
            for t in range(0, 10)
        ]
        + [
            _row(t, mean_abs_delta=0.5, mean_claim_belief_variance=0.5)
            for t in range(10, 12)
        ]
        + [
            _row(t, mean_abs_delta=0.0005, mean_claim_belief_variance=0.001)
            for t in range(12, 37)
        ]
    )

    tick = find_convergence_tick(
        rows, delta_threshold=0.001, disagreement_threshold=0.0025, window=20
    )

    # The first stable run (10 ticks) is too short; only the second (25
    # ticks, starting at tick 12) satisfies the window.
    assert tick == 12


def test_find_convergence_tick_returns_none_when_never_stable():
    rows = [
        _row(t, mean_abs_delta=0.5, mean_claim_belief_variance=0.5) for t in range(30)
    ]

    tick = find_convergence_tick(
        rows, delta_threshold=0.001, disagreement_threshold=0.0025, window=20
    )

    assert tick is None


def test_find_convergence_tick_returns_none_when_fewer_rows_than_window():
    rows = [
        _row(t, mean_abs_delta=0.0, mean_claim_belief_variance=0.0) for t in range(5)
    ]

    tick = find_convergence_tick(
        rows, delta_threshold=0.001, disagreement_threshold=0.0025, window=20
    )

    assert tick is None


def test_find_convergence_tick_ignores_initial_row():
    # The initial row (tick=-1) sits below the thresholds by construction,
    # but it must not count toward the stability window.
    initial = [_row(-1, mean_abs_delta=0.0, mean_claim_belief_variance=0.0)]
    stepped = [
        _row(t, mean_abs_delta=0.0005, mean_claim_belief_variance=0.001)
        for t in range(19)
    ]

    tick = find_convergence_tick(
        initial + stepped,
        delta_threshold=0.001,
        disagreement_threshold=0.0025,
        window=20,
    )

    # Only 19 stepped rows are stable -- one short of the window.
    assert tick is None


def test_compute_run_summary_raises_on_empty_telemetry():
    with pytest.raises(ValueError):
        compute_run_summary([], total_runtime_ms=0.0)


def test_compute_run_summary_uses_initial_row_when_no_steps_ran():
    initial = _row(-1, mean_abs_error_to_truth=0.4, fraction_truth_aligned=0.1)

    summary = compute_run_summary([initial], total_runtime_ms=0.0)

    assert summary.initial_mean_truth_error == 0.4
    assert summary.final_mean_truth_error == 0.4
    assert summary.min_mean_truth_error == 0.4
    assert summary.max_mean_truth_error == 0.4
    assert summary.mean_belief_volatility == 0.0
    assert summary.mean_step_runtime_ms == 0.0
    assert summary.p95_step_runtime_ms == 0.0
    assert summary.convergence_tick is None
    assert summary.converged is False


def test_compute_run_summary_initial_and_final_match_first_and_last_row():
    initial = _row(-1, mean_abs_error_to_truth=0.5, fraction_confident_wrong=0.3)
    steps = [
        _row(0, mean_abs_error_to_truth=0.4),
        _row(1, mean_abs_error_to_truth=0.3, mean_trust=0.7, trust_std=0.2),
    ]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=12.0)

    assert summary.initial_mean_truth_error == 0.5
    assert summary.initial_fraction_confident_wrong == 0.3
    assert summary.final_mean_truth_error == 0.3
    assert summary.final_mean_trust == 0.7
    assert summary.final_trust_std == 0.2
    assert summary.total_runtime_ms == 12.0


def test_compute_run_summary_totals_equal_sum_over_trajectory():
    initial = _row(-1)
    steps = [
        _row(0, num_observations=2, num_verifications=1, num_communicate_edges=3),
        _row(1, num_observations=1, num_verifications=0, num_communicate_edges=1),
        _row(2, num_observations=4, num_verifications=2, num_communicate_edges=0),
    ]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=0.0)

    assert summary.total_observations == 7
    assert summary.total_verifications == 3
    assert summary.total_communicate_edges == 4


def test_compute_run_summary_min_max_truth_error_over_trajectory():
    initial = _row(-1, mean_abs_error_to_truth=0.9)
    steps = [
        _row(0, mean_abs_error_to_truth=0.6),
        _row(1, mean_abs_error_to_truth=0.2),
        _row(2, mean_abs_error_to_truth=0.5),
    ]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=0.0)

    # Initial row is excluded from the trajectory min/max window.
    assert summary.min_mean_truth_error == 0.2
    assert summary.max_mean_truth_error == 0.6


def test_compute_run_summary_step_runtime_stats():
    initial = _row(-1)
    steps = [_row(t, step_runtime_ms=rt) for t, rt in enumerate([1.0, 2.0, 3.0, 4.0])]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=10.0)

    assert summary.mean_step_runtime_ms == pytest.approx(2.5)
    assert summary.p95_step_runtime_ms == 4.0


def test_compute_run_summary_stable_consensus_and_aligned():
    initial = _row(-1)
    steps = [
        _row(
            t,
            mean_abs_delta=0.0005,
            mean_claim_belief_variance=0.001,
            fraction_truth_aligned=0.95,
        )
        for t in range(25)
    ]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=0.0)

    assert summary.converged is True
    assert summary.final_consensus is True
    assert summary.final_truth_aligned is True
    assert summary.final_false_consensus is False


def test_compute_run_summary_false_consensus_when_stable_but_wrong():
    initial = _row(-1)
    steps = [
        _row(
            t,
            mean_abs_delta=0.0005,
            mean_claim_belief_variance=0.001,
            fraction_truth_aligned=0.0,
        )
        for t in range(25)
    ]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=0.0)

    assert summary.final_consensus is True
    assert summary.final_truth_aligned is False
    assert summary.final_false_consensus is True


def test_compute_run_summary_no_consensus_when_disagreement_high():
    initial = _row(-1)
    steps = [
        _row(t, mean_claim_belief_variance=0.5, fraction_truth_aligned=0.9)
        for t in range(5)
    ]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=0.0)

    assert summary.final_consensus is False
    assert summary.final_truth_aligned is True
    assert summary.final_false_consensus is False
