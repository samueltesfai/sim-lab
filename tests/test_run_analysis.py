import pytest

from simlab.config import validate_config, world_from_config
from simlab.config_schema import SimConfig
from simlab.run_analysis import (
    _agent_parameter_features,
    _graph_features,
    _profile_features,
    compute_run_summary,
    extract_scenario_features,
    find_convergence_tick,
)
from simlab.telemetry import Telemetry, TelemetryRow
from simlab.world import World

_DEFAULT_TRUTHS = {0: True, 1: False}


def _build_scenario(
    profiles: list[dict], *, truths: dict | None = None
) -> tuple[World, SimConfig]:
    """Build a World plus its resolved config from a list of profile dicts
    (``{"name":, "count":, **settings_overrides}``), exercising the same
    config -> validate_config -> world_from_config pipeline that
    extract_scenario_features's ``resolved_config`` argument comes from in
    production.
    """
    cfg_dict = {
        "world": {
            "rng_seed": 1,
            "truths": truths or _DEFAULT_TRUTHS,
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
            "observation": {"private_event_rate": 0.0, "global_event_rate": 0.0},
        },
        "agent": {"defaults": {}, "profiles": profiles},
    }
    cfg = validate_config(cfg_dict)
    return world_from_config(cfg), cfg


# ---------------------------------------------------------------------------
# extract_scenario_features
# ---------------------------------------------------------------------------


def test_extract_scenario_features_population_and_world_settings():
    world, resolved_config = _build_scenario(
        [
            {"name": "default", "count": 2},
            {"name": "skeptic", "count": 1, "trust": {"default": 0.1}},
        ]
    )
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row, resolved_config)

    assert features["num_agents"] == 3
    assert features["num_claims"] == 2
    assert features["private_event_rate"] == world.private_event_rate
    assert features["global_event_rate"] == world.global_event_rate
    assert features["profile_count.default"] == 2
    assert features["profile_count.skeptic"] == 1
    assert features["profile_fraction.default"] == pytest.approx(2 / 3)
    assert features["profile_fraction.skeptic"] == pytest.approx(1 / 3)


def test_extract_scenario_features_graph_stats_match_network():
    world, resolved_config = _build_scenario([{"name": "default", "count": 6}])
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row, resolved_config)

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
    world, resolved_config = _build_scenario(
        [
            {"name": "low", "count": 1, "observation": {"attention": 0.2}},
            {"name": "high", "count": 1, "observation": {"attention": 0.8}},
        ]
    )
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row, resolved_config)

    assert features["agent_attention_mean"] == pytest.approx(0.5)
    assert features["agent_attention_std"] == pytest.approx(0.3)


def test_extract_scenario_features_includes_all_behavior_driving_params():
    """Channel weights, action preferences/costs, and update_trust_on_rejection
    must be captured -- two scenarios differing only in these would otherwise
    look identical in scenario features despite behaving differently."""
    world, resolved_config = _build_scenario(
        [
            {
                "name": "vocal",
                "count": 1,
                "learning": {
                    "observe_weight": 0.9,
                    "hear_weight": 0.1,
                    "verify_weight": 0.2,
                },
                "social": {"update_trust_on_rejection": True},
                "action_preference": {"VERIFY": 1.0},
                "action_cost": {"VERIFY": 0.1},
            },
            {
                "name": "quiet",
                "count": 1,
                "learning": {
                    "observe_weight": 0.1,
                    "hear_weight": 0.9,
                    "verify_weight": 0.8,
                },
                "social": {"update_trust_on_rejection": False},
                "action_preference": {"VERIFY": 0.0},
                "action_cost": {"VERIFY": 0.9},
            },
        ]
    )
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row, resolved_config)

    assert features["agent_observe_weight_mean"] == pytest.approx(0.5)
    assert features["agent_hear_weight_mean"] == pytest.approx(0.5)
    assert features["agent_verify_weight_mean"] == pytest.approx(0.5)
    assert features["agent_update_trust_on_rejection_fraction"] == pytest.approx(0.5)
    assert features["agent_action_preference.VERIFY_mean"] == pytest.approx(0.5)
    assert features["agent_action_cost.VERIFY_mean"] == pytest.approx(0.5)


def test_extract_scenario_features_per_profile_distinguishes_parameter_pairing():
    """Population-wide mean/std alone can't tell apart which parameter
    values are paired on the same agent: two profile assignments with
    identical per-parameter marginals must still produce different
    per-profile features if the pairing differs."""

    def build(attentive_learning_rate, distracted_learning_rate):
        return _build_scenario(
            [
                {
                    "name": "attentive",
                    "count": 1,
                    "observation": {"attention": 0.9},
                    "learning": {"rate": attentive_learning_rate},
                },
                {
                    "name": "distracted",
                    "count": 1,
                    "observation": {"attention": 0.1},
                    "learning": {"rate": distracted_learning_rate},
                },
            ]
        )

    world_paired, resolved_paired = build(0.9, 0.1)
    world_swapped, resolved_swapped = build(0.1, 0.9)
    telemetry = Telemetry()
    row_paired = telemetry.record_initial(world_paired)
    row_swapped = telemetry.record_initial(world_swapped)

    features_paired = extract_scenario_features(
        world_paired, row_paired, resolved_paired
    )
    features_swapped = extract_scenario_features(
        world_swapped, row_swapped, resolved_swapped
    )

    # Marginals are identical -- this is exactly what makes them insufficient.
    assert features_paired["agent_learning_rate_mean"] == pytest.approx(
        features_swapped["agent_learning_rate_mean"]
    )
    assert features_paired["agent_learning_rate_std"] == pytest.approx(
        features_swapped["agent_learning_rate_std"]
    )

    # Per-profile features must still tell the two scenarios apart.
    assert features_paired["agent_profile.attentive.learning_rate"] == pytest.approx(
        0.9
    )
    assert features_paired["agent_profile.distracted.learning_rate"] == pytest.approx(
        0.1
    )
    assert features_swapped["agent_profile.attentive.learning_rate"] == pytest.approx(
        0.1
    )
    assert features_swapped["agent_profile.distracted.learning_rate"] == pytest.approx(
        0.9
    )


def test_extract_scenario_features_encodes_profile_order():
    """world_from_config assigns agent ids (and therefore each agent's
    rng_seed and position in the network-generation RNG stream) sequentially
    in agent.profiles list order, so two configs with the same named
    profiles/counts/settings but a reversed profile list build genuinely
    different simulations. Confirmed directly before this feature existed:
    such a pair produced byte-identical scenario feature dicts despite the
    different per-agent id/rng_seed assignment -- order_index (combined with
    the already-recorded profile_count.<name>) is enough to tell them apart
    and reconstruct which agent id range each profile occupies."""

    def build(profiles):
        return _build_scenario(profiles)

    profile_a = {"name": "a", "count": 3, "learning": {"rate": 0.9}}
    profile_b = {"name": "b", "count": 3, "learning": {"rate": 0.1}}

    world_ab, resolved_ab = build([profile_a, profile_b])
    world_ba, resolved_ba = build([profile_b, profile_a])
    telemetry = Telemetry()
    row_ab = telemetry.record_initial(world_ab)
    row_ba = telemetry.record_initial(world_ba)

    features_ab = extract_scenario_features(world_ab, row_ab, resolved_ab)
    features_ba = extract_scenario_features(world_ba, row_ba, resolved_ba)

    assert features_ab["agent_profile.a.order_index"] == 0
    assert features_ab["agent_profile.b.order_index"] == 1
    assert features_ba["agent_profile.a.order_index"] == 1
    assert features_ba["agent_profile.b.order_index"] == 0
    assert features_ab != features_ba


def test_extract_scenario_features_initial_state_matches_telemetry_row():
    world, resolved_config = _build_scenario([{"name": "default", "count": 4}])
    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)

    features = extract_scenario_features(world, initial_row, resolved_config)

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


def test_graph_features_no_agents_does_not_crash():
    """A World with no agents is still directly constructible (unlike a
    SimConfig with no profiles, which the schema disallows -- agent.profiles
    has always required at least one entry)."""
    world = World.from_dict([], {"truths": {0: True}, "rng_seed": 1})

    features = _graph_features(world)

    assert features["graph.num_nodes"] == 0
    assert features["graph.edge_density"] == 0.0
    assert features["graph.fraction_isolated"] == 0.0


def test_agent_parameter_features_empty_profiles_does_not_crash():
    """The population-wide aggregation helpers must not divide by zero when
    given no profiles -- unreachable via a real config (agent.profiles is
    always non-empty), but a defensive property of the aggregation math
    worth guarding directly."""
    assert _profile_features([]) == {}

    features = _agent_parameter_features([])

    assert features["agent_attention_mean"] == 0.0
    assert features["agent_attention_std"] == 0.0
    assert features["agent_update_trust_on_rejection_fraction"] == 0.0


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


@pytest.mark.parametrize("bad_window", [0, -1])
def test_find_convergence_tick_rejects_nonpositive_window(bad_window):
    """window=0 previously let an unstable first row satisfy
    consecutive == window on the first iteration, indexing past the end of a
    single-row trajectory instead of raising."""
    rows = [_row(0, mean_abs_delta=0.5, mean_claim_belief_variance=0.5)]

    with pytest.raises(ValueError, match="window must be >= 1"):
        find_convergence_tick(
            rows,
            delta_threshold=0.001,
            disagreement_threshold=0.0025,
            window=bad_window,
        )


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


def test_compute_run_summary_auc_includes_initial_to_first_step_interval():
    # A one-step run going from 0.9 to 0.1 spends the whole tick traversing
    # that interval, so the time-average over it is 0.5, not the tick-0
    # value (0.1) that a fix excluding the initial row would produce.
    initial = _row(-1, mean_abs_error_to_truth=0.9)
    steps = [_row(0, mean_abs_error_to_truth=0.1)]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=0.0)

    assert summary.mean_truth_error_auc == pytest.approx(0.5)


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
            fraction_confident_wrong=1.0,
        )
        for t in range(25)
    ]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=0.0)

    assert summary.final_consensus is True
    assert summary.final_truth_aligned is False
    assert summary.final_false_consensus is True


def test_compute_run_summary_no_false_consensus_when_converged_but_uncertain():
    """Agents converging near 0.5 (mutual agreement that they don't know)
    have low belief variance (consensus) and low truth alignment, but no
    fraction_confident_wrong -- not a false consensus, just genuine
    uncertainty. `not final_truth_aligned` alone can't distinguish this from
    a confidently-held wrong belief; fraction_confident_wrong can."""
    initial = _row(-1)
    steps = [
        _row(
            t,
            mean_abs_delta=0.0005,
            mean_claim_belief_variance=0.001,
            fraction_truth_aligned=0.0,
            fraction_confident_wrong=0.0,
        )
        for t in range(25)
    ]

    summary = compute_run_summary([initial, *steps], total_runtime_ms=0.0)

    assert summary.final_consensus is True
    assert summary.final_truth_aligned is False
    assert summary.final_false_consensus is False


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
