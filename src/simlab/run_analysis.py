from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from simlab.config_schema import AgentProfile, SimConfig
from simlab.kernel_types import ActionType
from simlab.telemetry import TelemetryRow
from simlab.world import World


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def _percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; ``p`` in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(p * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def _trapezoidal_mean(values: Sequence[float]) -> float:
    """Time-averaged value under the curve (dx=1 per tick)."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    area = sum((values[i] + values[i + 1]) / 2 for i in range(len(values) - 1))
    return area / (len(values) - 1)


def _graph_features(world: World) -> dict[str, float | int]:
    agent_ids = [agent.id for agent in world.agents]
    num_nodes = len(agent_ids)
    out_degrees = [len(world.neighbors(agent_id)) for agent_id in agent_ids]
    num_edges = sum(out_degrees)

    max_possible_edges = num_nodes * (num_nodes - 1)
    edge_density = num_edges / max_possible_edges if max_possible_edges else 0.0
    mean_out_degree, std_out_degree = _mean_std([float(d) for d in out_degrees])
    fraction_isolated = (
        sum(1 for d in out_degrees if d == 0) / num_nodes if num_nodes else 0.0
    )

    return {
        "graph.num_nodes": num_nodes,
        "graph.num_edges": num_edges,
        "graph.edge_density": edge_density,
        "graph.mean_out_degree": mean_out_degree,
        "graph.std_out_degree": std_out_degree,
        "graph.min_out_degree": min(out_degrees) if out_degrees else 0,
        "graph.max_out_degree": max(out_degrees) if out_degrees else 0,
        "graph.fraction_isolated": fraction_isolated,
    }


def _weighted_mean_std(
    values_with_weights: list[tuple[float, int]],
) -> tuple[float, float]:
    """Population mean/std of a value repeated ``weight`` times, without
    materializing the repeated list -- used for per-profile parameters,
    where ``weight`` is the profile's agent count.
    """
    total = sum(weight for _, weight in values_with_weights)
    if not total:
        return 0.0, 0.0
    mean = sum(value * weight for value, weight in values_with_weights) / total
    var = (
        sum(weight * (value - mean) ** 2 for value, weight in values_with_weights)
        / total
    )
    return mean, math.sqrt(var)


def _profile_features(agent_profiles: list[AgentProfile]) -> dict[str, float | int]:
    num_agents = sum(profile.count for profile in agent_profiles)
    features: dict[str, float | int] = {}
    for profile in agent_profiles:
        features[f"profile_count.{profile.name}"] = profile.count
        features[f"profile_fraction.{profile.name}"] = (
            profile.count / num_agents if num_agents else 0.0
        )
    return features


_PARAMETER_PATHS: dict[str, tuple[str, ...]] = {
    "agent_attention": ("observation", "attention"),
    "agent_bias": ("observation", "bias"),
    "agent_learning_rate": ("learning", "rate"),
    "agent_observe_weight": ("learning", "observe_weight"),
    "agent_hear_weight": ("learning", "hear_weight"),
    "agent_verify_weight": ("learning", "verify_weight"),
    "agent_default_trust": ("trust", "default"),
    "agent_confidence_bound": ("social", "confidence_bound"),
    "agent_trust_update_rate": ("social", "trust_update_rate"),
}


def _get_path(profile: AgentProfile, path: tuple[str, ...]) -> float:
    value: Any = profile
    for key in path:
        value = getattr(value, key)
    return value


def _agent_parameter_features(
    agent_profiles: list[AgentProfile],
) -> dict[str, float | int]:
    """Agent parameter distributions derived directly from each profile's
    materialized settings -- every agent in a profile shares those settings
    exactly (no per-agent jitter), so this needs no constructed ``World``.

    Reports both population-wide marginal mean/std (weighted by profile
    count) and a per-profile snapshot. The marginals alone can't
    distinguish which parameter values are paired together on the same
    agents -- two profile assignments with identical per-parameter
    marginals can still behave differently depending on that pairing
    (e.g. attentive agents also learning fast vs. attentive agents
    learning slowly) -- so the per-profile snapshot preserves it.

    :param agent_profiles: One resolved settings object per profile, as
        returned by ``config.load_config(path).agent.profiles``
    :type agent_profiles: list[AgentProfile]
    :return: Population-wide ``{label}_mean``/``{label}_std`` plus
        ``agent_profile.<name>.<parameter>`` keys
    :rtype: dict[str, float | int]
    """
    features: dict[str, float | int] = {}

    for label, path in _PARAMETER_PATHS.items():
        mean, std = _weighted_mean_std(
            [(_get_path(p, path), p.count) for p in agent_profiles]
        )
        features[f"{label}_mean"] = mean
        features[f"{label}_std"] = std

    total_agents = sum(profile.count for profile in agent_profiles)
    features["agent_update_trust_on_rejection_fraction"] = (
        sum(
            profile.count
            for profile in agent_profiles
            if profile.social.update_trust_on_rejection
        )
        / total_agents
        if total_agents
        else 0.0
    )

    for action in ActionType:
        pref_mean, pref_std = _weighted_mean_std(
            [
                (profile.action_preference[action.name], profile.count)
                for profile in agent_profiles
            ]
        )
        cost_mean, cost_std = _weighted_mean_std(
            [
                (profile.action_cost[action.name], profile.count)
                for profile in agent_profiles
            ]
        )
        features[f"agent_action_preference.{action.name}_mean"] = pref_mean
        features[f"agent_action_preference.{action.name}_std"] = pref_std
        features[f"agent_action_cost.{action.name}_mean"] = cost_mean
        features[f"agent_action_cost.{action.name}_std"] = cost_std

    features.update(_per_profile_parameter_features(agent_profiles))

    return features


def _per_profile_parameter_features(
    agent_profiles: list[AgentProfile],
) -> dict[str, float | int]:
    """One parameter snapshot per profile, keyed by ``profile_name`` -- see
    ``_agent_parameter_features`` for why this is necessary alongside the
    population-wide marginals.

    Also records each profile's ``order_index`` -- ``world_from_config``
    assigns agent ids/``rng_seed``s sequentially in ``agent.profiles`` list
    order, so a reversed profile list builds a different simulation even
    though every other feature here is keyed by name or is an aggregate.
    Combined with ``profile_count.<name>`` (see ``_profile_features``),
    ``order_index`` is enough to reconstruct which agent id range each
    profile occupies.

    :param agent_profiles: One resolved settings object per profile, in
        ``agent.profiles`` order
    :type agent_profiles: list[AgentProfile]
    :return: ``agent_profile.<name>.<parameter>`` -> value, one set of keys
        per profile
    :rtype: dict[str, float | int]
    """
    features: dict[str, float | int] = {}
    for order_index, profile in enumerate(agent_profiles):
        prefix = f"agent_profile.{profile.name}"
        features[f"{prefix}.order_index"] = order_index
        for label, path in _PARAMETER_PATHS.items():
            key = label.removeprefix("agent_")
            features[f"{prefix}.{key}"] = _get_path(profile, path)
        features[f"{prefix}.update_trust_on_rejection"] = float(
            profile.social.update_trust_on_rejection
        )
        for action in ActionType:
            features[f"{prefix}.action_preference.{action.name}"] = (
                profile.action_preference[action.name]
            )
            features[f"{prefix}.action_cost.{action.name}"] = profile.action_cost[
                action.name
            ]

    return features


def _initial_state_features(initial_row: TelemetryRow) -> dict[str, float]:
    return {
        "initial.belief_mean": initial_row.belief_mean,
        "initial.belief_std": initial_row.belief_std,
        "initial.mean_abs_error_to_truth": initial_row.mean_abs_error_to_truth,
        "initial.fraction_truth_aligned": initial_row.fraction_truth_aligned,
        "initial.mean_claim_belief_variance": initial_row.mean_claim_belief_variance,
        "initial.fraction_confident_wrong": initial_row.fraction_confident_wrong,
        "initial.mean_trust": initial_row.mean_trust,
        "initial.trust_std": initial_row.trust_std,
    }


def extract_scenario_features(
    world: World, initial_row: TelemetryRow, resolved_config: SimConfig
) -> dict[str, float | int]:
    """
    Extract scenario features describing conditions known before the run:
    graph structure, population/profile composition, agent parameter
    distributions (both population-wide marginals and per-profile
    snapshots, since marginals alone can't distinguish which parameter
    values are paired on the same agents), world observation/noise
    settings, and the initial belief state.

    Everything except graph structure and initial state is a deterministic
    function of the config (no per-agent jitter is ever applied to a
    profile's settings), so those come from ``resolved_config`` rather than
    being re-derived from constructed ``Agent`` instances. Graph structure
    depends on the world's RNG draws at construction, and the initial
    belief state depends on each agent's own RNG, so those still require
    the constructed ``world``/``initial_row``.

    :param world: The constructed world, used for graph structure only
    :type world: World
    :param initial_row: Must come from ``Telemetry.record_initial(world)``
        for the same world, so belief/trust/truth-alignment stats aren't
        recomputed here
    :type initial_row: TelemetryRow
    :param resolved_config: The resolved, validated config the world was
        built from (see ``config.load_config``/``config.validate_config``)
    :type resolved_config: SimConfig
    :return: The scenario feature dict
    :rtype: dict[str, float | int]
    """
    world_settings = resolved_config.world
    agent_profiles = resolved_config.agent.profiles
    num_agents = sum(profile.count for profile in agent_profiles)

    features: dict[str, float | int] = {
        "num_agents": num_agents,
        "num_claims": len(world_settings.truths),
        "private_event_rate": world_settings.observation.private_event_rate,
        "global_event_rate": world_settings.observation.global_event_rate,
    }
    for memory_type, value in world_settings.noise.items():
        features[f"noise.{memory_type}"] = value

    features.update(_graph_features(world))
    features.update(_profile_features(agent_profiles))
    features.update(_agent_parameter_features(agent_profiles))
    features.update(_initial_state_features(initial_row))

    return features


@dataclass(frozen=True, slots=True)
class RunSummary:
    # Initial state
    initial_mean_truth_error: float
    initial_fraction_truth_aligned: float
    initial_mean_claim_belief_variance: float
    initial_fraction_confident_wrong: float

    # Final state
    final_mean_truth_error: float
    final_fraction_truth_aligned: float
    final_mean_claim_belief_variance: float
    final_fraction_confident_wrong: float
    final_mean_trust: float
    final_trust_std: float

    # Trajectory
    min_mean_truth_error: float
    max_mean_truth_error: float
    mean_truth_error_auc: float
    mean_belief_volatility: float
    max_belief_volatility: float
    convergence_tick: int | None

    # Activity totals
    total_observation_events: int
    total_observations: int
    total_verifications: int
    total_communicate_edges: int
    total_broadcast_edges: int
    total_memory_processing_agent_ticks: int
    total_belief_updating_agent_ticks: int
    total_trust_updating_agent_ticks: int

    # Runtime
    total_runtime_ms: float
    mean_step_runtime_ms: float
    p95_step_runtime_ms: float

    # Labels
    converged: bool
    final_consensus: bool
    final_truth_aligned: bool
    final_false_consensus: bool


def find_convergence_tick(
    rows: Sequence[TelemetryRow],
    *,
    delta_threshold: float,
    disagreement_threshold: float,
    window: int,
) -> int | None:
    """
    Find the tick at which the trajectory settles into a stable window.

    Convergence begins at tick ``t`` when, for the next ``window`` recorded
    ticks (starting at ``t``), mean absolute belief movement stays at or
    below ``delta_threshold`` and mean per-claim belief variance stays at or
    below ``disagreement_threshold``. A single quiet tick does not count;
    stability must hold for the full window. Returns ``None`` if no such
    window exists, including when fewer than ``window`` ticks were recorded.

    These are analysis thresholds for detecting stability after the fact,
    not kernel parameters -- they say nothing about *why* a trajectory
    stabilized (agreement vs. correctness).

    :raises ValueError: if ``window`` is less than 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    stepped = [row for row in rows if row.tick >= 0]
    if len(stepped) < window:
        return None

    is_stable = [
        row.mean_abs_delta <= delta_threshold
        and row.mean_claim_belief_variance <= disagreement_threshold
        for row in stepped
    ]

    consecutive = 0
    for i, stable in enumerate(is_stable):
        consecutive = consecutive + 1 if stable else 0
        if consecutive == window:
            return stepped[i - window + 1].tick

    return None


def compute_run_summary(
    telemetry: Sequence[TelemetryRow],
    *,
    total_runtime_ms: float,
    convergence_delta_threshold: float = 0.001,
    convergence_variance_threshold: float = 0.0025,
    convergence_window: int = 20,
    truth_alignment_threshold: float = 0.8,
) -> RunSummary:
    """
    Aggregate a full telemetry trajectory (initial row plus one row per
    step) into a single run-level record: initial/final state, trajectory
    shape, activity totals, runtime, and a conservative set of mechanically
    clear labels.

    ``truth_alignment_threshold`` and the convergence thresholds are
    analysis parameters, not kernel parameters -- they define how this
    summary interprets a trajectory, not how the simulation behaves.
    """
    if not telemetry:
        raise ValueError("telemetry must contain at least the initial row")

    initial_row = telemetry[0]
    stepped_rows = [row for row in telemetry if row.tick >= 0]
    final_row = stepped_rows[-1] if stepped_rows else initial_row

    truth_errors = [row.mean_abs_error_to_truth for row in stepped_rows]
    # The trapezoidal average needs the initial-to-tick-0 interval too, or a
    # one-step run's error change (e.g. 0.9 -> 0.1) is reported as the final
    # value (0.1) instead of the interval average (0.5).
    truth_error_trajectory = [initial_row.mean_abs_error_to_truth, *truth_errors]
    deltas = [row.mean_abs_delta for row in stepped_rows]
    step_runtimes = [
        row.step_runtime_ms for row in stepped_rows if row.step_runtime_ms is not None
    ]

    convergence_tick = find_convergence_tick(
        telemetry,
        delta_threshold=convergence_delta_threshold,
        disagreement_threshold=convergence_variance_threshold,
        window=convergence_window,
    )

    final_consensus = (
        final_row.mean_claim_belief_variance <= convergence_variance_threshold
    )
    final_truth_aligned = final_row.fraction_truth_aligned >= truth_alignment_threshold
    # `not final_truth_aligned` isn't evidence of a *wrong* belief -- e.g.
    # agents converged near 0.5 are uncertain, not confidently wrong.
    final_confidently_wrong = (
        final_row.fraction_confident_wrong >= truth_alignment_threshold
    )

    return RunSummary(
        initial_mean_truth_error=initial_row.mean_abs_error_to_truth,
        initial_fraction_truth_aligned=initial_row.fraction_truth_aligned,
        initial_mean_claim_belief_variance=initial_row.mean_claim_belief_variance,
        initial_fraction_confident_wrong=initial_row.fraction_confident_wrong,
        final_mean_truth_error=final_row.mean_abs_error_to_truth,
        final_fraction_truth_aligned=final_row.fraction_truth_aligned,
        final_mean_claim_belief_variance=final_row.mean_claim_belief_variance,
        final_fraction_confident_wrong=final_row.fraction_confident_wrong,
        final_mean_trust=final_row.mean_trust,
        final_trust_std=final_row.trust_std,
        min_mean_truth_error=(
            min(truth_errors) if truth_errors else initial_row.mean_abs_error_to_truth
        ),
        max_mean_truth_error=(
            max(truth_errors) if truth_errors else initial_row.mean_abs_error_to_truth
        ),
        mean_truth_error_auc=_trapezoidal_mean(truth_error_trajectory),
        mean_belief_volatility=_mean(deltas),
        max_belief_volatility=max(deltas) if deltas else 0.0,
        convergence_tick=convergence_tick,
        total_observation_events=sum(
            row.num_observation_events for row in stepped_rows
        ),
        total_observations=sum(row.num_observations for row in stepped_rows),
        total_verifications=sum(row.num_verifications for row in stepped_rows),
        total_communicate_edges=sum(row.num_communicate_edges for row in stepped_rows),
        total_broadcast_edges=sum(row.num_broadcast_edges for row in stepped_rows),
        total_memory_processing_agent_ticks=sum(
            row.num_memory_processing_agents for row in stepped_rows
        ),
        total_belief_updating_agent_ticks=sum(
            row.num_belief_updating_agents for row in stepped_rows
        ),
        total_trust_updating_agent_ticks=sum(
            row.num_trust_updating_agents for row in stepped_rows
        ),
        total_runtime_ms=total_runtime_ms,
        mean_step_runtime_ms=_mean(step_runtimes),
        p95_step_runtime_ms=_percentile(step_runtimes, 0.95),
        converged=convergence_tick is not None,
        final_consensus=final_consensus,
        final_truth_aligned=final_truth_aligned,
        final_false_consensus=final_consensus and final_confidently_wrong,
    )
