from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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


def _profile_features(world: World) -> dict[str, float | int]:
    num_agents = len(world.agents)
    features: dict[str, float | int] = {}
    for name, count in world.profile_counts.items():
        features[f"profile_count.{name}"] = count
        features[f"profile_fraction.{name}"] = count / num_agents if num_agents else 0.0
    return features


def _agent_parameter_features(world: World) -> dict[str, float]:
    parameter_series = {
        "agent_attention": [a.observation_attention for a in world.agents],
        "agent_bias": [a.observation_bias for a in world.agents],
        "agent_learning_rate": [a.learning_rate for a in world.agents],
        "agent_default_trust": [a.default_trust for a in world.agents],
        "agent_confidence_bound": [a.social_confidence_bound for a in world.agents],
        "agent_trust_update_rate": [a.social_trust_update_rate for a in world.agents],
    }

    features: dict[str, float] = {}
    for label, values in parameter_series.items():
        mean, std = _mean_std(values)
        features[f"{label}_mean"] = mean
        features[f"{label}_std"] = std
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
    world: World, initial_row: TelemetryRow
) -> dict[str, float | int]:
    """
    Extract scenario features describing conditions known before the run:
    graph structure, population/profile composition, agent parameter
    distributions, world observation/noise settings, and the initial belief
    state. ``initial_row`` must come from ``Telemetry.record_initial(world)``
    for the same world, so belief/trust/truth-alignment stats aren't
    recomputed here.
    """
    features: dict[str, float | int] = {
        "num_agents": len(world.agents),
        "num_claims": len(world.claims),
        "private_event_rate": world.private_event_rate,
        "global_event_rate": world.global_event_rate,
    }
    for memory_type, value in world.noise.items():
        features[f"noise.{memory_type.name}"] = value

    features.update(_graph_features(world))
    features.update(_profile_features(world))
    features.update(_agent_parameter_features(world))
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
    """
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
        mean_truth_error_auc=_trapezoidal_mean(truth_errors),
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
        final_false_consensus=final_consensus and not final_truth_aligned,
    )
