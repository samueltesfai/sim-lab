from __future__ import annotations

import math

from simlab.telemetry import TelemetryRow
from simlab.world import World


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


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
