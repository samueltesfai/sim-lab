from dataclasses import dataclass
from simlab.viz.scene import Scene
from simlab.sim import World, Snapshot
from simlab.telemetry import TelemetryRow


@dataclass(frozen=True)
class ViewModel:
    tick: int
    claim_id: int
    truth_bool: bool
    beliefs: dict[int, float]
    agent_memory_sizes: dict[int, int]

    observed_ids: list[int]
    verified_ids: list[int]
    active_edges: list[tuple[int, int]]  # directed (sender, receiver)
    communicate_edges: list[tuple[int, int]]  # directed (sender, receiver)
    broadcast_edges: list[tuple[int, int]]  # directed (sender, receiver)
    heard_receivers: list[int]
    communicate_receivers: list[int]
    broadcast_receivers: list[int]

    stats: dict[str, float]  # mean/min/max
    pos: dict[int, tuple[float, float]]  # node positions


def compute_viewmodel(
    scene: Scene,
    claim_id: int,
    step_snapshot: Snapshot,
    telemetry_row: TelemetryRow | None = None,
) -> ViewModel:
    # `telemetry_row` is intentionally unused for now.
    _ = telemetry_row

    agent_beliefs = step_snapshot.agent_beliefs
    tick = step_snapshot.tick
    observed = step_snapshot.observed_ids
    verified = step_snapshot.verified_ids
    communicate_edges = step_snapshot.communicate_edges
    broadcast_edges = step_snapshot.broadcast_edges
    agent_memory_sizes = step_snapshot.agent_memory_sizes
    agent_claim_beliefs = { 
        # agent_id -> belief_value for the given claim
        agent_id: claim_beliefs.get(claim_id, 0.0)
        for agent_id, claim_beliefs in agent_beliefs.items()
    }

    active_edges = list(dict.fromkeys(communicate_edges + broadcast_edges))
    receivers = list({r for (_s, r) in active_edges})
    communicate_receivers = list({r for (_s, r) in communicate_edges})
    broadcast_receivers = list({r for (_s, r) in broadcast_edges})

    vals = list(agent_claim_beliefs.values())
    mean = sum(vals) / len(vals) if vals else 0.0
    mn = min(vals) if vals else 0.0
    mx = max(vals) if vals else 0.0

    truth_bool = scene.truths.get(claim_id, False)

    return ViewModel(
        tick=tick,
        claim_id=claim_id,
        truth_bool=truth_bool,
        beliefs=agent_claim_beliefs,
        agent_memory_sizes=agent_memory_sizes,
        observed_ids=observed,
        verified_ids=verified,
        active_edges=active_edges,
        communicate_edges=communicate_edges,
        broadcast_edges=broadcast_edges,
        heard_receivers=receivers,
        communicate_receivers=communicate_receivers,
        broadcast_receivers=broadcast_receivers,
        stats={"mean": mean, "min": mn, "max": mx},
        pos=scene.pos,
    )