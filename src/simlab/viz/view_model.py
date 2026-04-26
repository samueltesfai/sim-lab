from dataclasses import dataclass
from simlab.viz.scene import Scene
from simlab.sim import World, Snapshot


@dataclass(frozen=True)
class ViewModel:
    tick: int
    claim_id: int
    truth_bool: bool
    beliefs: dict[int, float]
    mem_counts: dict[int, int]

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


def _step_get(step_snapshot: Snapshot | dict | None, key: str, default):
    if step_snapshot is None:
        return default
    if isinstance(step_snapshot, dict):
        return step_snapshot.get(key, default)
    return getattr(step_snapshot, key, default)


def compute_viewmodel(
    world: World,
    scene: Scene,
    claim_id: int,
    step_snapshot: Snapshot | dict | None = None,
    latest_metrics=None,
) -> ViewModel:
    # `latest_metrics` is intentionally unused for now (future telemetry wiring).
    _ = latest_metrics

    # Get beliefs for the selected claim_id from world state
    # Optionally, could also get from snapshot.beliefs if needed
    beliefs = {a.id: a.beliefs[claim_id] for a in world.agents}
    mem_counts = {a.id: len(a.memory) for a in world.agents}

    observed = _step_get(step_snapshot, "observed_ids", [])
    verified = _step_get(step_snapshot, "verified_ids", [])

    communicate_edges = _step_get(step_snapshot, "communicate_edges", [])
    broadcast_edges = _step_get(step_snapshot, "broadcast_edges", [])

    # directed union, deduplicated while preserving direction
    active_edges = list(dict.fromkeys(communicate_edges + broadcast_edges))
    receivers = list({r for (_s, r) in active_edges})
    communicate_receivers = list({r for (_s, r) in communicate_edges})
    broadcast_receivers = list({r for (_s, r) in broadcast_edges})

    vals = list(beliefs.values())
    mean = sum(vals) / len(vals)
    mn, mx = min(vals), max(vals)

    truth_bool = world.truths.get(claim_id, False)

    return ViewModel(
        tick=world.tick,
        claim_id=claim_id,
        truth_bool=truth_bool,
        beliefs=beliefs,
        mem_counts=mem_counts,
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
