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
    scene: Scene,
    claim_id: int,
    step_snapshot: Snapshot | dict | None = None,
    latest_metrics=None,
) -> ViewModel:
    # `latest_metrics` is intentionally unused for now (future telemetry wiring).
    _ = latest_metrics

    agent_beliefs = _step_get(step_snapshot, "agent_beliefs", {})
    beliefs = {a: agent_beliefs[a][claim_id] for a in agent_beliefs}

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
    mean = sum(vals) / len(vals) if vals else 0.0
    mn = min(vals) if vals else 0.0
    mx = max(vals) if vals else 0.0

    truth_bool = scene.truths.get(claim_id, False)

    return ViewModel(
        tick=_step_get(step_snapshot, "tick", 0),
        claim_id=claim_id,
        truth_bool=truth_bool,
        beliefs=beliefs,
        mem_counts=_step_get(step_snapshot, "agent_memory_sizes", {}),
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
