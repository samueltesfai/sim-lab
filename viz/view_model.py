from dataclasses import dataclass
from viz.scene import Scene
from sim import World


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
    heard_receivers: list[int]

    stats: dict[str, float]  # mean/min/max
    pos: dict[int, tuple[float, float]]  # node positions


def compute_viewmodel(world: World, scene: Scene, claim_id: int) -> ViewModel:
    ls = world.last_step or {}

    beliefs = {a.id: a.beliefs[claim_id] for a in world.agents}
    mem_counts = {a.id: len(a.memory) for a in world.agents}

    observed = ls.get("observed_ids", [])
    verified = ls.get("verified_ids", [])

    communicate_edges = ls.get("communicate_edges", [])
    broadcast_edges = ls.get("broadcast_edges", [])

    # directed union, deduplicated while preserving direction
    active_edges = list(dict.fromkeys(communicate_edges + broadcast_edges))
    receivers = list({r for (_s, r) in active_edges})

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
        heard_receivers=receivers,
        stats={"mean": mean, "min": mn, "max": mx},
        pos=scene.pos,
    )
