from dataclasses import dataclass
import networkx as nx
import numpy as np
from sim import World


@dataclass(frozen=True)
class Scene:
    """
    A handy container for attributes related to graph data structure. Along with `build_scene` factory, helps isolate
    graph creation logic (i.e. testing different layouts).
    """

    G: nx.DiGraph
    pos: dict[int, tuple[float, float]]
    nodes: list[int]
    degrees: dict[int, int]
    sizes_base: np.ndarray


def build_scene(world: World, layout_seed: int = 0) -> Scene:
    G = nx.DiGraph()
    G.add_nodes_from([a.id for a in world.agents])
    G.add_edges_from(world.edges)

    pos = nx.spring_layout(G, seed=layout_seed)

    nodes = list(G.nodes())
    degrees = dict(G.out_degree())
    sizes_base = np.array(
        [200 + 120 * degrees.get(n, 0) for n in nodes],
        dtype=float,
    )

    return Scene(
        G=G,
        pos=pos,
        nodes=nodes,
        degrees=degrees,
        sizes_base=sizes_base,
    )
