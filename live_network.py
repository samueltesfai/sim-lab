# viz/live_network.py
from __future__ import annotations

import math
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from sim import World, init_world

def _belief_to_gray(b: float) -> tuple[float, float, float]:
    """
    Map belief in [0,1] to grayscale RGB.
    0 -> black, 1 -> white.
    """
    b = max(0.0, min(1.0, b))
    return (b, b, b)

class LiveNetworkViz:
    """
    Live network visualization:
    - node color = belief (grayscale)
    - node size = out-degree (rough proxy for influence)
    - title shows tick + summary stats
    """
    def __init__(self, world: World, claim_id: int = 0, layout_seed: int = 0):
        self.world = world
        self.claim_id = claim_id

        # Build static graph structure once
        self.G = nx.DiGraph()
        self.G.add_nodes_from([a.id for a in world.agents])
        self.G.add_edges_from(world.edges)

        # Fixed layout for readability (doesn't jump each frame)
        self.pos = nx.spring_layout(self.G, seed=layout_seed)

        self.fig, self.ax = plt.subplots()
        self._initialized = False

    def _compute_stats(self):
        vals = [a.beliefs[self.claim_id] for a in self.world.agents]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(var)
        mn, mx = min(vals), max(vals)
        return mean, std, mn, mx

    def draw(self):
        # Pull beliefs
        beliefs = {a.id: a.beliefs[self.claim_id] for a in self.world.agents}

        # Node colors
        node_colors = [_belief_to_gray(beliefs[n]) for n in self.G.nodes()]

        # Node sizes by out-degree (scaled)
        degrees = dict(self.G.out_degree())
        node_sizes = [200 + 120 * degrees[n] for n in self.G.nodes()]

        self.ax.clear()

        # Draw edges first
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, alpha=0.25, arrows=False)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            ax=self.ax,
            node_color=node_colors,
            node_size=node_sizes,
            linewidths=1.0,
        )

        # Optional: labels (turn off if cluttered)
        # nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=8)

        mean, std, mn, mx = self._compute_stats()
        truth = 1.0 if self.world.truths.get(self.claim_id, False) else 0.0
        err = abs(truth - mean)

        self.ax.set_title(
            f"tick={self.world.tick}  "
            f"mean={mean:.3f} std={std:.3f} min={mn:.3f} max={mx:.3f}  "
            f"|truth-mean|={err:.3f}"
        )
        self.ax.axis("off")

        self.fig.canvas.draw_idle()

def run_live(world, steps: int = 500, claim_id: int = 0, draw_every: int = 1):
    """
    Simple live loop using plt.pause. Very easy to prototype.
    """
    viz = LiveNetworkViz(world, claim_id=claim_id)
    plt.ion()
    viz.draw()
    plt.show()

    for _ in range(steps):
        world.step()
        if world.tick % draw_every == 0:
            viz.draw()
            plt.pause(0.001)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=15)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--claim-id", type=int, default=0)
    parser.add_argument("--draw-every", type=int, default=1)
    args = parser.parse_args()

    world = init_world(num_agents=args.num_agents, rng_seed=args.rng_seed)
    run_live(world, steps=args.steps, claim_id=args.claim_id, draw_every=args.draw_every)
    