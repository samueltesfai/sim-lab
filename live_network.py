# viz/live_network.py
from __future__ import annotations

import math
from typing import Literal
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from sim import World, init_world



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
    
    @staticmethod
    def _belief_to_gray(b: float, invert: bool = False) -> tuple[float, float, float]:
        """
        Convert belief in [0, 1] to a grayscale color (r, g, b) where 0=black and 1=white.
        If invert=True, then 0=white and 1=black.
        """
        b = max(0.0, min(1.0, b))
        if invert:
            b = 1.0 - b
        return (b, b, b)
    
    @staticmethod
    def _color_truth_gradient(b: float, truth: bool) -> tuple[float, float, float]:
        """
        Map belief and truth alignment to a red-yellow-green gradient.
        - Red: Opposed (low alignment)
        - Yellow: Neutral (medium alignment)
        - Green: Aligned (high alignment)
        """
        b = max(0.0, min(1.0, b))
        t = 1.0 if truth else 0.0

        # Alignment in [-1, 1]
        alignment = (2.0 * b - 1.0) * (2.0 * t - 1.0)

        if alignment <= 0:
            # Opposed: Red to Yellow
            intensity = abs(alignment)
            return (1.0, intensity, 0.0)  # Red fades to Yellow
        else:
            # Aligned: Yellow to Green
            intensity = alignment
            return (1.0 - intensity, 1.0, 0.0)  # Yellow fades to Green
    
    @staticmethod
    def node_color(b: float, mode: Literal["gray", "invert_gray", "truth_rg"], truth: bool | None = None) -> tuple[float, float, float]:
        """
        :param b: belief in [0, 1]
        :type b: float
        :param truth: ground truth for the claim (True/False)
        :type truth: bool
        :param mode: color mode to use for visualization
        :type mode: Literal["gray", "invert_gray", "truth_rg"]
        :return: RGB color tuple in [0, 1]
        :rtype: tuple[float, float, float]
        """
        if mode == "gray":
            return LiveNetworkViz._belief_to_gray(b, invert=False)
        elif mode == "invert_gray":
            return LiveNetworkViz._belief_to_gray(b, invert=True)
        elif mode == "truth_rg":
            if truth is None:
                raise ValueError("truth must be provided for truth_rg mode")
            return LiveNetworkViz._color_truth_gradient(b, truth)
        else:
            raise ValueError(f"Unknown color mode: {mode}")

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
        node_colors = [self.node_color(beliefs[n], mode="truth_rg", truth=self.world.truths.get(self.claim_id, False)) for n in self.G.nodes()]

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

def run_live(world, steps: int = 500, claim_id: int = 0, draw_every: int = 1, layout_seed: int = 0, pause_time: float = 0.001):
    """
    Simple live loop using plt.pause. Very easy to prototype.
    """
    viz = LiveNetworkViz(world, claim_id=claim_id, layout_seed=layout_seed)
    plt.ion()
    viz.draw()
    plt.show()

    for _ in range(steps):
        world.step()
        if world.tick % draw_every == 0:
            viz.draw()
            plt.pause(pause_time)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=15)
    parser.add_argument("-s", "--rng-seed", type=int, default=42)
    parser.add_argument("-t", "--steps", type=int, default=500)
    parser.add_argument("-c", "--claim-id", type=int, default=0)
    parser.add_argument("-d", "--draw-every", type=int, default=1)
    parser.add_argument("-g", "--layout-seed", type=int, default=0)
    parser.add_argument("-p", "--pause-time", type=float, default=0.001)
    args = parser.parse_args()

    world = init_world(num_agents=args.num_agents, rng_seed=args.rng_seed)
    run_live(world, steps=args.steps, claim_id=args.claim_id, draw_every=args.draw_every, layout_seed=args.layout_seed, pause_time=args.pause_time)
    