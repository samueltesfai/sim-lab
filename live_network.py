# viz/live_network.py
from __future__ import annotations

import math
from typing import Literal
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D

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
        # beliefs
        vals_by_id = {a.id: a.beliefs[self.claim_id] for a in self.world.agents}
        vals = list(vals_by_id.values())
        n = len(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(var)
        mn, mx = min(vals), max(vals)

        truth_bool = self.world.truths.get(self.claim_id, False)
        truth = 1.0 if truth_bool else 0.0
        err = abs(truth - mean)

        # last step (safe on tick 0)
        ls = self.world.last_step or {}
        observed_ids = ls.get("observed_ids", [])
        verified_ids = ls.get("verified_ids", [])
        heard = ls.get("heard_edges", [])

        heard_edges = [(s, r) for (s, r, cid) in heard if cid == self.claim_id]
        n_obs = len(observed_ids)
        n_ver = len(verified_ids)
        n_hear = len(heard_edges)

        # movers (only if step provides before/after)
        before = ls.get("belief_before", {})
        after = ls.get("belief_after", vals_by_id)

        movers = []
        for aid, b0 in before.items():
            b1 = after.get(aid, b0)
            movers.append((aid, b1 - b0))
        movers.sort(key=lambda t: abs(t[1]), reverse=True)
        top_movers = movers[:3]

        # quick “mass near extremes” metrics (optional but helpful)
        frac_low = sum(v < 0.2 for v in vals) / n
        frac_high = sum(v > 0.8 for v in vals) / n

        return {
            "n": n,
            "mean": mean,
            "std": std,
            "min": mn,
            "max": mx,
            "truth": truth,
            "truth_bool": truth_bool,
            "err": err,
            "observed_ids": observed_ids,
            "verified_ids": verified_ids,
            "heard_edges": heard_edges,
            "n_obs": n_obs,
            "n_ver": n_ver,
            "n_hear": n_hear,
            "top_movers": top_movers,
            "frac_low": frac_low,
            "frac_high": frac_high,
        }


    def draw(self):
        # --- collect beliefs once ---
        beliefs = {a.id: a.beliefs[self.claim_id] for a in self.world.agents}

        # --- colors / sizes ---
        truth_bool = self.world.truths.get(self.claim_id, False)
        node_colors = [
            self.node_color(beliefs[n], mode="gray", truth=truth_bool)
            for n in self.G.nodes()
        ]

        degrees = dict(self.G.out_degree())
        node_sizes = [200 + 120 * degrees.get(n, 0) for n in self.G.nodes()]

        # --- stats + per-tick annotations (from last_step) ---
        stats = self._compute_stats()

        observed_ids = stats["observed_ids"]
        verified_ids = stats["verified_ids"]
        active_edges = stats["heard_edges"]  # already filtered to claim_id in _compute_stats

        # optional: highlight receivers (useful to see “who got influenced” this tick)
        heard_receivers = list({r for (_s, r) in active_edges})

        # --- clear frame ---
        self.ax.clear()

        # --- base edges ---
        nx.draw_networkx_edges(
            self.G, self.pos, ax=self.ax,
            alpha=0.20, arrows=False
        )

        # --- active edges overlay (communication this tick) ---
        if active_edges:
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax,
                edgelist=active_edges,
                alpha=0.85, width=2.5, arrows=False
            )

        # --- base nodes ---
        nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax,
            node_color=node_colors,
            node_size=node_sizes,
            linewidths=0.8,
        )

        # --- receiver ring (optional) ---
        if heard_receivers:
            nx.draw_networkx_nodes(
                self.G, self.pos, ax=self.ax,
                nodelist=heard_receivers,
                node_color="none",
                edgecolors="orange",
                linewidths=1.8,
                node_size=[(200 + 120 * degrees.get(n, 0)) * 1.08 for n in heard_receivers],
            )

        # --- observed ring ---
        if observed_ids:
            nx.draw_networkx_nodes(
                self.G, self.pos, ax=self.ax,
                nodelist=observed_ids,
                node_color="none",
                edgecolors="deepskyblue",
                linewidths=2.5,
                node_size=[(200 + 120 * degrees.get(n, 0)) * 1.15 for n in observed_ids],
            )

        # --- verified ring ---
        if verified_ids:
            nx.draw_networkx_nodes(
                self.G, self.pos, ax=self.ax,
                nodelist=verified_ids,
                node_color="none",
                edgecolors="magenta",
                linewidths=2.5,
                node_size=[(200 + 120 * degrees.get(n, 0)) * 1.25 for n in verified_ids],
            )

        # --- HUD text ---
        movers_str = ", ".join([f"{aid}:{db:+.3f}" for aid, db in stats["top_movers"]])
        hud = (
            f"events: hear={stats['n_hear']} obs={stats['n_obs']} ver={stats['n_ver']}\n"
            f"extremes: <0.2={stats['frac_low']:.2f}  >0.8={stats['frac_high']:.2f}\n"
            f"top movers: {movers_str}"
        )
        self.ax.text(
            0.02, 0.98, hud,
            transform=self.ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
        )

        # --- legend (rings) ---
        handles = [
            Line2D([0], [0], marker="o", color="w", label="observe",
                markerfacecolor="none", markeredgecolor="deepskyblue",
                markersize=10, linewidth=0),
            Line2D([0], [0], marker="o", color="w", label="verify",
                markerfacecolor="none", markeredgecolor="magenta",
                markersize=10, linewidth=0),
            Line2D([0], [0], marker="o", color="w", label="heard (recv)",
                markerfacecolor="none", markeredgecolor="orange",
                markersize=10, linewidth=0),
        ]
        self.ax.legend(handles=handles, loc="lower left", framealpha=0.7)

        # --- title ---
        self.ax.set_title(
            f"tick={self.world.tick}  "
            f"mean={stats['mean']:.3f} std={stats['std']:.3f} "
            f"min={stats['min']:.3f} max={stats['max']:.3f}  "
            f"|truth-mean|={stats['err']:.3f}"
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
    