import matplotlib.pyplot as plt
import numpy as np

from sim import World

from viz.scene import build_scene
from viz.view_model import compute_viewmodel
from viz.components.graph import BaseEdges, Nodes
from viz.components.overlays import RingOverlay, ActiveEdges
from viz.components.ui import HUDText, LegendComponent
from viz.components.interaction import HoverTooltip

class LiveNetworkViz:
    def __init__(self, world: World, claim_id: int = 0, layout_seed: int = 0):
        self.world = world
        self.claim_id = claim_id
        self.scene = build_scene(world, layout_seed=layout_seed)

        self.fig, self.ax = plt.subplots()
        self._initialized = False

        self.components = [
            BaseEdges(self.scene, alpha=0.15, lw=1.0, z=1),
            Nodes(self.scene, color_mode="gray", z=2),  # let Nodes pull nodes/pos/sizes from scene/vm
            ActiveEdges(self.scene, key="active_edges", z=3, rad=0.08, arrowsize=14),
            RingOverlay(self.scene, color="orange",      scale=1.08, key="heard_receivers", z=4),
            RingOverlay(self.scene, color="deepskyblue", scale=1.15, key="observed_ids",    z=5),
            RingOverlay(self.scene, color="magenta",     scale=1.25, key="verified_ids",    z=6),
            HUDText(self.scene, z=10),
            LegendComponent(self.scene, z=11),
            HoverTooltip(self.scene, z=12),
        ]

    def _init_artists(self):
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # optional: set limits once using scene.pos
        xy = np.array([self.scene.pos[n] for n in self.scene.nodes], dtype=float)
        pad = 0.15
        self.ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        self.ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)

        for comp in self.components:
            comp.add_to_canvas(self.ax, self.fig)
        self._initialized = True

    def draw(self):
        if not self._initialized:
            self._init_artists()

        vm = compute_viewmodel(self.world, self.scene, claim_id=self.claim_id)

        for comp in self.components:
            comp.update(vm)

        self.fig.canvas.draw_idle()


def run_live(world, steps: int = 500, claim_id: int = 0, draw_every: int = 1, layout_seed: int = 0, pause_time: float = 0.001):
    """
    Simple live loop using plt.pause.
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

