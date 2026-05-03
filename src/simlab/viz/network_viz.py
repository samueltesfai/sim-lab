import matplotlib.pyplot as plt
import numpy as np
import time

from simlab.sim import World, Snapshot
from simlab.telemetry import Telemetry, TelemetryRow

from simlab.viz.scene import build_scene
from simlab.viz.view_model import compute_viewmodel
from simlab.viz.components.graph import BaseEdges, Nodes
from simlab.viz.components.overlays import RingOverlay, ActiveEdges
from simlab.viz.components.ui import HUDText, LegendComponent
from simlab.viz.components.interaction import HoverTooltip


class NetworkViz:
    def __init__(self, world: World, claim_id: int = 0, layout_seed: int = 0):
        self.claim_id = claim_id
        self.scene = build_scene(world, layout_seed=layout_seed)

        self.fig, self.ax = plt.subplots()
        self._initialized = False

        self.components = [
            BaseEdges(self.scene, alpha=0.15, lw=1.0, z=1),
            Nodes(
                self.scene, color_mode="gray", z=2
            ),  # let Nodes pull nodes/pos/sizes from scene/vm
            ActiveEdges(
                self.scene,
                key="communicate_edges",
                z=3,
                rad=0.08,
                arrowsize=14,
                color="darkorange",
            ),
            ActiveEdges(
                self.scene,
                key="broadcast_edges",
                z=3,
                rad=0.08,
                arrowsize=14,
                color="slateblue",
            ),
            RingOverlay(
                self.scene,
                color="cornflowerblue",
                scale=1.08,
                key="broadcast_receivers",
                z=4,
            ),
            RingOverlay(
                self.scene, color="orange", scale=1.08, key="communicate_receivers", z=4
            ),
            RingOverlay(
                self.scene, color="limegreen", scale=1.15, key="observed_ids", z=5
            ),
            RingOverlay(
                self.scene, color="magenta", scale=1.25, key="verified_ids", z=6
            ),
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

    def draw(self, snapshot: Snapshot, telemetry_row: TelemetryRow | None = None):
        if not self._initialized:
            self._init_artists()

        vm = compute_viewmodel(
            self.scene,
            claim_id=self.claim_id,
            step_snapshot=snapshot,
            telemetry_row=telemetry_row,
        )

        for comp in self.components:
            comp.update(vm)

        self.fig.canvas.draw_idle()


def run_viz(
    world,
    steps: int = 500,
    claim_id: int = 0,
    draw_every: int = 1,
    layout_seed: int = 0,
    pause_time: float = 0.001,
    telemetry: Telemetry | None = None,
    log_every: int = 10,
):
    """
    Simple visualization loop using plt.pause.
    """
    if log_every <= 0:
        raise ValueError("log_every must be >= 1")
    if draw_every <= 0:
        raise ValueError("draw_every must be >= 1")

    viz = NetworkViz(world, claim_id=claim_id, layout_seed=layout_seed)
    plt.ion()
    plt.show()

    # Create telemetry if not provided
    if telemetry is None:
        telemetry = Telemetry()

    # Record initial state and log
    print(telemetry.record_initial(world).format_cli())

    for i in range(steps):
        start_time = time.perf_counter()
        snapshot = world.step()
        end_time = time.perf_counter()
        step_runtime_ms = (end_time - start_time) * 1000

        row = telemetry.record(snapshot, world, step_runtime_ms=step_runtime_ms)

        if log_every and (i + 1) % log_every == 0:
            print(row.format_cli())

        if snapshot.tick % draw_every == 0:
            viz.draw(snapshot=snapshot, telemetry_row=row)
            plt.pause(pause_time)

    plt.ioff()
    plt.show()
