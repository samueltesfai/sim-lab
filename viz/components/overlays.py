from viz.components.base import _VizComponent
from viz.scene import Scene
from viz.view_model import ViewModel
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.axes import Axes
import numpy as np
from typing import Literal

class RingOverlay(_VizComponent):
    def __init__(self, scene: Scene, color: str, scale: float, key: Literal["observed_ids", "verified_ids", "heard_receivers"], lw: float = 2.0, z: int = 5):
        super().__init__(scene, z=z)
        self.color = color
        self.scale = scale
        self.key = key
        self.lw = lw
        self.sc = None

    def add_to_canvas(self, ax: Axes, fig: plt.Figure) -> None:
        self.sc = ax.scatter(
            [], [],
            s=[],
            facecolors="none",
            edgecolors=self.color,
            linewidths=self.lw,
            zorder=self.z,
        )

    def update(self, vm: ViewModel) -> None:
        if self.sc is None:
            return

        nodelist = getattr(vm, self.key)
        pos = vm.pos

        if not nodelist:
            self.sc.set_offsets(np.empty((0, 2)))
            self.sc.set_sizes(np.array([]))
            return

        pts = np.array([pos[n] for n in nodelist], dtype=float)
        sizes = np.array(
            [(200 + 120 * self.scene.degrees.get(n, 0)) * self.scale for n in nodelist],
            dtype=float
        )

        self.sc.set_offsets(pts)
        self.sc.set_sizes(sizes)


class ActiveEdges(_VizComponent):
    """
    Draw directed arrows for edges active this tick.

    Expects in vm:
      - active_edges: list[tuple[int, int]]  (sender, receiver)
      - pos: dict[int, tuple[float, float]]  node -> (x, y)
    """
    def __init__(
        self,
        scene: Scene,
        key: str = "active_edges",
        alpha: float = 0.9,
        lw: float = 2.0,
        arrowsize: float = 14.0,
        rad: float = 0.08,
        color: str = "black",
        z: int = 3,
        curve_bidirectional: bool = True,
    ):
        super().__init__(scene, z=z)
        self.key = key
        self.alpha = alpha
        self.lw = lw
        self.arrowsize = arrowsize
        self.rad = rad
        self.color = color
        self.curve_bidirectional = curve_bidirectional

        self._ax: Axes | None = None
        self._patches: list[FancyArrowPatch] = []

    def add_to_canvas(self, ax: Axes, fig: plt.Figure):
        # Remember the axes; we create/remove arrow patches each frame.
        self._ax = ax

    def _clear(self):
        for p in self._patches:
            try:
                p.remove()
            except Exception:
                pass
        self._patches.clear()

    @staticmethod
    def _vm_get(vm, key, default=None):
        # supports dict, namedtuple, dataclass-ish
        if isinstance(vm, dict):
            return vm.get(key, default)
        return getattr(vm, key, default)

    def update(self, vm):
        if self._ax is None:
            return

        pos = self._vm_get(vm, "pos", {})
        active_edges = self._vm_get(vm, self.key, []) or []

        # Remove last frame's arrows
        self._clear()

        if not active_edges:
            return

        # For bidirectional detection this tick
        active_set = set(active_edges)

        for (u, v) in active_edges:
            if u not in pos or v not in pos:
                continue

            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # Only flip curvature if BOTH directions are present
            rad = self.rad
            if self.curve_bidirectional and (v, u) in active_set and u != v:
                rad = self.rad if u < v else -self.rad

            patch = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>",
                mutation_scale=self.arrowsize,
                linewidth=self.lw,
                alpha=self.alpha,
                color=self.color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=self.z,
            )
            self._ax.add_patch(patch)
            self._patches.append(patch)