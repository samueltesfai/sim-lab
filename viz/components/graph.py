from viz.components.base import _VizComponent
from viz.scene import Scene
from viz.view_model import ViewModel
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
import numpy as np
from typing import Literal


class Nodes(_VizComponent):
    def __init__(self, scene: Scene, color_mode: Literal["gray", "invert_gray", "truth_rg"] = "gray", z: int = 2):
        super().__init__(scene, z=z)
        self.color_mode = color_mode
        self.sc = None  # PathCollection

    def add_to_canvas(self, ax: Axes, fig: plt.Figure) -> None:
        # Use initial positions (will be updated every tick anyway)
        xy = np.zeros((len(self.scene.nodes), 2), dtype=float)
        self.sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            s=self.scene.sizes_base,
            c=[(0.5, 0.5, 0.5)] * len(self.scene.nodes),
            edgecolors="black",
            linewidths=0.8,
            zorder=self.z,
        )

    def update(self, vm: ViewModel) -> None:
        if self.sc is None:
            return

        # update positions
        pos = vm.pos
        pts = np.array([pos[n] for n in self.scene.nodes], dtype=float)
        self.sc.set_offsets(pts)

        # update colors
        truth = vm.truth_bool
        beliefs = vm.beliefs
        colors = np.array(
            [self.node_color(beliefs.get(n, 0.5), mode=self.color_mode, truth=truth) for n in self.scene.nodes],
            dtype=float
        )
        self.sc.set_facecolors(colors)

    @staticmethod
    def _belief_to_gray(b: float, invert: bool = False) -> tuple[float, float, float]:
        b = max(0.0, min(1.0, b))
        if invert:
            b = 1.0 - b
        return (b, b, b)

    @staticmethod
    def _color_truth_gradient(b: float, truth: bool) -> tuple[float, float, float]:
        b = max(0.0, min(1.0, b))
        t = 1.0 if truth else 0.0
        alignment = (2.0 * b - 1.0) * (2.0 * t - 1.0)

        if alignment <= 0:
            intensity = abs(alignment)
            return (1.0, intensity, 0.0)   # red -> yellow
        else:
            intensity = alignment
            return (1.0 - intensity, 1.0, 0.0)  # yellow -> green

    @staticmethod
    def node_color(b: float, mode: Literal["gray", "invert_gray", "truth_rg"], truth: bool | None = None) -> tuple[float, float, float]:
        if mode == "gray":
            return Nodes._belief_to_gray(b, invert=False)
        if mode == "invert_gray":
            return Nodes._belief_to_gray(b, invert=True)
        if mode == "truth_rg":
            if truth is None:
                raise ValueError("truth must be provided for truth_rg mode")
            return Nodes._color_truth_gradient(b, truth)
        raise ValueError(f"Unknown color mode: {mode}")
    

class BaseEdges(_VizComponent):
    """
    Fast base edges using a LineCollection.
    Assumes topology is static; if pos changes, we update the segments in update().
    """
    def __init__(self, scene: Scene, alpha: float = 0.15, lw: float = 1.0, z: int = 1):
        super().__init__(scene, z=z)
        self.alpha, self.lw = alpha, lw
        self.lc: LineCollection | None = None
        self._edges = list(scene.G.edges())  # stable order

    def add_to_canvas(self, ax: Axes, fig: plt.Figure) -> None:
        # initial segments (pos will come from first update)
        self.lc = LineCollection([], linewidths=self.lw, alpha=self.alpha, zorder=self.z)
        ax.add_collection(self.lc)

    def update(self, vm: ViewModel) -> None:
        if self.lc is None:
            return
        pos = vm.pos
        segs = [[pos[u], pos[v]] for (u, v) in self._edges]
        self.lc.set_segments(segs)



    


