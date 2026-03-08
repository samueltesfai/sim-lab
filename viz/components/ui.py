from viz.components.base import _VizComponent
from viz.scene import Scene
from viz.view_model import ViewModel
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes


class HUDText(_VizComponent):
    def __init__(self, scene: Scene, z: int = 10):
        super().__init__(scene, z=z)
        self.txt = None

    def add_to_canvas(self, ax: Axes, fig: plt.Figure) -> None:
        self.txt = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=self.z,
        )

    def update(self, vm: ViewModel) -> None:
        if self.txt is None:
            return
        s = vm.stats
        self.txt.set_text(
            f"tick={vm.tick}  claim={vm.claim_id}\n"
            f"mean={s['mean']:.3f} min={s['min']:.3f} max={s['max']:.3f}\n"
            f"events: hear={len(vm.active_edges)} obs={len(vm.observed_ids)} ver={len(vm.verified_ids)}"
        )


class LegendComponent(_VizComponent):
    def __init__(self, scene: Scene, z: int = 20):
        super().__init__(scene, z=z)
        self.legend = None

    def add_to_canvas(self, ax: Axes, fig: plt.Figure) -> None:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="observe",
                markerfacecolor="none",
                markeredgecolor="deepskyblue",
                markersize=10,
                linewidth=0,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="verify",
                markerfacecolor="none",
                markeredgecolor="magenta",
                markersize=10,
                linewidth=0,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="heard (recv)",
                markerfacecolor="none",
                markeredgecolor="orange",
                markersize=10,
                linewidth=0,
            ),
        ]
        self.legend = ax.legend(handles=handles, loc="lower left", framealpha=0.7)
        self.legend.set_zorder(self.z)
