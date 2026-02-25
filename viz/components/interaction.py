from viz.components.base import _VizComponent
from viz.scene import Scene
from viz.view_model import ViewModel
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class HoverTooltip(_VizComponent):
    def __init__(self, scene: Scene, radius_px: float = 20.0, z: int = 30):
        super().__init__(scene, z=z)
        self.radius_px = radius_px
        self.ax: Axes | None = None
        self.fig: plt.Figure | None = None
        self.annot = None

        self._hover_last_node: int | None = None
        self._vm: ViewModel | None = None

    def add_to_canvas(self, ax: Axes, fig: plt.Figure) -> None:
        self.ax = ax
        self.fig = fig
        self.annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="0.5"),
            zorder=self.z,
        )
        self.annot.set_visible(False)
        fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def update(self, vm: ViewModel) -> None:
        self._vm = vm
        if self.annot and self.annot.get_visible() and self._hover_last_node is not None:
            self._update_hover_for_node(self._hover_last_node)

    def _nearest_node_px(self, event) -> int | None:
        if self._vm is None or self.ax is None:
            return None

        pos = self._vm.pos
        if event.x is None or event.y is None:
            return None

        best = None
        best_d2 = self.radius_px * self.radius_px
        for n, (x, y) in pos.items():
            px, py = self.ax.transData.transform((x, y))
            dx = px - event.x
            dy = py - event.y
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best = n
        return best

    def _on_hover(self, event) -> None:
        if self.ax is None or self.fig is None or self.annot is None:
            return

        if event.inaxes != self.ax:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self._hover_last_node = None
                self.fig.canvas.draw_idle()
            return

        if self._vm is None:
            return

        node = self._nearest_node_px(event)
        if node is None:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self._hover_last_node = None
                self.fig.canvas.draw_idle()
            return

        if node == self._hover_last_node and self.annot.get_visible():
            return

        self._hover_last_node = node
        self._update_hover_for_node(node)
        self.fig.canvas.draw_idle()

    def _update_hover_for_node(self, node: int) -> None:
        if self._vm is None or self.annot is None:
            return

        vm = self._vm
        pos = vm.pos
        b = vm.beliefs.get(node, 0.5)

        out_deg = self.scene.degrees.get(node, 0)
        in_deg = self.scene.G.in_degree(node)
        mem = vm.mem_counts.get(node, 0)

        tags = []
        if node in vm.observed_ids:
            tags.append("OBS")
        if node in vm.verified_ids:
            tags.append("VER")

        involved = any(s == node or r == node for (s, r) in vm.active_edges)
        if involved:
            tags.append("HEAR")

        sent_to = [r for (s, r) in vm.active_edges if s == node]
        heard_from = [s for (s, r) in vm.active_edges if r == node]

        def fmt(xs, max_show=5):
            if not xs:
                return "-"
            if len(xs) <= max_show:
                return ", ".join(map(str, xs))
            return ", ".join(map(str, xs[:max_show])) + f", …(+{len(xs)-max_show})"

        tag_str = f" [{' '.join(tags)}]" if tags else ""
        text = (
            f"agent {node}{tag_str}\n"
            f"claim={vm.claim_id} belief={b:.3f}\n"
            f"in={in_deg} out={out_deg}\n"
            f"mem={mem}\n"
            f"sent→ {fmt(sent_to)}\n"
            f"heard← {fmt(heard_from)}"
        )

        self.annot.xy = pos[node]
        self.annot.set_text(text)
        self.annot.set_visible(True)
