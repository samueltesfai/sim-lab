
from abc import ABC, abstractmethod
from viz.scene import Scene
from viz.viewmodel import ViewModel
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


class _VizComponent(ABC):
    def __init__(self, scene: Scene, z: int = 0):
        self.scene = scene
        self.z = z

    @abstractmethod
    def add_to_canvas(self, ax: Axes, fig: plt.Figure) -> None:
        """Create artists once."""
        raise NotImplementedError

    def update(self, vm: ViewModel) -> None:
        """Update artists based on the viewmodel."""
        return  # default no-op
