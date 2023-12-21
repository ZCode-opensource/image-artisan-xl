from abc import abstractmethod, ABCMeta
from PyQt6.QtWidgets import QWidget, QStatusBar

from iartisanxl.app.event_bus import EventBus
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.graph.iartisanxl_node_graph import ImageArtisanNodeGraph


class ABCQWidgetMeta(ABCMeta, type(QWidget)):
    pass


class BaseModule(QWidget, metaclass=ABCQWidgetMeta):
    def __init__(
        self,
        status_bar: QStatusBar,
        show_snackbar,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        node_graph: ImageArtisanNodeGraph,
        *args,
        **kwargs,
    ):
        if not isinstance(status_bar, QStatusBar):
            raise TypeError(f"status_bar must be an instance of QStatusBar, not {type(status_bar)}")
        super().__init__(*args, **kwargs)
        self.status_bar = status_bar
        self.show_snackbar = show_snackbar
        self.directories = directories
        self.preferences = preferences
        self.dialogs = {}
        self.node_graph = node_graph

        self.event_bus = EventBus()

    @abstractmethod
    def init_ui(self):
        pass

    def update_status_bar(self, text):
        self.status_bar.showMessage(text)

    def closeEvent(self, event):
        self.event_bus.unsubscribe_all()
        self.event_bus = None
        super().closeEvent(event)
