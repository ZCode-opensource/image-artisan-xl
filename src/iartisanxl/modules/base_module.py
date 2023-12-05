from abc import abstractmethod, ABCMeta
from PyQt6.QtWidgets import QWidget, QStatusBar

from iartisanxl.app.event_bus import EventBus
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject


class ABCQWidgetMeta(ABCMeta, type(QWidget)):
    pass


class BaseModule(QWidget, metaclass=ABCQWidgetMeta):
    def __init__(
        self,
        status_bar: QStatusBar,
        show_snackbar,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
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

        self.event_bus = EventBus()

    @abstractmethod
    def init_ui(self):
        pass
