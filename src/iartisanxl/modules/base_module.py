from abc import abstractmethod, ABCMeta
from PyQt6.QtWidgets import QWidget, QStatusBar

from iartisanxl.app.event_bus import EventBus
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog


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
        self.event_bus.subscribe("dialogs", self.on_dialogs)

    @abstractmethod
    def init_ui(self):
        pass

    def on_dialogs(self, data):
        pass

    def open_dialog(self, dialog: BaseDialog):
        dialog_class = type(dialog)
        if dialog_class in self.dialogs:
            self.dialogs[dialog_class].raise_()
            self.dialogs[dialog_class].activateWindow()
        else:
            self.dialogs[dialog_class] = dialog
            dialog.closed.connect(lambda: self.on_dialog_closed(dialog_class))
            dialog.show()
            dialog.dialog_raised()

    def on_dialog_closed(self, dialog_class):
        self.dialogs.pop(dialog_class)

    def closeEvent(self, event):
        for dialog in list(self.dialogs.values()):
            dialog.close()
        super().closeEvent(event)
