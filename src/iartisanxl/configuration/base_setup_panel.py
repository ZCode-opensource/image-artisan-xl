from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject


class BaseSetupPanel(QWidget):
    finish_setup = pyqtSignal()
    next_step = pyqtSignal()
    back_step = pyqtSignal()

    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.directories = directories
        self.preferences = preferences

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.main_layout)
        self.buttons_widget = QWidget()

    def on_next_step(self):
        self.buttons_widget.setVisible(False)
        self.next_step.emit()

    def on_back_step(self):
        self.buttons_widget.setVisible(False)
        self.back_step.emit()
