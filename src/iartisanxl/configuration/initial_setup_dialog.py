from PyQt6.QtWidgets import QDialog, QWidget
from PyQt6.QtCore import Qt, QPropertyAnimation, QRect
from PyQt6.QtGui import QPainter, QPen, QColor

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.configuration.welcome_panel import WelcomePanel
from iartisanxl.configuration.directories_panel import DirectoriesPanel
from iartisanxl.configuration.control_adapters_panel import ControlAdaptersPanel
from iartisanxl.configuration.optimizations_panel import OptimizationsPanel


class InitialSetupDialog(QDialog):
    border_color = QColor("#ff6b6b6b")

    steps_panels = [
        WelcomePanel,
        DirectoriesPanel,
        ControlAdaptersPanel,
        OptimizationsPanel,
    ]

    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dialog_width = 600
        self.dialog_height = 600

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setFixedSize(self.dialog_width, self.dialog_height)

        self.directories = directories
        self.preferences = preferences
        self.current_panel_index = 0
        self.enter_animation = None
        self.leave_animation = None

        self.init_ui()

    def init_ui(self):
        self.load_panel(self.steps_panels[0])

    def load_panel(self, Panel):
        panel = Panel(self.directories, self.preferences)
        panel.finish_setup.connect(self.finish_setup)
        panel.next_step.connect(self.next_step)
        panel.back_step.connect(self.step_back)
        panel.setFixedSize(self.dialog_width, self.dialog_height)
        panel.setParent(self)

        if self.current_panel_index != 0:
            panel.setGeometry(self.dialog_width, 0, self.dialog_width, self.dialog_height)
            panel.show()
            self.enter_animation = QPropertyAnimation(panel, b"geometry")
            self.enter_animation.setDuration(350)
            self.enter_animation.setStartValue(QRect(self.dialog_width, 0, self.dialog_width, self.dialog_height))
            self.enter_animation.setEndValue(QRect(0, 0, self.dialog_width, self.dialog_height))
            self.enter_animation.start()

    def next_step(self):
        if self.current_panel_index >= len(self.steps_panels) - 1:
            return

        current_panel = self.findChild(QWidget)

        self.leave_animation = QPropertyAnimation(current_panel, b"geometry")
        self.leave_animation.setDuration(350)
        self.leave_animation.setStartValue(QRect(0, 0, self.dialog_width, self.dialog_height))
        self.leave_animation.setEndValue(QRect(-self.dialog_width, 0, self.dialog_width, self.dialog_height))
        self.leave_animation.finished.connect(current_panel.deleteLater)
        self.leave_animation.start()

        self.current_panel_index += 1
        self.load_panel(self.steps_panels[self.current_panel_index])

    def step_back(self):
        if self.current_panel_index <= 0:
            return

        current_panel = self.findChild(QWidget)

        self.leave_animation = QPropertyAnimation(current_panel, b"geometry")
        self.leave_animation.setDuration(350)
        self.leave_animation.setStartValue(QRect(0, 0, self.dialog_width, self.dialog_height))
        self.leave_animation.setEndValue(QRect(self.dialog_width, 0, self.dialog_width, self.dialog_height))
        self.leave_animation.finished.connect(current_panel.deleteLater)
        self.leave_animation.start()

        self.current_panel_index -= 1
        self.load_previous_panel(self.steps_panels[self.current_panel_index])

    def load_previous_panel(self, Panel):
        panel = Panel(self.directories, self.preferences)
        panel.finish_setup.connect(self.finish_setup)
        panel.next_step.connect(self.next_step)
        panel.back_step.connect(self.step_back)
        panel.setFixedSize(self.dialog_width, self.dialog_height)
        panel.setParent(self)

        panel.setGeometry(-self.dialog_width, 0, self.dialog_width, self.dialog_height)
        panel.show()
        self.enter_animation = QPropertyAnimation(panel, b"geometry")
        self.enter_animation.setDuration(350)
        self.enter_animation.setStartValue(QRect(-self.dialog_width, 0, self.dialog_width, self.dialog_height))
        self.enter_animation.setEndValue(QRect(0, 0, self.dialog_width, self.dialog_height))
        self.enter_animation.start()

    def finish_setup(self):
        self.close()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(0, 0, self.width(), 0)
        painter.drawLine(self.width(), 0, self.width(), self.height())
        painter.drawLine(0, self.height(), self.width(), self.height())
