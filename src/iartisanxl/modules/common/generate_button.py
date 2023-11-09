import logging

from PyQt6.QtWidgets import QPushButton, QSizePolicy, QMenu
from PyQt6.QtGui import QContextMenuEvent, QAction


class GenerateButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.logger = logging.getLogger()

        self.auto_save = False
        self.continuous_generation = False

    def contextMenuEvent(self, event: QContextMenuEvent):
        context_menu = QMenu(self)

        auto_save_action: QAction | None = context_menu.addAction("Auto save")
        auto_save_action.setCheckable(True)
        auto_save_action.setChecked(self.auto_save)
        auto_save_action.triggered.connect(self.handle_auto_save_triggered)

        continuous_generation_action: QAction | None = context_menu.addAction(
            "Continuous generation"
        )
        continuous_generation_action.setCheckable(True)
        continuous_generation_action.setChecked(self.continuous_generation)
        continuous_generation_action.triggered.connect(
            self.handle_continuous_generation_triggered
        )

        context_menu.exec(event.globalPos())

    def handle_auto_save_triggered(self, checked):
        self.auto_save = checked
        if checked:
            self.logger.debug("Auto save enabled")
        else:
            self.logger.debug("Auto save disabled")

    def handle_continuous_generation_triggered(self, checked):
        self.continuous_generation = checked
        if checked:
            self.logger.debug("Continuous generation enabled")
        else:
            self.logger.debug("Continuous generation disabled")
