from importlib.resources import files


from PyQt6.QtWidgets import QFrame, QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, pyqtSignal

from iartisanxl.buttons.menu_button import MenuButton
from iartisanxl.buttons.expand_button import ExpandButton


class LeftMenu(QFrame):
    open_preferences = pyqtSignal()
    open_downloader = pyqtSignal()

    EXPANDED_WIDTH = 150
    NORMAL_WIDTH = 43
    CONFIG_ICON = files("iartisanxl.theme.icons").joinpath("config.png")
    DOWNLOADER_ICON = files("iartisanxl.theme.icons").joinpath("downloader.png")

    def __init__(self, gui_options: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gui_options = gui_options

        self.expanded = self.gui_options.get("left_menu_expanded")
        self.animating = False
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.finished.connect(self.animation_finished)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.setDuration(300)

        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(self.EXPANDED_WIDTH, 50)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        container = QWidget()
        container.setObjectName("container")
        self.container_layout = QVBoxLayout(container)
        self.container_layout.setContentsMargins(1, 0, 0, 0)
        self.container_layout.setSpacing(0)

        hamburguer_btn = ExpandButton()
        hamburguer_btn.clicked.connect(self.on_expand_clicked)
        self.container_layout.addWidget(hamburguer_btn)

        self.container_layout.addStretch()

        main_layout.addWidget(container)

        downloader_button = MenuButton(icon=self.DOWNLOADER_ICON, label="Downloader")
        downloader_button.clicked.connect(self.open_downloader.emit)
        main_layout.addWidget(downloader_button)

        configuration_button = MenuButton(icon=self.CONFIG_ICON, label="Preferences")
        configuration_button.clicked.connect(self.open_preferences.emit)
        main_layout.addWidget(configuration_button)

        self.setLayout(main_layout)

        if not self.expanded:
            self.setFixedWidth(self.NORMAL_WIDTH)
            hamburguer_btn.extended = False
            self.gui_options["left_menu_expanded"] = False

    def add_module_button(self, icon, label, module_class, callback):
        module_button = MenuButton(
            icon=icon,
            label=label,
        )
        module_button.clicked.connect(lambda: callback(module_class, label))
        index = self.container_layout.count() - 1
        self.container_layout.insertWidget(index, module_button)

    def on_expand_clicked(self):
        if self.expanded:
            self.contract()
        else:
            self.expand()

    def expand(self):
        if self.animating:
            return

        self.animation.setStartValue(self.NORMAL_WIDTH)
        self.animation.setEndValue(self.EXPANDED_WIDTH)
        self.animation.start()
        self.animating = True

    def contract(self):
        if self.animating:
            return

        self.animation.setStartValue(self.EXPANDED_WIDTH)
        self.animation.setEndValue(self.NORMAL_WIDTH)
        self.animation.start()
        self.animating = True

    def animation_finished(self):
        if self.expanded:
            self.expanded = False
        else:
            self.expanded = True

        self.gui_options["left_menu_expanded"] = self.expanded
        self.animating = False
