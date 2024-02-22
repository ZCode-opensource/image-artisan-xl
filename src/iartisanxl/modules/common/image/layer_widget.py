from importlib.resources import files

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from iartisanxl.buttons.transparent_button import TransparentButton


class LayerWidget(QWidget):
    LINK_IMG = files("iartisanxl.theme.icons").joinpath("link.png")
    UNLINK_IMG = files("iartisanxl.theme.icons").joinpath("unlink.png")

    lock_changed = pyqtSignal(int, bool)

    def __init__(self, layer_id: int, name: str):
        super().__init__()

        self.layer_id = layer_id
        self.name = name
        self.lock = True
        self.lock_parent = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(3, 0, 0, 0)
        main_layout.setSpacing(0)

        self.layer_name_label = QLabel(self.name)
        main_layout.addWidget(self.layer_name_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.lock_button = TransparentButton(self.LINK_IMG, 25, 25)
        self.lock_button.clicked.connect(self.on_lock_clicked)
        main_layout.addWidget(self.lock_button, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        self.setLayout(main_layout)

    def on_lock_clicked(self):
        self.lock = not self.lock
        self.lock_button.icon = self.LINK_IMG if self.lock else self.UNLINK_IMG
        self.lock_changed.emit(self.layer_id, self.lock)
