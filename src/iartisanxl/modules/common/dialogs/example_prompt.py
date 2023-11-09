from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal


class ExamplePrompt(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setMinimumHeight(10)

        self.setObjectName("example_prompt_text_label")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def mousePressEvent(self, _):
        self.clicked.emit(self.text())
