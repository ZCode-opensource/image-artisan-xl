from PyQt6.QtWidgets import QWidget, QSizePolicy, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, QSize, pyqtSignal

from iartisanxl.modules.common.weighted_text_edit import WeightedTextEdit


class PromptInput(QWidget):
    text_changed = pyqtSignal()

    def __init__(
        self,
        positive: bool,
        token_count: int,
        max_tokens: int,
        *args,
        title: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setMinimumSize(QSize(511, 80))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.positive = positive
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.title = title

        if self.title is None:
            if positive:
                self.title = "Positive"
            else:
                self.title = "Negative"

        self.init_ui()

    def init_ui(self):
        self.weighted_text_edit = WeightedTextEdit(self)
        self.weighted_text_edit.textChanged.connect(self.text_changed.emit)

        self.count_widget = QWidget(self)
        top_layout = QHBoxLayout(self.count_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        self.title_label = QLabel(self.title)
        top_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignLeft)

        self.token_count_label = QLabel(f"{self.token_count}/{self.max_tokens}", self)
        top_layout.addWidget(
            self.token_count_label, alignment=Qt.AlignmentFlag.AlignRight
        )

        if self.positive:
            self.weighted_text_edit.setStyleSheet(
                "color: #4da460; border: 1px solid #264624; padding: 2px;"
            )
            self.title_label.setStyleSheet(
                "color: #4da460; background-color: rgba(100, 100, 100, 50); font-weight: bold; border-radius: 5px; padding: 1px;"
            )
            self.token_count_label.setStyleSheet(
                "color: #4da460; background-color: rgba(100, 100, 100, 50); font-weight: bold; border-radius: 5px; padding: 1px;"
            )
        else:
            self.weighted_text_edit.setStyleSheet(
                "color: #c25e5f; border: 1px solid #582d2d;"
            )
            self.title_label.setStyleSheet(
                "color: #c25e5f; background-color: rgba(100, 100, 100, 50); font-weight: bold; border-radius: 5px; padding: 1px;"
            )
            self.token_count_label.setStyleSheet(
                "color: #c25e5f; background-color: rgba(100, 100, 100, 50); font-weight: bold; border-radius: 5px; padding: 1px;"
            )

    def resizeEvent(self, _event):
        self.count_widget.setFixedWidth(self.width())
        self.weighted_text_edit.setFixedSize(QSize(self.width(), self.height() - 12))
        self.weighted_text_edit.move(0, 12)

    def setPlainText(self, text):
        self.weighted_text_edit.setPlainText(text)

    def toPlainText(self):
        return self.weighted_text_edit.toPlainText()

    def update_token_count(self, count: int):
        self.token_count_label.setText(f"{count}/{self.max_tokens}")

    def set_title_text(self, title: str):
        self.title_label.setText(title)

    def insertTextAtCursor(self, text):
        self.weighted_text_edit.insertTextAtCursor(text)

    def insertTriggerAtCursor(self, text):
        self.weighted_text_edit.insertTriggerAtCursor(text)
