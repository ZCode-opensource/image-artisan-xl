from PyQt6.QtWidgets import QTextEdit, QSizePolicy
from PyQt6.QtGui import QKeyEvent, QIntValidator, QTextCursor
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData


class CustomTextEdit(QTextEdit):
    char_changed = pyqtSignal(int)

    def __init__(self, *args, char_limit: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumHeight(50)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.validator = QIntValidator(0, char_limit)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            return

        super().keyPressEvent(event)
        current_text = self.toPlainText()

        if len(current_text) > self.validator.top():
            self.setPlainText(current_text[: self.validator.top()])
            self.moveCursor(QTextCursor.MoveOperation.End)

        self.char_changed.emit(len(self.toPlainText()))

    def insertFromMimeData(self, source: QMimeData) -> None:
        if source.hasText():
            text = source.text()
            current_text = self.toPlainText()
            if len(current_text + text) > self.validator.top():
                text = text[: self.validator.top() - len(current_text)]
            self.insertPlainText(text)
            self.char_changed.emit(len(self.toPlainText()))

    def setPlainText(self, text: str) -> None:
        super().setPlainText(text)
        if text is not None:
            self.char_changed.emit(len(text))
