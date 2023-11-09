import re

from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QTextCursor


class WeightedTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.installEventFilter(self)

    def insertFromMimeData(self, source):
        self.insertPlainText(source.text())

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if (
                event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down)
                and event.modifiers() == Qt.KeyboardModifier.ControlModifier
            ):
                cursor = self.textCursor()
                if cursor.hasSelection():
                    selected_text = cursor.selectedText()
                    start = cursor.selectionStart()
                    end = cursor.selectionEnd()
                    pattern = r"\(([^:]+):(-?\d+\.\d+)\)"
                    matches = list(re.finditer(pattern, self.toPlainText()))
                    match = None
                    for m in matches:
                        if start <= m.start() and end >= m.end():
                            match = m
                            break
                        elif start >= m.start() and end <= m.end():
                            match = m
                            break
                    if match:
                        value = float(match.group(2))
                        if event.key() == Qt.Key.Key_Up:
                            new_value = min(2.0, value + 0.1)
                        else:
                            new_value = max(0, value - 0.1)
                        new_text = f"({match.group(1)}:{new_value:.1f})"
                        cursor.setPosition(match.start())
                        cursor.setPosition(match.end(), QTextCursor.MoveMode.KeepAnchor)
                        cursor.insertText(new_text)
                        cursor.setPosition(match.start())
                        cursor.setPosition(
                            match.start() + len(new_text),
                            QTextCursor.MoveMode.KeepAnchor,
                        )
                        self.setTextCursor(cursor)
                    else:
                        new_text = f"({selected_text}:1.0)"
                        cursor.removeSelectedText()
                        cursor.insertText(new_text)
                        cursor.setPosition(start)
                        cursor.setPosition(
                            start + len(new_text),
                            QTextCursor.MoveMode.KeepAnchor,
                        )
                        self.setTextCursor(cursor)
                    return True
        return super().eventFilter(obj, event)

    def insertTextAtCursor(self, text):
        cursor = self.textCursor()
        cursor.insertText(text)

    def insertTriggerAtCursor(self, text):
        cursor = self.textCursor()
        cursor_position = cursor.position()

        if cursor_position > 0:
            cursor.setPosition(cursor_position - 1, QTextCursor.MoveMode.KeepAnchor)
            if cursor.selectedText() == " ":
                if cursor_position > 1:
                    cursor.setPosition(
                        cursor_position - 2, QTextCursor.MoveMode.KeepAnchor
                    )
                    if cursor.selectedText()[0] != ",":
                        text = ", " + text
            elif cursor.selectedText() != ",":
                text = ", " + text
            else:
                text = " " + text

        cursor.setPosition(cursor_position)

        if cursor_position < len(self.toPlainText()):
            cursor.setPosition(cursor_position + 1, QTextCursor.MoveMode.KeepAnchor)
            if cursor.selectedText() != "," and cursor.selectedText() != "":
                text = text + ", "

        cursor.setPosition(cursor_position)
        cursor.insertText(text)
        cursor.setPosition(cursor_position + len(text))
