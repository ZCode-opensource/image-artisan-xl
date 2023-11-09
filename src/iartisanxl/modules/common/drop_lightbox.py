from PyQt6 import QtWidgets, QtCore


class DropLightBox(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background-color: rgba(0, 0, 0, 150); color: white; font-size: 24px;"
        )
        self.hide()

    def resizeEvent(self, _event):
        self.resize(self.parentWidget().size())  # type: ignore
