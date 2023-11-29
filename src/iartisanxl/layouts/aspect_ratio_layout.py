from PyQt6.QtWidgets import QWidgetItem, QLayout
from PyQt6.QtCore import QRect, QSize


class AspectRatioLayout(QLayout):
    def __init__(self, aspect_ratio, widget):
        super().__init__()
        self.aspect_ratio = aspect_ratio
        self.widget = widget
        self.item = QWidgetItem(widget)

    def addItem(self, item):
        pass

    def count(self):
        return 1

    def itemAt(self, index):
        if index == 0:
            return self.item
        return None

    def takeAt(self, _index):
        return None

    def setGeometry(self, rect):
        w = rect.width()
        h = int(w / self.aspect_ratio)
        if h > rect.height():
            h = rect.height()
            w = int(h * self.aspect_ratio)
        x = (rect.width() - w) // 2
        self.widget.setGeometry(QRect(rect.left() + x, rect.top(), w, h))

    def sizeHint(self):
        return QSize(1, 1)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return int(width / self.aspect_ratio)
