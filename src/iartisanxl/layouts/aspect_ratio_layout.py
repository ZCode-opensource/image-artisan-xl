from PyQt6.QtWidgets import QLayout
from PyQt6.QtCore import QRect, QSize


class AspectRatioLayout(QLayout):
    def __init__(self, widget, aspect_ratio):
        super().__init__(widget)
        self.aspect_ratio = aspect_ratio
        self.item_list = []

    def addItem(self, item):
        self.item_list.append(item)

    def count(self):
        return len(self.item_list)

    def itemAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list.pop(index)
        return None

    def setGeometry(self, rect):
        super().setGeometry(rect)

        w = rect.width()
        h = rect.height()

        if w > h * self.aspect_ratio:
            w = h * self.aspect_ratio
        else:
            h = w / self.aspect_ratio

        # Convert w and h to integers
        w = int(w)
        h = int(h)

        # Calculate left and top positions to center the item
        left = (rect.width() - w) // 2
        top = (rect.height() - h) // 2

        for i in range(self.count()):
            item = self.itemAt(i)
            if item:
                item.setGeometry(QRect(left, top, w, h))

    def sizeHint(self):
        return QSize(int(self.aspect_ratio), 1)
