from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QListWidget


class LayerListWidget(QListWidget):
    layers_reordered = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setDragDropMode(self.DragDropMode.InternalMove)

    def dropEvent(self, event):
        pre_move_index = self.currentRow()
        super().dropEvent(event)
        post_move_index = self.currentRow()

        if pre_move_index != post_move_index:
            layers_list = []

            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                inverted_index = self.count() - 1 - i
                layers_list.append((widget.layer_id, inverted_index))

            self.layers_reordered.emit(layers_list)
