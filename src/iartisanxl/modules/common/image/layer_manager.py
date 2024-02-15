from PyQt6.QtWidgets import QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap

from iartisanxl.modules.common.image.image_editor_layer import ImageEditorLayer


class LayerManager:
    def __init__(self):
        self.layers = []
        self.next_layer_id = 0

    def shift_order(self, order: int):
        for layer in self.layers:
            if layer.order >= order:
                layer.order += 1

    def add_new_layer(
        self,
        pixmap: QPixmap,
        original_pixmap: QGraphicsPixmapItem = None,
        image_path: str = None,
        parent_id: int = None,
        name: str = None,
        order: int = None,
    ) -> ImageEditorLayer:
        if original_pixmap is None:
            original_pixmap = pixmap

        if order is not None:
            self.shift_order(order)
        else:
            order = max(layer.order for layer in self.layers) + 1 if self.layers else 0

        pixmap_item = QGraphicsPixmapItem(pixmap)
        layer = ImageEditorLayer(
            pixmap_item=pixmap_item, original_pixmap=original_pixmap, image_path=image_path, parent_id=parent_id, name=name, order=order
        )

        layer.layer_id = self.next_layer_id
        self.layers.append(layer)
        self.next_layer_id += 1

        return layer

    def edit_layer(
        self,
        layer_id: int,
        pixmap: QPixmap = None,
        original_pixmap: QPixmap = None,
        image_path: str = None,
        parent_id: int = None,
        name: str = None,
        order: int = None,
    ) -> ImageEditorLayer:
        layer = self.get_layer_by_id(layer_id)
        if layer is not None:
            if pixmap is not None:
                layer.pixmap_item.setPixmap(pixmap)

            if original_pixmap is not None:
                layer.original_pixmap = original_pixmap

            layer.image_path = image_path if image_path is not None else layer.image_path
            layer.parent_id = parent_id if parent_id is not None else layer.parent_id
            layer.name = name if name is not None else layer.name

            if order is not None and order != layer.order:
                self.move_layer(layer_id, order)

            return layer
        return None

    def delete_layer(self, layer_id: int):
        deleted_order = None
        for i, layer in enumerate(self.layers):
            if layer.layer_id == layer_id:
                deleted_order = layer.order
                del self.layers[i]
                break

        if deleted_order is not None:
            for layer in self.layers:
                if layer.order > deleted_order:
                    layer.order -= 1

    def move_layer(self, layer_id: int, new_order: int):
        layer = self.get_layer_by_id(layer_id)
        if layer is not None:
            self.layers.remove(layer)

            for l in self.layers:
                if l.order >= new_order:
                    l.order += 1

            layer.order = new_order
            self.layers.append(layer)

            self.reorder_layers()

    def reorder_layers(self):
        self.layers.sort(key=lambda layer: layer.order)

    def get_layers(self):
        return self.layers

    def get_layer_by_id(self, layer_id: int) -> ImageEditorLayer:
        if len(self.layers) > 0:
            for layer in self.layers:
                if layer.layer_id == layer_id:
                    return layer
        return None

    def get_layer_at_order(self, order: int) -> ImageEditorLayer:
        if len(self.layers) > 0:
            for layer in self.layers:
                if layer.order == order:
                    return layer
        return None

    def delete_all(self):
        self.layers = []
        self.next_layer_id = 0
