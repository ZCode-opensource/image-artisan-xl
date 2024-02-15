import attr

from PyQt6.QtWidgets import QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap


@attr.s(auto_attribs=True, slots=True)
class ImageEditorLayer:
    layer_id: int = attr.ib(default=None)
    pixmap_item: QGraphicsPixmapItem = attr.ib(default=None)
    original_pixmap: QPixmap = attr.ib(default=None)
    image_path: str = attr.ib(default=None)
    parent_id: int = attr.ib(default=None)
    name: str = attr.ib(default=None)
    order: int = attr.ib(default=0)
