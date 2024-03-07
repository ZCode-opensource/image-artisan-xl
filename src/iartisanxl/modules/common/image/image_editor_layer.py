import attr
from PyQt6.QtWidgets import QGraphicsPixmapItem


@attr.s(auto_attribs=True, slots=True)
class ImageEditorLayer:
    layer_id: int = attr.ib(default=None)
    pixmap_item: QGraphicsPixmapItem = attr.ib(default=None)
    original_path: str = attr.ib(default=None)
    image_path: str = attr.ib(default=None)
    locked: bool = attr.ib(default=True)
    order: int = attr.ib(default=0)
