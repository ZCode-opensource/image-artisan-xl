import attr


@attr.s
class ImageDataObject:
    image_id = attr.ib(default=None)
    layer_id = attr.ib(default=None)
    layer_name = attr.ib(default=None)
    node_id = attr.ib(default=None)
    image_thumb = attr.ib(type=str, default=None)
    image_filename = attr.ib(type=str, default=None)
    image_original = attr.ib(type=str, default=None)
    image_drawings = attr.ib(type=str, default=None)
    image_scale = attr.ib(type=float, default=1.0)
    image_x_pos = attr.ib(type=int, default=0)
    image_y_pos = attr.ib(type=int, default=0)
    image_rotation = attr.ib(type=int, default=0)
    replace_original = attr.ib(type=bool, default=False)
    order = attr.ib(type=int, default=None)
