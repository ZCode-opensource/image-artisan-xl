import attr


@attr.s
class ImageDataObject:
    id = attr.ib(default=None)
    node_id = attr.ib(default=None)
    weight = attr.ib(type=float, default=1.0)
    image_thumb = attr.ib(type=str, default=None)
    image_filename = attr.ib(type=str, default=None)
    image_original = attr.ib(type=str, default=None)
    image_scale = attr.ib(type=float, default=1.0)
    image_x_pos = attr.ib(type=int, default=0)
    image_y_pos = attr.ib(type=int, default=0)
    image_rotation = attr.ib(type=int, default=0)
