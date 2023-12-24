import attr
from PIL import Image


@attr.s
class IPAdapterDataObject:
    enabled = attr.ib(type=bool)
    adapter_type = attr.ib(type=str)
    type_index = attr.ib(type=int, default=0)
    adapter_id = attr.ib(type=int, default=None)
    id = attr.ib(default=None)
    image = attr.ib(type=Image, default=None)
    image_thumb = attr.ib(type=Image, default=None)
    image_filename = attr.ib(type=Image, default=None)
    ip_adapter_scale = attr.ib(type=float, default=1.0)
