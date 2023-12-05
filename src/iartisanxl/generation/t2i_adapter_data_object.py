import attr
from PIL import Image


@attr.s
class T2IAdapterDataObject:
    enabled = attr.ib(type=bool)
    adapter_type = attr.ib(type=str)
    adapter_id = attr.ib(type=int, default=None)
    source_image = attr.ib(type=Image, default=None)
    source_image_thumb = attr.ib(type=Image, default=None)
    source_image_filename = attr.ib(type=Image, default=None)
    annotator_image = attr.ib(type=Image, default=None)
    annotator_image_thumb = attr.ib(type=Image, default=None)
    annotator_image_filename = attr.ib(type=Image, default=None)
    conditioning_scale = attr.ib(type=float, default=1.0)
    conditioning_factor = attr.ib(type=float, default=1.0)
    type_index = attr.ib(type=int, default=0)
    canny_low = attr.ib(type=int, default=100)
    canny_high = attr.ib(type=int, default=300)
    id = attr.ib(default=None)
