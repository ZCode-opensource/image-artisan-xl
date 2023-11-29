import attr
from PIL import Image


@attr.s
class ControlNetDataObject:
    enabled = attr.ib(type=bool)
    controlnet_type = attr.ib(type=str)
    guess_mode = attr.ib(type=bool)
    controlnet_id = attr.ib(type=int, default=None)
    source_image = attr.ib(type=Image, default=None)
    source_image_thumb = attr.ib(type=Image, default=None)
    source_image_filename = attr.ib(type=Image, default=None)
    annotator_image = attr.ib(type=Image, default=None)
    annotator_image_thumb = attr.ib(type=Image, default=None)
    annotator_image_filename = attr.ib(type=Image, default=None)
    conditioning_scale = attr.ib(type=float, default=1.0)
    guidance_start = attr.ib(type=float, default=0.0)
    guidance_end = attr.ib(type=float, default=1.0)
    type_index = attr.ib(type=int, default=0)
    canny_low = attr.ib(type=int, default=100)
    canny_high = attr.ib(type=int, default=300)
    id = attr.ib(default=None)
