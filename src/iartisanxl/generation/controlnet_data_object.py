import attr
from PIL import Image


@attr.s
class ControlNetDataObject:
    enabled = attr.ib(type=bool)
    name = attr.ib(type=str)
    model_path = attr.ib(type=str)
    guess_mode = attr.ib(type=bool)
    source_image = attr.ib(type=Image, default=None)
    source_image_filename = attr.ib(type=Image, default=None)
    annotator_image = attr.ib(type=Image, default=None)
    annotator_image_filename = attr.ib(type=Image, default=None)
    conditioning_scale = attr.ib(type=float, default=1.0)
    guidance_start = attr.ib(type=float, default=0.0)
    guidance_end = attr.ib(type=float, default=1.0)

    def copy(self):
        new_obj = ControlNetDataObject(
            enabled=self.enabled,
            name=self.name,
            model_path=self.model_path,
            source_image=self.source_image,
            source_image_filename=self.source_image_filename,
            annotator_image=self.annotator_image,
            annotator_image_filename=self.annotator_image_filename,
            guess_mode=self.guess_mode,
            conditioning_scale=self.conditioning_scale,
            guidance_start=self.guidance_start,
            guidance_end=self.guidance_end,
        )
        return new_obj
