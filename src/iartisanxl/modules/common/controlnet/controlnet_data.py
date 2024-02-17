import attr

from iartisanxl.modules.common.controlnet.controlnet_image import ControlNetImage


@attr.s(auto_attribs=True, slots=True)
class ControlNetData:
    enabled: bool = attr.ib(default=True)
    adapter_name: str = attr.ib(default=None)
    adapter_type: str = attr.ib(default=None)
    type_index: int = attr.ib(default=0)
    guess_mode: bool = attr.ib(default=False)
    adapter_id: int = attr.ib(default=None)
    source_images: ControlNetImage = attr.ib(default=attr.Factory(ControlNetImage))
    source_image: str = attr.ib(default=None)
    source_thumb: str = attr.ib(default=None)
    preprocessor_images: ControlNetImage = attr.ib(default=attr.Factory(ControlNetImage))
    preprocessor_image: str = attr.ib(default=None)
    preprocessor_thumb: str = attr.ib(default=None)
    preprocessor_resolution: float = attr.ib(default=0.5)
    conditioning_scale: float = attr.ib(default=0.5)
    guidance_start: float = attr.ib(default=0.0)
    guidance_end: float = attr.ib(default=1.0)
    canny_low: int = attr.ib(default=100)
    canny_high: int = attr.ib(default=300)
    depth_type: str = attr.ib(default=None)
    depth_type_index: int = attr.ib(default=0)
    node_id: int = attr.ib(default=None)
    generation_width: int = attr.ib(default=None)
    generation_height: int = attr.ib(default=None)
