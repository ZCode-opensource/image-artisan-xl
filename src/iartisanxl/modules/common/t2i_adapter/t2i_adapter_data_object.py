import attr

from iartisanxl.modules.common.t2i_adapter.t2i_adapter_image import T2IAdapterImage


@attr.s(auto_attribs=True, slots=True)
class T2IAdapterDataObject:
    enabled: bool = attr.ib(default=True)
    adapter_name: str = attr.ib(default=None)
    adapter_type: str = attr.ib(default=None)
    type_index: int = attr.ib(default=0)
    adapter_id: int = attr.ib(default=None)
    source_images: T2IAdapterImage = attr.ib(default=attr.Factory(T2IAdapterImage))
    source_image: str = attr.ib(default=None)
    source_thumb: str = attr.ib(default=None)
    preprocessor_images: T2IAdapterImage = attr.ib(default=attr.Factory(T2IAdapterImage))
    preprocessor_image: str = attr.ib(default=None)
    preprocessor_thumb: str = attr.ib(default=None)
    preprocessor_resolution: float = attr.ib(default=0.5)
    conditioning_scale: float = attr.ib(default=0.5)
    conditioning_factor: float = attr.ib(default=1.0)
    canny_low: int = attr.ib(default=100)
    canny_high: int = attr.ib(default=300)
    depth_type: str = attr.ib(default=None)
    depth_type_index: int = attr.ib(default=0)
    lineart_type: str = attr.ib(default=None)
    lineart_type_index: int = attr.ib(default=0)
    sketch_type: str = attr.ib(default=None)
    sketch_type_index: int = attr.ib(default=0)
    node_id: int = attr.ib(default=None)
    generation_width: int = attr.ib(default=None)
    generation_height: int = attr.ib(default=None)
