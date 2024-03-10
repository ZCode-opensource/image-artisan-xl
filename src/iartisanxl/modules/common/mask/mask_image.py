import attr

from iartisanxl.modules.common.image.image_data_object import ImageDataObject


@attr.s
class MaskImage:
    ip_adapter_id: int = attr.ib(default=None)
    node_id: int = attr.ib(default=None)
    background_image: ImageDataObject = attr.ib(default=None)
    mask_image: ImageDataObject = attr.ib(default=None)
    thumb: str = attr.ib(default=None)
