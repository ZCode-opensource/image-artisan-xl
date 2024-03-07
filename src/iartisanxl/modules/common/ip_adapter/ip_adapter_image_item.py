from iartisanxl.modules.common.image.image_item import ImageItem
from iartisanxl.modules.common.ip_adapter.ip_adapter_image import IPAdapterImage


class IPAdapterImageItem(ImageItem):
    def __init__(self, *args, ip_adapter_image: IPAdapterImage):
        super().__init__(*args)

        self.ip_adapter_image = ip_adapter_image
