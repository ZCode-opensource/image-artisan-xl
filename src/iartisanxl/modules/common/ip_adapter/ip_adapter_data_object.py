import copy

import attr

from iartisanxl.modules.common.ip_adapter.ip_adapter_image import IPAdapterImage
from iartisanxl.modules.common.mask.mask_image import MaskImage


@attr.s(auto_attribs=True, slots=True)
class IPAdapterDataObject:
    _image_id_counter: int = attr.ib(default=0)
    adapter_name: str = attr.ib(default=None)
    adapter_type: str = attr.ib(default=None)
    type_index: int = attr.ib(default=0)
    ip_adapter_scale: float = attr.ib(default=1.0)
    enabled: bool = attr.ib(default=True)
    node_id: int = attr.ib(default=None)
    adapter_id: int = attr.ib(default=None)
    images: list[IPAdapterImage] = attr.ib(factory=list)
    mask_image: MaskImage = attr.ib(default=None)
    _original_images: list[IPAdapterImage] = attr.Factory(list)

    def add_image(self, images, weight, noise_type, noise_type_index, noise):
        new_image = IPAdapterImage(
            ip_adapter_id=self._generate_unique_id(),
            images=images,
            weight=weight,
            noise_type=noise_type,
            noise_type_index=noise_type_index,
            noise=noise,
        )
        self.images.append(new_image)

        return new_image

    def add_ip_adapter_image(self, ip_adapter_image: IPAdapterImage):
        ip_adapter_image.ip_adapter_id = self._generate_unique_id()
        self.images.append(ip_adapter_image)

    def update_ip_adapter_image(self, ip_adapter_image: IPAdapterImage):
        for i, image in enumerate(self.images):
            if image.ip_adapter_id == ip_adapter_image.ip_adapter_id:
                node_id = self.images[i].node_id
                ip_adapter_image.node_id = node_id
                self.images[i] = ip_adapter_image
                return
        raise ValueError(f"No image found with id {ip_adapter_image.ip_adapter_id}")

    def delete_ip_adapter_image(self, ip_adapter_id):
        images = self.images
        for image in images:
            if image.ip_adapter_id == ip_adapter_id:
                images.remove(image)
                return

    def _generate_unique_id(self):
        self._image_id_counter = getattr(self, "_image_id_counter", 0) + 1
        return self._image_id_counter

    def get_image_data_object(self, ip_adapter_id):
        for image in self.images:
            if image.ip_adapter_id == ip_adapter_id:
                return image

        return None

    def save_image_state(self):
        self._original_images = copy.deepcopy(self.images)

    def get_added_images(self):
        original_ids = [image.ip_adapter_id for image in self._original_images]
        return [image for image in self.images if image.ip_adapter_id not in original_ids]

    def get_removed_images(self):
        current_ids = [image.ip_adapter_id for image in self.images]
        return [image for image in self._original_images if image.ip_adapter_id not in current_ids]

    def get_modified_images(self):
        original_images_dict = {image.ip_adapter_id: image for image in self._original_images}
        return [
            image
            for image in self.images
            if image.ip_adapter_id in original_images_dict and image != original_images_dict[image.ip_adapter_id]
        ]
