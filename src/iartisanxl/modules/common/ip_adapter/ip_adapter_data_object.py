import copy
import attr

from iartisanxl.modules.common.image.image_data_object import ImageDataObject


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
    images: list[ImageDataObject] = attr.ib(factory=list)
    _original_images: list[ImageDataObject] = attr.Factory(list)

    def add_image(self, image_filename, image_thumb, weight=1.0, image_scale=1.0, image_x_pos=0, image_y_pos=0, image_rotation=0):
        """Adds an image to the images list and generates a unique ID."""

        new_image = ImageDataObject(
            id=self._generate_unique_id(),
            weight=weight,
            image_thumb=image_thumb,
            image_filename=image_filename,
            image_scale=image_scale,
            image_x_pos=image_x_pos,
            image_y_pos=image_y_pos,
            image_rotation=image_rotation,
        )
        self.images.append(new_image)

        return new_image

    def add_image_data_object(self, image_data_object: ImageDataObject):
        """Adds an image data object to the images list and generates a unique ID."""

        image_data_object.id = self._generate_unique_id()
        self.images.append(image_data_object)

    def delete_image(self, image_id):
        """Deletes an image from the images list by its ID."""

        images = self.images
        for image in images:
            if image.id == image_id:
                images.remove(image)
                return  # Image found and removed

    def _generate_unique_id(self):
        """Generates a unique ID using a simple counter."""

        self._image_id_counter = getattr(self, "_image_id_counter", 0) + 1
        return self._image_id_counter

    def get_image_data_object(self, image_id):
        """Retrieves the image data object with the given ID from the list of images.

        Args:
            image_id (int): The ID of the image data object to retrieve.

        Returns:
            ImageDataObject: The image data object with the matching ID, or None if not found.
        """

        for image in self.images:
            if image.id == image_id:
                return image

        return None

    def save_image_state(self):
        self._original_images = copy.deepcopy(self.images)

    def get_added_images(self):
        original_ids = [image.id for image in self._original_images]
        return [image for image in self.images if image.id not in original_ids]

    def get_removed_images(self):
        current_ids = [image.id for image in self.images]
        return [image for image in self._original_images if image.id not in current_ids]

    def get_modified_images(self):
        original_images_dict = {image.id: image for image in self._original_images}
        return [image for image in self.images if image.id in original_images_dict and image != original_images_dict[image.id]]
