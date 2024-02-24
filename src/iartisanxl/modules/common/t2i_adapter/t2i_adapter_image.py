import attr

from iartisanxl.modules.common.image.image_data_object import ImageDataObject


@attr.s
class T2IAdapterImage:
    t2i_adapter_id = attr.ib(default=None)
    node_id = attr.ib(default=None)
    images: list[ImageDataObject] = attr.ib(factory=list)

    def add_image(
        self,
        image_original: str,
        image_filename: str,
        image_scale: float,
        image_x_pos: int,
        image_y_pos: int,
        image_rotation: float,
        layer_name: str,
        order: int,
    ):
        new_image = ImageDataObject(
            image_id=self._generate_unique_id(),
            image_original=image_original,
            image_filename=image_filename,
            image_scale=image_scale,
            image_x_pos=image_x_pos,
            image_y_pos=image_y_pos,
            image_rotation=image_rotation,
            layer_name=layer_name,
            order=order,
        )
        self.images.append(new_image)

        return new_image

    def _generate_unique_id(self):
        self._image_id_counter = getattr(self, "_image_id_counter", 0) + 1
        return self._image_id_counter
