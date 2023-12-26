from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

from iartisanxl.modules.common.image_cropper import ImageCropper
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout


class ImageCropperWidget(QWidget):
    def __init__(self):
        super(ImageCropperWidget, self).__init__()

        self.original_width = 1024
        self.original_height = 1024
        self.aspect_ratio = 1.0

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        cropper_widget = QWidget()
        self.image_cropper = ImageCropper(1024, 1024, 1.0)
        self.image_cropper.imageMoved.connect(self.update_image_position)
        cropper_layout = AspectRatioLayout(cropper_widget, 1)
        cropper_layout.addWidget(self.image_cropper)
        cropper_widget.setLayout(cropper_layout)

        image_controls_layout = QHBoxLayout()
        self.image_scale_control = ImageControl("Scale: ", 1.000, 3)
        self.image_scale_control.value_changed.connect(self.image_cropper.set_image_scale)
        image_controls_layout.addWidget(self.image_scale_control)
        self.image_x_pos_control = ImageControl("X Pos: ", 0, 0)
        self.image_x_pos_control.value_changed.connect(self.image_cropper.set_image_x)
        image_controls_layout.addWidget(self.image_x_pos_control)
        self.image_y_pos_control = ImageControl("Y Pos: ", 0, 0)
        self.image_y_pos_control.value_changed.connect(self.image_cropper.set_image_y)
        image_controls_layout.addWidget(self.image_y_pos_control)
        self.image_rotation_control = ImageControl("Rotation: ", 0, 0)
        self.image_rotation_control.value_changed.connect(self.image_cropper.rotate_image)
        image_controls_layout.addWidget(self.image_rotation_control)

        main_layout.addWidget(cropper_widget)
        main_layout.addLayout(image_controls_layout)

        self.setLayout(main_layout)

    def set_image(self, image_path):
        self.image_cropper.set_image(image_path)

    def set_pixmap(self, pixmap):
        self.image_cropper.set_pixmap(pixmap)

    def get_image(self):
        pixmap = None

        if self.image_cropper.image_modified:
            pixmap = self.image_cropper.get_current_view_as_pixmap()

        return pixmap

    def update_image_position(self, x, y):
        self.image_x_pos_control.set_value(x)
        self.image_y_pos_control.set_value(y)

    def reset_values(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()

    def clear(self):
        self.reset_values()
        self.image_cropper.clear()
