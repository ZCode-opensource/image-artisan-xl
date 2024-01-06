from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtCore import pyqtSignal

from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.modules.common.image.image_adder_preview import ImageAdderPreview
from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout


class ImageCropperWidget(QWidget):
    image_loaded = pyqtSignal()

    def __init__(self):
        super(ImageCropperWidget, self).__init__()

        self.aspect_index = 0
        self.cropper_width = 1024
        self.cropper_height = 1024
        self.aspect_ratio = float(self.cropper_width) / float(self.cropper_height)

        self.setAcceptDrops(True)

        self.image_path = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        cropper_widget = QWidget()
        self.image_cropper = ImageAdderPreview(self.cropper_width, self.cropper_height, self.aspect_ratio)
        self.image_cropper.image_moved.connect(self.update_image_position)
        self.image_cropper.image_scaled.connect(self.update_image_scale)
        self.cropper_layout = AspectRatioLayout(cropper_widget, self.aspect_ratio)
        self.cropper_layout.addWidget(self.image_cropper)
        cropper_widget.setLayout(self.cropper_layout)

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

    def set_aspect(self, aspect_index):
        self.aspect_index = aspect_index

        if self.aspect_index == 0:
            self.cropper_width = 1024
            self.cropper_height = 1024
        elif self.aspect_index == 1:
            self.cropper_width = 896
            self.cropper_height = 1152
        elif self.aspect_index == 2:
            self.cropper_width = 1152
            self.cropper_height = 896
        elif self.aspect_index == 3:
            self.cropper_width = 1344
            self.cropper_height = 768
        elif self.aspect_index == 4:
            self.cropper_width = 1536
            self.cropper_height = 704

        self.aspect_ratio = float(self.cropper_width) / float(self.cropper_height)
        self.image_cropper.set_aspect(self.cropper_width, self.cropper_height, self.aspect_ratio)
        self.cropper_layout.aspect_ratio = self.aspect_ratio
        self.image_cropper.update()
        self.cropper_layout.update()

    def set_image(self, image_path):
        self.image_cropper.set_image(image_path)

    def set_pixmap(self, pixmap):
        self.image_cropper.set_pixmap(pixmap)

    def get_image(self):
        pixmap = self.image_cropper.get_current_view_as_pixmap()
        return pixmap

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.image_cropper.drop_lightbox.show()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.image_cropper.drop_lightbox.hide()
        event.accept()

    def dropEvent(self, event):
        self.image_cropper.drop_lightbox.hide()

        for url in event.mimeData().urls():
            path = url.toLocalFile()

            reader = QImageReader(path)

            if reader.canRead():
                self.clear_image()
                self.image_path = path
                pixmap = QPixmap(self.image_path)
                self.image_cropper.set_pixmap(pixmap)
                self.image_loaded.emit()

    def reset_values(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()

    def update_image_position(self, x, y):
        self.image_x_pos_control.set_value(x)
        self.image_y_pos_control.set_value(y)

    def update_image_scale(self, scale):
        self.image_scale_control.set_value(scale)

    def update_image_rotation(self, angle):
        self.image_rotation_control.set_value(angle)

    def update_image_params(self, pos_x, pos_y, scale, angle):
        self.image_x_pos_control.set_value(pos_x)
        self.image_y_pos_control.set_value(pos_y)
        self.image_scale_control.set_value(scale)
        self.image_rotation_control.set_value(angle)

        self.image_cropper.set_image_x(pos_x)
        self.image_cropper.set_image_y(pos_y)
        self.image_cropper.set_image_scale(scale)
        self.image_cropper.rotate_image(angle)

    def clear_image(self):
        self.reset_values()
        self.image_cropper.clear()
