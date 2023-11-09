from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from iartisanxl.generation.generation_data_object import ImageGenData


class ImageDimensionsWidget(QtWidgets.QWidget):
    ALLOWED_VALUES = [656, 768, 832, 896, 1024, 1152, 1216, 1344, 1536]

    def __init__(self, image_generation_data: ImageGenData, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_generation_data = image_generation_data

        self.init_ui()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout()

        image_sliders_layout = QtWidgets.QGridLayout()

        width_label = QtWidgets.QLabel("Width")
        image_sliders_layout.addWidget(width_label, 0, 0)

        self.width_slider = QtWidgets.QSlider()
        self.width_slider.setRange(656, 1536)
        self.width_slider.setSingleStep(1)
        self.width_slider.setPageStep(1)
        self.width_slider.setOrientation(Qt.Orientation.Horizontal)
        image_sliders_layout.addWidget(self.width_slider, 0, 1)

        self.image_width_value_label = QtWidgets.QLabel()
        image_sliders_layout.addWidget(self.image_width_value_label, 0, 2)

        height_label = QtWidgets.QLabel("Height")
        image_sliders_layout.addWidget(height_label, 1, 0)

        self.height_slider = QtWidgets.QSlider()
        self.height_slider.setRange(656, 1536)
        self.height_slider.setSingleStep(1)
        self.height_slider.setPageStep(1)
        self.height_slider.setOrientation(Qt.Orientation.Horizontal)
        image_sliders_layout.addWidget(self.height_slider, 1, 1)

        self.image_height_value_label = QtWidgets.QLabel()
        image_sliders_layout.addWidget(self.image_height_value_label, 1, 2)

        self.width_slider.valueChanged.connect(self.on_slider_value_changed)
        self.height_slider.valueChanged.connect(self.on_slider_value_changed)

        main_layout.addLayout(image_sliders_layout)

        self.update()
        self.setLayout(main_layout)

    def on_slider_value_changed(self):
        slider = self.sender()
        current_value = slider.value()
        nearest_value = min(self.ALLOWED_VALUES, key=lambda x: abs(x - current_value))

        if slider == self.width_slider:
            self.image_width_value_label.setText(str(nearest_value))
            self.image_generation_data.image_width = nearest_value
        else:
            self.image_height_value_label.setText(str(nearest_value))
            self.image_generation_data.image_height = nearest_value

    def update(self):
        self.width_slider.setValue(self.image_generation_data.image_width)
        self.image_width_value_label.setText(
            f"{self.image_generation_data.image_width}"
        )
        self.height_slider.setValue(self.image_generation_data.image_height)
        self.image_height_value_label.setText(
            f"{self.image_generation_data.image_height}"
        )
