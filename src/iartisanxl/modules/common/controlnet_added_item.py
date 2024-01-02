from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal

from iartisanxl.generation.controlnet_data_object import ControlNetDataObject
from iartisanxl.buttons.remove_button import RemoveButton
from iartisanxl.modules.common.image.image_processor import ImageProcessor


class ControlNetAddedItem(QWidget):
    remove_clicked = pyqtSignal(object)
    edit_clicked = pyqtSignal(object)
    enabled = pyqtSignal(int, bool)

    def __init__(self, controlnet: ControlNetDataObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlnet = controlnet
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        upper_layout = QHBoxLayout()

        self.enabled_checkbox = QCheckBox(self.controlnet.adapter_type)
        self.enabled_checkbox.setChecked(self.controlnet.enabled)
        self.enabled_checkbox.stateChanged.connect(self.on_check_enabled)
        upper_layout.addWidget(self.enabled_checkbox)

        remove_button = RemoveButton()
        remove_button.setFixedSize(20, 20)
        remove_button.clicked.connect(lambda: self.remove_clicked.emit(self))
        upper_layout.addWidget(remove_button)

        upper_layout.setStretch(0, 1)
        upper_layout.setStretch(1, 0)

        image_processor = ImageProcessor()

        lower_layout = QHBoxLayout()
        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(lambda: self.edit_clicked.emit(self.controlnet))
        lower_layout.addWidget(edit_button)
        self.source_thumb = QLabel()
        self.source_thumb.setFixedSize(80, 80)
        image_processor.set_pillow_image(self.controlnet.source_image_thumb)
        self.source_thumb.setPixmap(image_processor.get_qpixmap())
        lower_layout.addWidget(self.source_thumb)
        self.annotator_thumb = QLabel()
        self.annotator_thumb.setFixedSize(80, 80)
        image_processor.set_pillow_image(self.controlnet.annotator_image_thumb)
        self.annotator_thumb.setPixmap(image_processor.get_qpixmap())
        lower_layout.addWidget(self.annotator_thumb)

        main_layout.addLayout(upper_layout)
        main_layout.addLayout(lower_layout)

        self.setLayout(main_layout)

    def on_check_enabled(self):
        self.enabled.emit(self.controlnet.adapter_id, self.enabled_checkbox.isChecked())
