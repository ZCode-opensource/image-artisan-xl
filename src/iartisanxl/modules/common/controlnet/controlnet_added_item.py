from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from iartisanxl.buttons.remove_button import RemoveButton
from iartisanxl.modules.common.controlnet.controlnet_data_object import ControlNetDataObject


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
        self.enabled_checkbox = QCheckBox()
        self.enabled_checkbox.stateChanged.connect(self.on_check_enabled)
        upper_layout.addWidget(self.enabled_checkbox)

        remove_button = RemoveButton()
        remove_button.setFixedSize(20, 20)
        remove_button.clicked.connect(lambda: self.remove_clicked.emit(self))
        upper_layout.addWidget(remove_button)

        upper_layout.setStretch(0, 1)
        upper_layout.setStretch(1, 0)

        lower_layout = QHBoxLayout()
        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(lambda: self.edit_clicked.emit(self.controlnet))
        lower_layout.addWidget(edit_button)
        self.source_thumb = QLabel()
        self.source_thumb.setFixedSize(80, 80)
        lower_layout.addWidget(self.source_thumb)
        self.preprocessor_thumb = QLabel()
        self.preprocessor_thumb.setFixedSize(80, 80)
        lower_layout.addWidget(self.preprocessor_thumb)

        main_layout.addLayout(upper_layout)
        main_layout.addLayout(lower_layout)

        self.setLayout(main_layout)

    def update_ui(self):
        self.enabled_checkbox.setText(self.controlnet.adapter_name)
        self.enabled_checkbox.setChecked(self.controlnet.enabled)

        source_thumb_pixmap = QPixmap(self.controlnet.source_image.image_thumb)
        self.source_thumb.setPixmap(source_thumb_pixmap)

        preprocessor_thumb_pixmap = QPixmap(self.controlnet.preprocessor_image.image_thumb)
        self.preprocessor_thumb.setPixmap(preprocessor_thumb_pixmap)

    def on_check_enabled(self):
        self.enabled.emit(self.controlnet.adapter_id, self.enabled_checkbox.isChecked())
