from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from iartisanxl.buttons.remove_button import RemoveButton
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject


class IPAdapterAddedItem(QWidget):
    remove_clicked = pyqtSignal(object)
    edit_clicked = pyqtSignal(object)
    enabled = pyqtSignal(int, bool)

    def __init__(self, adapter: IPAdapterDataObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter = adapter
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
        edit_button.clicked.connect(lambda: self.edit_clicked.emit(self.adapter))
        lower_layout.addWidget(edit_button)
        self.image_thumb = QLabel()
        self.image_thumb.setFixedSize(80, 80)

        thumb_pixmap = QPixmap(self.adapter.images[0].thumb)
        self.image_thumb.setPixmap(thumb_pixmap)
        lower_layout.addWidget(self.image_thumb)

        main_layout.addLayout(upper_layout)
        main_layout.addLayout(lower_layout)

        self.setLayout(main_layout)

    def update_ui(self):
        self.enabled_checkbox.setText(self.adapter.adapter_name)
        self.enabled_checkbox.setChecked(self.adapter.enabled)

        thumb_pixmap = QPixmap(self.adapter.images[0].thumb)
        self.image_thumb.setPixmap(thumb_pixmap)

    def on_check_enabled(self):
        self.enabled.emit(self.adapter.adapter_id, self.enabled_checkbox.isChecked())
