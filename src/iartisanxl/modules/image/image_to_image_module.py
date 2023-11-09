from PyQt6 import QtWidgets

from iartisanxl.modules.base_module import BaseModule


class ImageToImageModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()

    def init_ui(self):
        super().init_ui()
        main_layout = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel("Image to image")
        main_layout.addWidget(label)

        self.setLayout(main_layout)
