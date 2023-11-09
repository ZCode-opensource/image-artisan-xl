from PyQt6.QtWidgets import QDialog, QVBoxLayout

from iartisanxl.modules.common.image_label import ImageLabel


class FullScreenPreview(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dialog_layout = QVBoxLayout()
        self.image_preview_label = ImageLabel()
        dialog_layout.addWidget(self.image_preview_label)
        self.setLayout(dialog_layout)
