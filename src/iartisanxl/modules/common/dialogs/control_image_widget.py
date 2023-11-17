from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImageReader, QPixmap

from iartisanxl.modules.common.image_editor import ImageEditor
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.generation.generation_data_object import ImageGenData


class ControlImageWidget(QWidget):
    def __init__(
        self,
        text: str,
        image_viewer: ImageViewerSimple,
        image_generation_data: ImageGenData,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setObjectName("control_image_widget")
        self.text = text
        self.image_viewer = image_viewer
        self.image_generation_data = image_generation_data
        self.image_path = ""

        self.setAcceptDrops(True)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        source_text_label = QLabel(self.text)
        top_layout.addWidget(source_text_label, alignment=Qt.AlignmentFlag.AlignCenter)

        reset_image_button = QPushButton("Reset")
        top_layout.addWidget(reset_image_button)
        undo_button = QPushButton("Undo")

        top_layout.addWidget(undo_button)
        redo_button = QPushButton("Redo")
        top_layout.addWidget(redo_button)
        blank_image_button = QPushButton("Blank")
        top_layout.addWidget(blank_image_button)
        current_image_button = QPushButton("Current")
        top_layout.addWidget(current_image_button)
        load_image_button = QPushButton("Load")
        top_layout.addWidget(load_image_button)

        main_layout.addLayout(top_layout)

        self.image_editor = ImageEditor()
        main_layout.addWidget(self.image_editor)

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
        self.setLayout(main_layout)

        reset_image_button.clicked.connect(self.image_editor.clear_and_restore)
        undo_button.clicked.connect(self.image_editor.undo)
        redo_button.clicked.connect(self.image_editor.redo)
        current_image_button.clicked.connect(self.set_current_image)
        blank_image_button.clicked.connect(self.set_blank_image)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.image_editor.drop_lightbox.show()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.image_editor.drop_lightbox.hide()
        event.accept()

    def dropEvent(self, event):
        self.image_editor.drop_lightbox.hide()

        for url in event.mimeData().urls():
            path = url.toLocalFile()

            reader = QImageReader(path)

            if reader.canRead():
                self.image_path = path
                pixmap = QPixmap(self.image_path)
                self.image_editor.set_pixmap(pixmap)

    def set_current_image(self):
        if self.image_viewer.pixmap_item is not None:
            pixmap = self.image_viewer.pixmap_item.pixmap()
            self.image_editor.set_pixmap(pixmap)

    def set_blank_image(self):
        width = self.image_generation_data.image_width
        height = self.image_generation_data.image_height
        self.image_editor.set_white_pixmap(width, height)