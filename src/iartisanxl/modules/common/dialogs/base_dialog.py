from PyQt6.QtWidgets import QDialog, QVBoxLayout, QSizeGrip
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.title_bar import TitleBar
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.prompt_window import PromptWindow
from iartisanxl.generation.generation_data_object import ImageGenData


class CustomSizeGrip(QSizeGrip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(5, 5)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0, 0))
        painter.setPen(QPen(QColor(0, 0, 0, 0)))
        painter.drawRect(event.rect())


class BaseDialog(QDialog):
    generation_updated = pyqtSignal()
    border_color = QColor("#ff6b6b6b")

    def __init__(
        self,
        directories: DirectoriesObject,
        title: str,
        image_generation_data: ImageGenData,
        image_viewer: ImageViewerSimple,
        prompt_window: PromptWindow,
        auto_generate_function: callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.directories = directories
        self.image_generation_data = image_generation_data
        self.image_viewer = image_viewer
        self.prompt_window = prompt_window
        self.auto_generate_function = auto_generate_function

        self.dialog_layout = QVBoxLayout()
        self.dialog_layout.setContentsMargins(0, 0, 0, 0)
        self.dialog_layout.setSpacing(0)

        title_bar = TitleBar(title=title, is_dialog=True)
        self.dialog_layout.addWidget(title_bar)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(1, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.dialog_layout.addLayout(self.main_layout)

        size_grip = CustomSizeGrip(self)
        self.dialog_layout.addWidget(
            size_grip,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        )

        self.setLayout(self.dialog_layout)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(self.width(), 0, self.width(), self.height())
        painter.drawLine(0, self.height(), self.width(), self.height())

    def dialog_raised(self):
        pass
