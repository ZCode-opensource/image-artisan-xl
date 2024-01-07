from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt, QSettings

from iartisanxl.configuration.base_setup_panel import BaseSetupPanel


class SelectDirectoryWidget(QWidget):
    def __init__(
        self,
        text: str,
        directory_text: str,
        directory_type: int,
        select_function: callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.text = text
        self.directory_text = directory_text
        self.directory_type = directory_type
        self.select_function = select_function

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        directory_widget = QWidget(self)
        directory_widget.setObjectName("setup_directory_widget")
        directory_layout = QVBoxLayout(directory_widget)

        explanation_label = QLabel(self.text)
        explanation_label.setWordWrap(True)
        directory_layout.addWidget(explanation_label)

        select_directory_button = QPushButton(f"Select {self.directory_text.lower()} directory")
        select_directory_button.clicked.connect(lambda: self.select_function(self.directory_type))
        select_directory_button.parent_widget = self
        directory_layout.addWidget(select_directory_button)

        path_layout = QHBoxLayout()
        path_label = QLabel(f"{self.directory_text} path: ")
        path_label.setFixedWidth(90)
        path_layout.addWidget(path_label)
        self.directory_label = QLabel("")
        path_layout.addWidget(self.directory_label, alignment=Qt.AlignmentFlag.AlignLeft)
        directory_layout.addLayout(path_layout)

        main_layout.addWidget(directory_widget)
        self.setLayout(main_layout)


class UtilitiesPanel(BaseSetupPanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_ui()

    def init_ui(self):
        upscaler_widget = SelectDirectoryWidget(
            "<html><body>Directory for the upscalers models.</body></html>",
            "Upscalers",
            1,
            self.select_directory,
        )
        self.main_layout.addWidget(upscaler_widget)

        self.main_layout.addStretch()

        button_layout = QHBoxLayout()
        finish_button = QPushButton("Back")
        finish_button.clicked.connect(self.on_back_step)
        button_layout.addWidget(finish_button)
        next_step_button = QPushButton("Optimizations")
        next_step_button.clicked.connect(self.on_next_step)
        button_layout.addWidget(next_step_button)

        self.buttons_widget.setLayout(button_layout)
        self.main_layout.addWidget(self.buttons_widget)

    def select_directory(self, dir_type):
        dialog = QFileDialog()
        options = (
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontUseNativeDialog
            | QFileDialog.Option.ReadOnly
            | QFileDialog.Option.HideNameFilterDetails
        )
        dialog.setOptions(options)

        selected_path = dialog.getExistingDirectory(None, "Select a directory")

        settings = QSettings("ZCode", "ImageArtisanXL")

        if dir_type == 1:
            self.directories.models_upscalers = selected_path
            settings.setValue("models_upscalers", selected_path)

        sender_button = self.sender()
        sender_button.parent_widget.directory_label.setText(selected_path)
