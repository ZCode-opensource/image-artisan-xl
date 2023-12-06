from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QCheckBox


class DownloadItem(QFrame):
    def __init__(self, title: str, description: str, destination_directory: str, destination_subdirectory: str, files: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName("download_item")

        self.setFixedHeight(120)

        self.title = title
        self.description = description
        self.destination_directory = destination_directory
        self.destination_subdirectory = destination_subdirectory
        self.files = files
        self.setDisabled(False)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        description_label = QLabel(self.description)
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)

        main_layout.addStretch()

        self.item_checkbox = QCheckBox(self.title)
        main_layout.addWidget(self.item_checkbox)

        self.downloaded_label = QLabel(f"{self.title} - Already downloaded")
        self.downloaded_label.setVisible(False)
        main_layout.addWidget(self.downloaded_label)

        self.setLayout(main_layout)

    def set_disabled(self):
        self.setDisabled(True)
        self.item_checkbox.setVisible(False)
        self.downloaded_label.setVisible(True)
