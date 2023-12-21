import os
import json
from collections import deque

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QPushButton, QProgressBar, QTabWidget, QWidget
from PyQt6.QtCore import Qt, QSettings, QUrl
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.download_item import DownloadItem
from iartisanxl.app.title_bar import TitleBar
from iartisanxl.windows.log_window import LogWindow


class DownloaderDialog(QDialog):
    def __init__(self, directories: DirectoriesObject, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Downloader")
        self.setMinimumSize(1050, 750)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("downloader_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.manager = QNetworkAccessManager(self)
        self.directories = directories
        self.response = None
        self.init_ui()

        self.download_queue = deque()
        self.current_item = None

        self.load_items()

    def init_ui(self):
        self.dialog_layout = QVBoxLayout()
        self.dialog_layout.setContentsMargins(0, 0, 0, 0)
        self.dialog_layout.setSpacing(0)

        title_bar = TitleBar(title="Downloader", is_dialog=True)
        self.dialog_layout.addWidget(title_bar)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        tab_widget = QTabWidget(self)
        tab_widget.setObjectName("tab_downloader")

        essentials_widget = QWidget()
        self.essentials_items_layout = QGridLayout()
        self.essentials_items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        essentials_widget.setLayout(self.essentials_items_layout)

        controlnets_widget = QWidget()
        self.controlnets_items_layout = QGridLayout()
        self.controlnets_items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        controlnets_widget.setLayout(self.controlnets_items_layout)

        t2i_widget = QWidget()
        self.t2i_items_layout = QGridLayout()
        self.t2i_items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        t2i_widget.setLayout(self.t2i_items_layout)

        captions_widget = QWidget()
        self.captions_items_layout = QGridLayout()
        self.captions_items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        captions_widget.setLayout(self.captions_items_layout)

        tab_widget.addTab(essentials_widget, "Essentials")
        tab_widget.addTab(controlnets_widget, "ControlNet")
        tab_widget.addTab(t2i_widget, "T2I Adapters")
        tab_widget.addTab(captions_widget, "Captions")
        self.main_layout.addWidget(tab_widget)

        sdxl_download_button = QPushButton("Download")
        sdxl_download_button.clicked.connect(self.on_start_download)
        self.main_layout.addWidget(sdxl_download_button)

        self.log_window = LogWindow()
        self.main_layout.addWidget(self.log_window)

        self.progress_bar = QProgressBar()
        self.main_layout.addWidget(self.progress_bar)

        self.main_layout.setStretch(0, 8)
        self.main_layout.setStretch(1, 0)
        self.main_layout.setStretch(2, 4)
        self.main_layout.setStretch(3, 0)

        self.dialog_layout.addLayout(self.main_layout)
        self.setLayout(self.dialog_layout)

    def closeEvent(self, event):
        self.settings.beginGroup("downloader_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()
        super().closeEvent(event)

    def load_items(self):
        with open("configs/download_list.json", encoding="utf-8") as json_file:
            data = json.load(json_file)
            layouts = {
                "essential_items": self.essentials_items_layout,
                "controlnet_items": self.controlnets_items_layout,
                "t2i_items": self.t2i_items_layout,
                "captions_items": self.captions_items_layout,
            }

            for category, layout in layouts.items():
                row = 0
                col = 0
                for item in data[category]:
                    try:
                        download_item = DownloadItem(
                            item["title"], item["description"], item["destination_directory"], item["destination_subdirectory"], item["files"]
                        )

                        final_directory = self.make_final_directory(item["destination_directory"], item["destination_subdirectory"])

                        if final_directory is not None:
                            check_directory = item.get("check_directory", None)

                            if check_directory is not None:
                                final_directory = os.path.join(final_directory, check_directory)

                            if os.path.isdir(final_directory):
                                download_item.set_disabled()

                        layout.addWidget(download_item, row, col)

                        col += 1
                        if col > 2:
                            col = 0
                            row += 1
                    except KeyError:
                        continue

    def make_final_directory(self, destination_directory, destination_subdirectory):
        root_directory = None
        final_directory = None

        if destination_directory == "app_models":
            root_directory = "models"
        elif destination_directory == "vae":
            root_directory = self.directories.vaes
        elif destination_directory == "diffusers":
            root_directory = self.directories.models_diffusers
        elif destination_directory == "controlnets":
            root_directory = self.directories.models_controlnets
        elif destination_directory == "t2i_adapters":
            root_directory = self.directories.models_t2i_adapters

        if root_directory is not None:
            if root_directory is not None:
                final_directory = os.path.join(root_directory, destination_subdirectory)

        return final_directory

    def on_start_download(self):
        layouts = [self.essentials_items_layout, self.controlnets_items_layout, self.t2i_items_layout, self.captions_items_layout]
        for layout in layouts:
            for i in range(layout.count()):
                item = layout.itemAt(i).widget()
                if isinstance(item, DownloadItem):
                    if item.item_checkbox.isChecked() and item.isEnabled():
                        self.download_queue.append(item)
        self.download_next_item()

    def download_next_item(self):
        if self.current_item and not self.current_item.files:
            self.current_item.set_disabled()
            self.current_item = None

        if not self.current_item and self.download_queue:
            self.current_item = self.download_queue.popleft()
            self.log_window.add_message(f"Downloading {self.current_item.title}...")

        if self.current_item:
            file = self.current_item.files.pop(0)
            self.download_item_start(self.current_item.destination_directory, self.current_item.destination_subdirectory, file)

    def download_item_start(self, destination_directory: str, destination_subdirectory: str, file: dict):
        final_directory = self.make_final_directory(destination_directory, destination_subdirectory)
        if final_directory is not None:
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)

            file_directory = file.get("destination_directory", None)
            if file_directory is not None:
                final_directory = os.path.join(final_directory, file_directory)
                if not os.path.exists(final_directory):
                    os.makedirs(final_directory)

            self.on_download(final_directory, file["file"], file["url"])

    def on_download(self, directory: str, file: str, url: str):
        self.log_window.add_message(f"Downloading {file}...")
        self.response = self.manager.get(QNetworkRequest(QUrl(url)))
        self.response.downloadProgress.connect(self.on_progress)
        self.response.finished.connect(lambda: self.on_finished(directory, file))

    def on_progress(self, bytesReceived, bytesTotal):
        MB = 1024 * 1024
        self.progress_bar.setMaximum(bytesTotal // MB)
        self.progress_bar.setValue(bytesReceived // MB)

    def on_finished(self, directory: str, file: str):
        if self.response.error() != QNetworkReply.NetworkError.NoError:
            print(f"Download error: {self.response.errorString()}")
            self.log_window.append_error(f"Download error: {self.response.errorString()}")
        else:
            path = os.path.join(directory, file)
            with open(path, "wb") as f:
                f.write(self.response.readAll().data())
            self.response.deleteLater()
            self.log_window.append_success("done.")
            self.download_next_item()
