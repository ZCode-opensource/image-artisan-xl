import logging

from PyQt6.QtWidgets import (
    QMainWindow,
    QHBoxLayout,
    QFrame,
    QVBoxLayout,
    QStatusBar,
    QWidget,
    QApplication,
)
from PyQt6.QtCore import Qt, QSettings, QPropertyAnimation, QEasingCurve, QPoint, QTimer

from iartisanxl.app.title_bar import TitleBar
from iartisanxl.app.modules import MODULES
from iartisanxl.menu.left_menu import LeftMenu
from iartisanxl.windows.snackbar import SnackBar
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.configuration.preferences_dialog import PreferencesDialog
from iartisanxl.app.downloader_dialog import DownloaderDialog


class MainWindow(QMainWindow):
    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        *args,
        splash_timer_duration: int = 2000,
        snackbar_duration: int = 3000,
        snackbar_animation_duration: int = 300,
        **kwargs,
    ):
        super(MainWindow, self).__init__(*args, **kwargs)

        splash_timer = QTimer()
        splash_timer.singleShot(splash_timer_duration, self.close_splash)
        self.window_loaded = False
        self.timer_finished = False

        self.logger = logging.getLogger()

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Image Artisan XL")
        self.setMinimumSize(1300, 800)

        self.directories = directories
        self.preferences = preferences
        self.module = None

        self.settings = QSettings("ZCode", "ImageArtisanXL")

        self.settings.beginGroup("main_window")
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        windowState = self.settings.value("windowState")
        if windowState is not None:
            self.restoreState(windowState)
        self.settings.endGroup()

        self.settings.beginGroup("gui")
        self.gui_options = {
            "left_menu_expanded": self.settings.value("left_menu_expanded", True, type=bool),
            "current_module": self.settings.value("current_module", "Text to image", type=str),
        }
        self.settings.endGroup()

        self.init_ui()

        self.snackbar_queue = []
        self.snackbar_duration = snackbar_duration
        self.snackbar_animation = QPropertyAnimation(self.snackbar, b"pos")
        self.snackbar_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.snackbar_animation.setDuration(snackbar_animation_duration)
        self.snackbar_animation.finished.connect(self.on_snackbar_animation_finished)
        self.snackbar_hide_animation = False
        self.snackbar_closed = True
        self.snackbar_timer = QTimer()
        self.snackbar_timer.timeout.connect(self.hide_snackbar)

        self.preferences_dialog = None
        self.downloader_dialog = None

        self.load_modules()

        _, module_class = MODULES[self.gui_options.get("current_module")]
        self.load_module(module_class, self.gui_options.get("current_module"))

        self.window_loaded = True
        if self.timer_finished:
            QApplication.instance().close_splash()

    def close_splash(self):
        if self.window_loaded:
            QApplication.instance().close_splash()
        else:
            self.timer_finished = True

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = TitleBar(title="ImageArtisan XL")
        main_layout.addWidget(title_bar)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.left_menu = LeftMenu(self.gui_options)
        self.left_menu.open_preferences.connect(self.on_open_preferences)
        self.left_menu.open_downloader.connect(self.on_open_downloader)
        content_layout.addWidget(self.left_menu)

        workspace = QFrame()
        workspace.setObjectName("workspace")
        workspace.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.workspace_layout = QVBoxLayout()
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_layout.setSpacing(0)
        workspace.setLayout(self.workspace_layout)
        content_layout.addWidget(workspace)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 12)

        main_layout.addLayout(content_layout)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.snackbar = SnackBar(self)
        self.snackbar.closed.connect(self.on_snackbar_close)

    def closeEvent(self, event):
        self.settings.beginGroup("main_window")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.endGroup()

        self.settings.beginGroup("gui")
        self.settings.setValue("left_menu_expanded", self.gui_options.get("left_menu_expanded"))
        self.settings.setValue("current_module", self.gui_options.get("current_module"))
        self.settings.endGroup()

        if self.module is not None:
            self.module.close()

        for child in self.findChildren(QWidget):
            child.deleteLater()

        super().closeEvent(event)

    def load_modules(self):
        for label, (icon, module_class) in MODULES.items():
            self.left_menu.add_module_button(icon, label, module_class, self.load_module)

    def load_module(self, module_class, label):
        if self.workspace_layout.count() > 0:
            current_module: QWidget = self.workspace_layout.itemAt(0).widget()  # type: ignore
            current_module.close()
            self.workspace_layout.removeWidget(current_module)
            current_module.deleteLater()

        try:
            self.module = module_class(self.status_bar, self.show_snackbar, self.directories, self.preferences)
            self.workspace_layout.addWidget(self.module)
            self.gui_options["current_module"] = label
        except TypeError as module_error:
            self.logger.error("Error loading the module with this message: %s", str(module_error))
            self.logger.debug("TypeError exception", exc_info=True)
            self.show_snackbar(f"{module_error}")

    def show_snackbar(self, message):
        self.snackbar_queue.append(message)

        if self.snackbar_closed:
            self.snackbar_closed = False
            self.show_next_snackbar()

    def show_next_snackbar(self):
        if not self.snackbar_queue:
            return

        self.snackbar_closed = False
        message = self.snackbar_queue.pop(0)
        self.snackbar.message = message
        snackbar_x = (self.width() - self.snackbar.width()) // 2
        snackbar_y = self.height() - self.snackbar.height() - 40
        self.snackbar.show()

        self.snackbar_animation.setStartValue(QPoint(snackbar_x, snackbar_y + 100))
        self.snackbar_animation.setEndValue(QPoint(snackbar_x, snackbar_y))
        self.snackbar_animation.start()

    def hide_snackbar(self):
        self.snackbar_timer.stop()
        snackbar_x = (self.width() - self.snackbar.width()) // 2
        snackbar_y = self.height() - self.snackbar.height() - 40

        self.snackbar_animation.setStartValue(QPoint(snackbar_x, snackbar_y))
        self.snackbar_animation.setEndValue(QPoint(snackbar_x, snackbar_y + 100))
        self.snackbar_animation.start()

    def on_snackbar_animation_finished(self):
        if self.snackbar_hide_animation:
            self.snackbar_closed = True
            self.snackbar_hide_animation = False
            self.snackbar.hide()

        if self.snackbar_closed:
            self.show_next_snackbar()
            return

        self.snackbar_hide_animation = True
        self.snackbar_timer.start(self.snackbar_duration)

    def on_snackbar_close(self):
        self.snackbar_timer.stop()
        self.snackbar_hide_animation = False
        self.snackbar_closed = True
        self.snackbar.hide()

        self.show_next_snackbar()

    def on_open_preferences(self):
        self.preferences_dialog = PreferencesDialog(self.directories, self.preferences)
        self.preferences_dialog.show()

    def on_open_downloader(self):
        self.downloader_dialog = DownloaderDialog(self.directories)
        self.downloader_dialog.show()
