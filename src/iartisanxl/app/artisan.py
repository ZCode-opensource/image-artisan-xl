import ctypes

from importlib.resources import files

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, QSettings

from iartisanxl.app.main_window import MainWindow
from iartisanxl.configuration.initial_setup_dialog import InitialSetupDialog
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject


class ArtisanApplication(QApplication):
    APP_ICON = files("iartisanxl.theme.icons").joinpath("iartisan_icon.ico")
    SPLASH_IMG = str(files("iartisanxl.theme.images").joinpath("iartisan_splash.webp"))

    def __init__(self, *args, **kwargs):
        myappid = "zcode.imageartisanxl.010"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        super(ArtisanApplication, self).__init__(*args, **kwargs)

        style_data = files("iartisanxl.theme").joinpath("stylesheet.qss").read_bytes()
        stylesheet = style_data.decode("utf-8")
        self.setStyleSheet(stylesheet)

        self.setWindowIcon(QIcon(str(self.APP_ICON)))

        self.directories = None
        self.preferences = None
        self.window = None
        self.splash = None

        if not self.check_initial_setup():
            self.dialog = InitialSetupDialog(self.directories, self.preferences)
            self.dialog.exec()

        self.load_main_window()

    def check_initial_setup(self):
        settings = QSettings("ZCode", "ImageArtisanXL")

        intermediate_images = settings.value("intermediate_images", False, type=bool)
        use_tomes = settings.value("use_tomes", False, type=bool)
        sequential_offload = settings.value("sequential_offload", False, type=bool)
        model_offload = settings.value("model_offload", False, type=bool)
        save_image_metadata = settings.value("save_image_metadata", False, type=bool)
        save_image_control_annotators = settings.value("save_image_control_annotators", False, type=bool)
        save_image_control_sources = settings.value("save_image_control_sources", False, type=bool)
        hide_nsfw = settings.value("hide_nsfw", True, type=bool)

        self.preferences = PreferencesObject(
            intermediate_images=intermediate_images,
            use_tomes=use_tomes,
            sequential_offload=sequential_offload,
            model_offload=model_offload,
            save_image_metadata=save_image_metadata,
            save_image_control_annotators=save_image_control_annotators,
            save_image_control_sources=save_image_control_sources,
            hide_nsfw=hide_nsfw,
        )

        models_diffusers = settings.value("models_diffusers", None, type=str)
        models_safetensors = settings.value("models_safetensors", None, type=str)
        vaes = settings.value("vaes", None, type=str)
        models_loras = settings.value("models_loras", None, type=str)
        models_controlnets = settings.value("models_controlnets", None, type=str)
        models_t2i_adapters = settings.value("models_t2i_adapters", None, type=str)
        models_ip_adapters = settings.value("models_ip_adapters", None, type=str)
        outputs_images = settings.value("outputs_images", None, type=str)

        self.directories = DirectoriesObject(
            models_diffusers=models_diffusers,
            models_safetensors=models_safetensors,
            vaes=vaes,
            models_loras=models_loras,
            models_controlnets=models_controlnets,
            models_t2i_adapters=models_t2i_adapters,
            models_ip_adapters=models_ip_adapters,
            outputs_images=outputs_images,
        )

        if any(
            not v
            for v in [
                models_diffusers,
                models_safetensors,
                vaes,
                models_loras,
                models_controlnets,
                models_t2i_adapters,
                models_ip_adapters,
                outputs_images,
            ]
        ):
            return False
        return True

    def close_splash(self):
        if self.splash:
            self.splash.close()
            self.window.show()

    def load_main_window(self):
        splash_pix = QPixmap(self.SPLASH_IMG)
        self.splash = QSplashScreen(splash_pix)
        self.splash.showMessage(
            "Loading...",
            alignment=Qt.AlignmentFlag.AlignBottom,
            color=Qt.GlobalColor.white,
        )

        self.splash.show()

        self.window = MainWindow(self.directories, self.preferences)
