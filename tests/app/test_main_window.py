import sys
import unittest
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication

from iartisanxl.app.main_window import MainWindow
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.modules.base_module import BaseModule


app = QApplication(sys.argv)


class TestMainWindow(unittest.TestCase):
    def setUp(self):
        directories = DirectoriesObject(
            models_diffusers="",
            models_safetensors="",
            models_vaes="",
            models_loras="",
            models_controlnets="",
            models_t2i_adapters="",
            models_ip_adapters="",
            models_upscalers="",
            outputs_images="",
            outputs_loras="",
            datasets="",
        )

        preferences = PreferencesObject(
            intermediate_images=False,
            use_tomes=False,
            sequential_offload=False,
            model_offload=False,
            save_image_metadata=False,
            save_image_control_annotators=False,
            save_image_control_sources=False,
            hide_nsfw=False,
        )

        self.window = MainWindow(directories, preferences)

    def test_window_title(self):
        self.assertEqual(self.window.windowTitle(), "Image Artisan XL")

    def test_load_module(self):
        class ConcreteModule(BaseModule):
            def init_ui(self):
                pass

        # When the app starts by default loads the last module open or
        # the default one which is 'Text to Image'
        self.assertEqual(self.window.workspace_layout.count(), 1)
        initial_module = self.window.workspace_layout.itemAt(0).widget()

        self.window.load_module(ConcreteModule, "MockModule")

        # Assert the initial module has been replaced
        self.assertEqual(self.window.workspace_layout.count(), 1)
        self.assertIsNot(self.window.workspace_layout.itemAt(0).widget(), initial_module)

    def test_load_module_TypeError(self):
        class MockModule(BaseModule):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                raise TypeError("Mock TypeError")

            def init_ui(self):
                pass

        # Mock the logger.error and show_snackbar methods
        self.window.logger.error = MagicMock()
        self.window.show_snackbar = MagicMock()

        self.window.load_module(MockModule, "MockModule")

        # Assert the logger.error and show_snackbar methods were called with the correct arguments
        self.window.logger.error.assert_called_once_with(
            "Error loading the module with this message: %s",
            "Mock TypeError",
        )
        self.window.show_snackbar.assert_called_once_with("Mock TypeError")
