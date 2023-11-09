import unittest
from unittest.mock import patch, Mock, MagicMock
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication

from iartisanxl.app.artisan import ArtisanApplication
from iartisanxl.app.main_window import MainWindow
from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.modules.base_module import BaseModule


app = ArtisanApplication([])


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directories = DirectoriesObject(
            models_diffusers="",
            models_safetensors="",
            vaes="",
            models_loras="",
            outputs_images="",
        )

        cls.preferences = PreferencesObject(
            intermediate_images=False,
            use_tomes=False,
            sequential_offload=False,
            model_offload=False,
        )

        cls.window = MainWindow(cls.directories, cls.preferences)

    def test_window_title(self):
        self.assertEqual(self.window.windowTitle(), "Image Artisan XL")

    def test_window_close(self):
        self.window.show()
        event = QCloseEvent()
        self.window.closeEvent(event)
        settings = QSettings("ZCode", "ImageArtisanXL")
        settings.beginGroup("main_window")
        self.assertIsNotNone(settings.value("geometry"))
        self.assertIsNotNone(settings.value("windowState"))
        settings.endGroup()

    def test_show_and_hide_snackbar(self):
        with patch.object(ArtisanApplication, "instance") as mock_instance:
            mock_instance.return_value.close_splash = Mock()

            window = MainWindow(
                self.directories,
                self.preferences,
                snackbar_duration=0,
                snackbar_animation_duration=0,
            )

            window.show()

            self.assertFalse(window.snackbar.isVisible())

            window.show_snackbar("Test message")
            self.assertEqual(window.snackbar.message, "Test message")
            self.assertTrue(window.snackbar.isVisible())

            app.processEvents()  # Process any pending events

            self.assertFalse(window.snackbar.isVisible())

    def test_show_and_close_snackbar(self):
        with patch.object(ArtisanApplication, "instance") as mock_instance:
            mock_instance.return_value.close_splash = Mock()

            window = MainWindow(
                self.directories,
                self.preferences,
                snackbar_duration=500,
                snackbar_animation_duration=0,
            )

            window.show()
            window.show_snackbar("Test message")

            window.on_snackbar_close()
            self.assertFalse(window.snackbar.isVisible())

    def test_close_splash_called_after_timer(self):
        with patch.object(QApplication, "instance") as mock_instance:
            mock_instance.return_value.close_splash = Mock()

            window = MainWindow(
                self.directories, self.preferences, splash_timer_duration=100
            )

            # Process events for a duration longer than splash_timer_duration
            timer = QTimer()
            timer.setSingleShot(True)
            timer.start(200)

            while timer.isActive():
                app.processEvents()

            self.assertTrue(mock_instance.return_value.close_splash.called)

    def test_timer_end_before_app_load(self):
        with patch.object(ArtisanApplication, "instance") as mock_instance:
            mock_instance.return_value.close_splash = Mock()

            window = MainWindow(
                self.directories, self.preferences, splash_timer_duration=100
            )
            window.window_loaded = False

            timer = QTimer()
            timer.setSingleShot(True)  # Ensure the timer only runs once
            timer.start(200)
            while timer.isActive():
                app.processEvents()

            # At this point, the timer has finished but the window hasn't loaded yet
            # So, close_splash should not have been called
            self.assertTrue(window.timer_finished)
            self.assertFalse(mock_instance.return_value.close_splash.called)

    def test_load_module(self):
        class ConcreteModule(BaseModule):
            def init_ui(self):
                pass

        window = MainWindow(self.directories, self.preferences)

        # When the app starts by default loads the last module open or
        # the default one which is 'Text to Image'
        self.assertEqual(window.workspace_layout.count(), 1)
        initial_module = window.workspace_layout.itemAt(0).widget()

        window.load_module(ConcreteModule, "MockModule")

        # Assert the initial module has been replaced
        self.assertEqual(window.workspace_layout.count(), 1)
        self.assertIsNot(window.workspace_layout.itemAt(0).widget(), initial_module)

    def test_load_module_TypeError(self):
        class MockModule(BaseModule):
            def __init__(self, *args, **kwargs):
                raise TypeError("Mock TypeError")

            def init_ui(self):
                pass

        window = MainWindow(self.directories, self.preferences)

        # Mock the logger.error and show_snackbar methods
        window.logger.error = MagicMock()
        window.show_snackbar = MagicMock()

        window.load_module(MockModule, "MockModule")

        # Assert the logger.error and show_snackbar methods were called with the correct arguments
        window.logger.error.assert_called_once_with(
            "Error loading the module with this message: %s",
            "Mock TypeError",
        )
        window.show_snackbar.assert_called_once_with("Mock TypeError")

    def test_on_open_preferences(self):
        window = MainWindow(self.directories, self.preferences)

        window.on_open_preferences()

        self.assertTrue(window.preferences_dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
